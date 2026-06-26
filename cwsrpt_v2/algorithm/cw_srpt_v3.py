"""
CW-SRPT V3: Frontier-Guided Forward-Pass Scheduling for dLLM Serving.

Three-pronged system for managing the dLLM request lifecycle:

  Admission:   Frontier-aware slot allocation (in scheduler.py)
  Execution:   Adaptive denoising stride — target-based top-k instead of threshold
  Termination: Active-set early break — return to scheduler when >50% blocks done

This is a drop-in replacement for LowConfidence / CW_SRPT_V2.

Launch:
  python -m sglang.launch_server --model-path /path/to/LLaDA2.0-mini \
    --dllm-algorithm CW_SRPT_V3 --max-running-requests 16
"""
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class CW_SRPT_V3(DllmAlgorithm):
    """Frontier-guided dLLM algorithm with stride control and active-set management."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        cfg = config.algorithm_config
        # Stride controller params
        self.target_iters = cfg.get("target_iters", 8)      # target iterations per block
        self.min_unmask = cfg.get("min_unmask", 2)           # min tokens per iter
        self.max_unmask = cfg.get("max_unmask", 16)          # max tokens per iter
        self.min_conf_threshold = cfg.get("min_conf", 0.3)   # quality guard floor
        # Active-set early break
        self.early_break_ratio = cfg.get("early_break_ratio", 0.5)  # break when >50% done
        # Stall handling
        self.stall_boost = cfg.get("stall_boost", 1)  # extra tokens when stalled

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        bs = self.block_size
        input_ids = forward_batch.input_ids  # [batch_size * block_size]

        # Fast path: no mask tokens (prefill / all converged)
        mask_index = (input_ids == self.mask_id)
        if mask_index.sum().item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        # Reshape to [batch_size, block_size]
        block_view = input_ids.view(batch_size, bs)

        # Record start positions
        block_masks = (block_view == self.mask_id)
        start_list = (bs - block_masks.sum(dim=1)).tolist()

        # Per-block tracking
        stall_counts = torch.zeros(batch_size, device=input_ids.device)
        block_done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        iters_run = 0

        # ─── Main denoising loop ─────────────────────────────────
        for iter_idx in range(bs):
            # ── Termination: active-set early break ──────────────
            # If >50% of blocks are done, break and return to scheduler
            # so that scheduler can compact done blocks out and refill
            if iter_idx > 0:
                done_ratio = block_done.float().mean().item()
                if done_ratio >= self.early_break_ratio and not block_done.all():
                    break
            if block_done.all():
                break

            # Forward
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output = out.logits_output
            can_run_cuda_graph = out.can_run_graph
            iters_run += 1

            # Reshape logits
            full_logits = logits_output.full_logits
            logits_view = full_logits.view(batch_size, bs, -1)

            # Vectorized argmax + confidence
            x0 = logits_view.argmax(dim=-1)
            probs = F.softmax(logits_view.float(), dim=-1)
            x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

            cur_masks = (block_view == self.mask_id)
            valid = cur_masks & (x0 != self.mask_id)
            x0_merged = torch.where(cur_masks, x0, block_view)
            confidence = torch.where(
                valid, x0_p,
                torch.tensor(-float('inf'), device=input_ids.device))

            # ── Execution: adaptive denoising stride ─────────────
            # Instead of "unmask everything above threshold",
            # compute target stride: how many tokens SHOULD we unmask?
            n_masked = cur_masks.sum(dim=1).float()  # [batch_size]

            # Remaining iterations estimate (at least 1 to avoid div0)
            remaining_iters = torch.clamp(
                (self.target_iters - iter_idx) * torch.ones_like(n_masked),
                min=1.0)

            # Stride = ceil(n_masked / remaining_iters)
            stride = torch.ceil(n_masked / remaining_iters).long()

            # Stall boost: stuck blocks get extra stride
            stride = stride + (stall_counts > 0).long() * self.stall_boost

            # Clamp stride
            stride = stride.clamp(min=self.min_unmask, max=self.max_unmask)

            # Don't stride more than what's masked
            stride = torch.min(stride, n_masked.long())

            # Quality guard: only unmask tokens above min confidence
            safe = confidence >= self.min_conf_threshold

            # Vectorized top-k per block
            max_k = int(stride.max().item())
            if max_k <= 0:
                continue

            # Pad confidence for topk (done blocks get -inf)
            conf_for_topk = torch.where(
                block_done.unsqueeze(1),
                torch.tensor(-float('inf'), device=input_ids.device),
                confidence)

            topk_val, topk_idx = torch.topk(conf_for_topk, k=min(max_k, bs), dim=1)

            # Build transfer mask: take top-stride tokens per block
            rank = torch.arange(min(max_k, bs), device=input_ids.device).unsqueeze(0)
            take = (rank < stride.unsqueeze(1)) & (topk_val > -float('inf'))

            transfer = torch.zeros_like(cur_masks, dtype=torch.bool)
            transfer.scatter_(1, topk_idx, take)

            # Apply quality guard
            transfer = transfer & safe & cur_masks & ~block_done.unsqueeze(1)

            # Ensure minimum progress: if transfer is empty, force top-1
            n_transfer = transfer.sum(dim=1)
            need_force = (n_transfer == 0) & (n_masked > 0) & ~block_done
            if need_force.any():
                for bi in need_force.nonzero(as_tuple=True)[0]:
                    best = confidence[bi].argmax()
                    transfer[bi, best] = True

            # Apply
            block_view[transfer] = x0_merged[transfer]

            # ── Track convergence ────────────────────────────────
            new_masks = (block_view == self.mask_id)
            new_n_masked = new_masks.sum(dim=1)
            n_unmasked = cur_masks.sum(dim=1) - new_n_masked

            # Stall detection
            stall_counts = torch.where(
                n_unmasked > 0,
                torch.zeros_like(stall_counts),
                stall_counts + 1)

            # Block completion
            block_done = block_done | (new_n_masked == 0)

        # ─── Final forward for output logits ─────────────────────
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output = out.logits_output
        can_run_cuda_graph = out.can_run_graph

        # Build per-request output token lists
        next_token_ids = block_view
        next_token_ids_list = [
            next_token_ids[i, start_list[i]:] for i in range(batch_size)
        ]

        # ── Frontier writeback via return value ──────────────────
        # Store per-block frontier info in logits_output for scheduler to read
        # This avoids needing direct Req access from the algorithm
        if hasattr(logits_output, '__dict__'):
            final_masks = (block_view == self.mask_id)
            final_n_masked = final_masks.sum(dim=1)

            # Compute final mean confidence for unfinished blocks
            final_conf = torch.zeros(batch_size, device=input_ids.device)
            for i in range(batch_size):
                mi = final_masks[i]
                if mi.any() and hasattr(logits_output, 'full_logits'):
                    fl = logits_output.full_logits.view(batch_size, bs, -1)
                    x0i = fl[i].argmax(-1)
                    pi = F.softmax(fl[i].float(), -1).gather(-1, x0i.unsqueeze(-1)).squeeze(-1)
                    vi = mi & (x0i != self.mask_id)
                    if vi.any():
                        final_conf[i] = pi[vi].mean()

            logits_output.dllm_frontier_n_masked = final_n_masked.cpu().tolist()
            logits_output.dllm_frontier_confidence = final_conf.cpu().tolist()
            logits_output.dllm_iters_run = iters_run

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = CW_SRPT_V3
