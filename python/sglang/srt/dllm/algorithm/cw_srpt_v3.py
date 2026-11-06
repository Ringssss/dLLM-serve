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
        # Quality guard
        self.min_conf_threshold = cfg.get("min_conf", 0.3)   # quality guard floor
        # Stability gate: commit only positions whose argmax survived a
        # forward, or that are already decisive.
        self.stable_gate = bool(cfg.get("stable_gate", False))
        self.stable_threshold = cfg.get("stable_threshold", 0.95)
        # Rows per fp32 confidence chunk; bounds scratch memory at high batch.
        self.confidence_chunk_rows = int(cfg.get("confidence_chunk_rows", 16))
        # Commit any position at or above this confidence immediately, in
        # addition to the paced stride. None disables the rule.
        self.saturate_threshold = cfg.get("saturate_threshold", None)
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

        # State from the last executed forward, reused for the frontier
        # writeback so that no extra forward is needed after the loop.
        ran_forward = False
        logits_output = None
        can_run_cuda_graph = False
        x0 = None
        x0_p = None
        prev_x0 = None

        # Fixed top-k width: avoids a device sync to size the top-k.
        k = min(self.max_unmask, bs)

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
            ran_forward = True

            # Reshape logits
            full_logits = logits_output.full_logits
            logits_view = full_logits.view(batch_size, bs, -1)

            # Vectorized argmax + confidence. The confidence is the softmax
            # probability of the argmax, obtained from the log-partition
            # instead of materialising a [batch, block, vocab] probability
            # tensor on every iteration. Rows are processed in chunks so the
            # fp32 scratch stays bounded as concurrency grows: at 256 rows an
            # unchunked fp32 copy alone is ~5 GiB and OOMs a 40 GiB A100.
            x0 = logits_view.argmax(dim=-1)
            x0_p = torch.empty(x0.shape, dtype=torch.float32, device=x0.device)
            step = max(1, self.confidence_chunk_rows)
            for lo in range(0, batch_size, step):
                hi = min(lo + step, batch_size)
                rows_f = logits_view[lo:hi].float()
                rows_x0 = x0[lo:hi]
                rows_max = rows_f.gather(-1, rows_x0.unsqueeze(-1)).squeeze(-1)
                rows_lse = torch.logsumexp(rows_f, dim=-1)
                x0_p[lo:hi] = (rows_max - rows_lse).exp()

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

            # Stability gate: a position may commit only if the model would
            # produce the same token again (argmax unchanged since the last
            # forward) or if it is already decisive. Positions committed from
            # a stale logit are what turn "#### 18" into "####188": the token
            # is frozen before the surrounding context can revise it.
            if self.stable_gate and prev_x0 is not None:
                committable = (x0 == prev_x0) | (x0_p >= self.stable_threshold)
            else:
                committable = torch.ones_like(cur_masks, dtype=torch.bool)

            # Pad confidence for topk (done blocks get -inf)
            conf_for_topk = torch.where(
                block_done.unsqueeze(1),
                torch.tensor(-float('inf'), device=input_ids.device),
                confidence)

            topk_val, topk_idx = torch.topk(conf_for_topk, k=k, dim=1)

            # Build transfer mask: take top-stride tokens per block
            rank = torch.arange(k, device=input_ids.device).unsqueeze(0)
            take = (rank < stride.unsqueeze(1)) & (topk_val > -float('inf'))

            transfer = torch.zeros_like(cur_masks, dtype=torch.bool)
            transfer.scatter_(1, topk_idx, take)

            # Threshold saturation beyond the paced stride: positions the
            # model already resolves decisively are committed without waiting
            # for their stride turn. The baseline commits exactly these
            # positions, so this can only remove forwards, never add risk.
            if self.saturate_threshold is not None:
                transfer = transfer | (x0_p >= self.saturate_threshold)

            # Apply quality guard and stability gate
            transfer = transfer & safe & committable & cur_masks & ~block_done.unsqueeze(1)

            # Ensure minimum progress: if transfer is empty, force the
            # best committable position. Evaluated unconditionally so the
            # host never blocks on a device-to-host copy mid-loop; the
            # scatter below is a no-op when nothing needs forcing.
            n_transfer = transfer.sum(dim=1)
            need_force = (n_transfer == 0) & (n_masked > 0) & ~block_done
            gated = torch.where(
                committable & cur_masks & ~block_done.unsqueeze(1),
                confidence,
                torch.full_like(confidence, -float('inf')))
            best = torch.where(gated > -float('inf'), gated, confidence).argmax(dim=1)
            transfer[need_force, best[need_force]] = True

            # Apply
            block_view[transfer] = x0_merged[transfer]

            # Remember this forward's predictions for the stability gate.
            if self.stable_gate:
                prev_x0 = x0

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

        # ─── Reuse the last iteration's logits as output logits ──
        # The loop below has already computed everything the frontier
        # writeback needs, so a second forward would only repeat work.
        # Falling back to one extra forward keeps the prefill fast path
        # (no masks) and any early-broken batch correct.
        if not ran_forward:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output = out.logits_output
            can_run_cuda_graph = out.can_run_graph
            x0_p = None
            x0 = None
            final_masks = (block_view == self.mask_id)

        # Build per-request output token lists
        next_token_ids = block_view
        next_token_ids_list = [
            # PAS returns a block snapshot, including holes, for lifecycle
            # writeback. The scheduler publishes it only once fully resolved.
            next_token_ids[i, :] for i in range(batch_size)
        ]

        # ── Frontier writeback via return value ──────────────────
        # Store per-block frontier info in logits_output for scheduler to read
        # This avoids needing direct Req access from the algorithm
        if hasattr(logits_output, '__dict__'):
            final_masks = (block_view == self.mask_id)
            final_n_masked = final_masks.sum(dim=1)

            # Mean confidence over unresolved positions, reusing the
            # confidence tensor from the last forward (no extra softmax).
            if x0_p is not None and x0 is not None:
                valid_final = final_masks & (x0 != self.mask_id)
                weighted = torch.where(valid_final, x0_p, torch.zeros_like(x0_p)).sum(dim=1)
                count = valid_final.sum(dim=1).clamp(min=1)
                final_conf = weighted / count
            else:
                final_conf = torch.zeros(batch_size, device=input_ids.device)

            logits_output.dllm_frontier_n_masked = final_n_masked.cpu().tolist()
            logits_output.dllm_frontier_confidence = final_conf.cpu().tolist()
            logits_output.dllm_iters_run = iters_run

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = CW_SRPT_V3
