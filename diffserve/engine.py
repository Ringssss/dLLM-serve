"""
DiffServe Engine: Core serving loop with batched CW-SRPT scheduling.

Supports two forward modes:
  - use_kv_cache=False (v3 path): full-sequence forward each iteration.
    Simple, correct, no per-request GPU state.
  - use_kv_cache=True (KV path): prefill prompt once into a shared KV buffer,
    then only forward the 32-token decode block each iteration.
    4-7x less compute per forward → higher throughput.

Both modes use the same CW-SRPT scheduling and threshold unmasking logic.
"""

import asyncio
import bisect
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import DiffServeConfig
from .request import DiffuseRequest
from .scheduler import SchedulingPolicy, pick_batch

logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    """Result returned when a request completes."""
    request_id: int
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    total_fwds: int
    latency_ms: float
    time_to_first_token_ms: float


class DiffServeEngine:
    """Core serving engine for dLLM with CW-SRPT scheduling.

    The engine manages the full request lifecycle:
      - Accepting new requests via add_request()
      - Running the continuous batching loop
      - Returning results via futures

    The engine runs in a single thread with the GPU. The async interface
    allows the HTTP server to submit requests without blocking.
    """

    def __init__(self, config: DiffServeConfig, model_runner, tokenizer):
        self.config = config
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)

        # Request management
        self._pending: deque = deque()  # (DiffuseRequest, Future) pairs
        self._active: List[DiffuseRequest] = []
        self._futures: Dict[int, asyncio.Future] = {}
        self._next_id: int = 0

        # Engine state
        self._running: bool = False
        self._loop_task: Optional[asyncio.Task] = None

        # KV cache pool (lazily allocated on first use)
        self._kv_pool: Optional[torch.Tensor] = None
        self._kv_slot_map: Dict[int, int] = {}  # request_id → slot index
        self._kv_free_slots: List[int] = []
        self._kv_max_slots: int = 0

        # Metrics
        self._total_requests: int = 0
        self._total_iters: int = 0
        self._total_latency_ms: float = 0.0
        self._total_tokens: int = 0

        # Confidence-Aware Arbiter
        self._arbiter = None
        if config.enable_arbiter:
            from .confidence_arbiter import ConfidenceAwareArbiter
            self._arbiter = ConfidenceAwareArbiter()

    # ─── KV Cache Pool Management ─────────────────────────────────

    def _init_kv_pool(self):
        """Lazily allocate the shared KV cache pool.

        Shape: [num_layers, 2, max_slots, num_kv_heads, max_seq_length, head_dim]
        Each slot holds one request's full KV cache.
        """
        if self._kv_pool is not None:
            return

        model = self.model_runner.model
        cfg = model.config
        num_layers = cfg.num_hidden_layers
        num_kv_heads = cfg.num_key_value_heads
        num_heads = cfg.num_attention_heads
        head_dim = cfg.hidden_size // num_heads
        max_seq = self.config.max_seq_length
        # Round up to next power-of-2 aligned cache boundary
        cache_align = 128
        n = math.ceil(max_seq / cache_align)
        next_pow2 = 1 << (n - 1).bit_length() if n > 1 else 1
        aligned_seq = next_pow2 * cache_align

        # Pool supports up to 256 concurrent requests
        self._kv_max_slots = 256
        tp_size = 1  # single GPU
        kv_heads_per_tp = max(1, num_kv_heads // tp_size)

        self._kv_pool = torch.zeros(
            num_layers, 2, self._kv_max_slots, kv_heads_per_tp,
            aligned_seq, head_dim,
            dtype=torch.bfloat16, device=self.device)
        self._kv_free_slots = list(range(self._kv_max_slots))
        self._kv_slot_map = {}
        logger.info(
            f"KV pool allocated: {self._kv_max_slots} slots, "
            f"{num_layers}L x {kv_heads_per_tp}H x {aligned_seq}S x {head_dim}D, "
            f"{self._kv_pool.element_size() * self._kv_pool.nelement() / 1e6:.0f} MB")

    def _alloc_kv_slot(self, req_id: int) -> int:
        """Allocate a KV cache slot for a request."""
        if not self._kv_free_slots:
            raise RuntimeError("KV cache pool exhausted")
        slot = self._kv_free_slots.pop()
        self._kv_slot_map[req_id] = slot
        # Zero out the slot
        self._kv_pool[:, :, slot].zero_()
        return slot

    def _free_kv_slot(self, req_id: int):
        """Release a KV cache slot back to the pool."""
        if req_id in self._kv_slot_map:
            slot = self._kv_slot_map.pop(req_id)
            self._kv_free_slots.append(slot)

    def _get_kv_slot(self, req_id: int) -> int:
        return self._kv_slot_map[req_id]

    # ─── Public API ───────────────────────────────────────────────

    def add_request(
        self,
        prompt_ids: torch.Tensor,
        gen_length: int = 128,
        threshold: Optional[float] = None,
    ) -> Tuple[int, asyncio.Future]:
        """Submit a new request for serving."""
        if threshold is None:
            threshold = self.config.threshold

        rid = self._next_id
        self._next_id += 1

        req = DiffuseRequest(
            id=rid,
            prompt_ids=prompt_ids,
            gen_length=gen_length,
            threshold=threshold,
            config=self.config,
        )
        req.init()

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._futures[rid] = future
        self._pending.append((req, future))

        return rid, future

    async def start(self):
        """Start the serving engine loop."""
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info(
            f"DiffServe engine started: policy={self.config.policy}, "
            f"max_batch={self.config.max_batch_size}, "
            f"kv_cache={self.config.use_kv_cache}")

    async def stop(self):
        """Stop the serving engine."""
        self._running = False
        if self._loop_task:
            await self._loop_task

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def metrics(self) -> dict:
        return {
            "total_requests": self._total_requests,
            "total_iters": self._total_iters,
            "active_requests": self.active_count,
            "pending_requests": self.pending_count,
            "mean_latency_ms": (
                self._total_latency_ms / max(self._total_requests, 1)),
            "total_tokens": self._total_tokens,
        }

    # ─── Synchronous serving (for benchmarks) ────────────────────

    def serve_batch_sync(
        self,
        requests: List[DiffuseRequest],
        arrivals_ms: List[float],
    ) -> Tuple[List[DiffuseRequest], float, dict]:
        """Run serving synchronously (blocking). Used by bench_online."""
        use_kv = self.config.use_kv_cache
        if use_kv:
            self._init_kv_pool()

        sorted_reqs = sorted(zip(requests, arrivals_ms), key=lambda x: x[1])
        pending = deque(sorted_reqs)
        active = []
        done = []
        clock = 0.0
        total_iters = 0
        total_padding_waste = 0

        policy = self.config.policy

        while pending or active:
            # Admit arrivals
            while pending and pending[0][1] <= clock:
                r, arr = pending.popleft()
                r.arrival_time = arr
                if r.first_token_time < 0:
                    r.first_token_time = clock
                if use_kv:
                    self._alloc_kv_slot(r.id)
                active.append(r)

            if not active:
                if pending:
                    clock = pending[0][1]
                    continue
                break

            # Schedule
            batch = pick_batch(active, self.config.max_batch_size, policy)

            # Arbiter: predict next batch size from confidence
            if self._arbiter is not None:
                arbiter_signal = self._arbiter.on_iteration(
                    active, batch, len(pending))

            # Track padding waste (only meaningful for no-KV mode)
            if not use_kv and len(batch) > 1:
                lens = [r.block_end for r in batch]
                max_l = max(lens)
                waste = sum(max_l - l for l in lens)
                total_padding_waste += waste

            # Execute: prefill unprefilled requests, then decode
            if use_kv:
                elapsed = self._run_kv_step(batch)
            else:
                elapsed = self._run_batched_iter(batch)
            clock += elapsed
            total_iters += 1

            # Process completions
            newly_done = [r for r in batch if r.done]
            for r in newly_done:
                r.finish_time = clock
                r.decode_output(self.tokenizer)
                if use_kv:
                    self._free_kv_slot(r.id)
                active.remove(r)
                done.append(r)

            # Admit late arrivals
            while pending and pending[0][1] <= clock:
                r2, arr2 = pending.popleft()
                r2.arrival_time = arr2
                if r2.first_token_time < 0:
                    r2.first_token_time = clock
                if use_kv:
                    self._alloc_kv_slot(r2.id)
                active.append(r2)

        meta = {
            "total_iters": total_iters,
            "padding_waste": total_padding_waste,
        }
        if self._arbiter is not None:
            meta["arbiter"] = self._arbiter.get_summary()
        return done, clock, meta

    # ─── Internal: async loop ─────────────────────────────────────

    async def _run_loop(self):
        """Main async serving loop."""
        use_kv = self.config.use_kv_cache
        if use_kv:
            self._init_kv_pool()

        while self._running:
            self._admit_pending()
            if not self._active:
                await asyncio.sleep(0.001)
                continue
            batch = pick_batch(
                self._active, self.config.max_batch_size, self.config.policy)
            if use_kv:
                self._run_kv_step(batch)
            else:
                self._run_batched_iter(batch)
            self._total_iters += 1
            self._process_completions()
            await asyncio.sleep(0)

    def _admit_pending(self):
        """Move requests from pending queue to active list."""
        use_kv = self.config.use_kv_cache
        while self._pending:
            req, future = self._pending.popleft()
            if req.first_token_time < 0:
                req.first_token_time = time.perf_counter() * 1000
            if use_kv:
                self._alloc_kv_slot(req.id)
            self._active.append(req)

    def _process_completions(self):
        """Check for and handle completed requests."""
        use_kv = self.config.use_kv_cache
        newly_done = [r for r in self._active if r.done]
        for r in newly_done:
            r.finish_time = time.perf_counter() * 1000
            r.decode_output(self.tokenizer)
            if use_kv:
                self._free_kv_slot(r.id)
            self._active.remove(r)

            latency = r.finish_time - r.arrival_time
            ttft = r.first_token_time - r.arrival_time
            result = RequestResult(
                request_id=r.id,
                output_text=r.output_text,
                prompt_tokens=r.prompt_len,
                completion_tokens=len(r.output_text.split()),
                total_fwds=r.total_fwds,
                latency_ms=latency,
                time_to_first_token_ms=ttft,
            )
            if r.id in self._futures:
                future = self._futures.pop(r.id)
                if not future.done():
                    future.set_result(result)
            self._total_requests += 1
            self._total_latency_ms += latency
            self._total_tokens += r.prompt_len + (r.total_len - r.prompt_len)

    # ═══════════════════════════════════════════════════════════════
    # KV Cache Path: prefill + decode-only forward
    # ═══════════════════════════════════════════════════════════════

    def _run_kv_step(self, batch: List[DiffuseRequest]) -> float:
        """Run one step with KV cache: prefill new requests, then decode block.

        1. For any request not yet prefilled → run prefill forward (prompt → KV)
        2. For all requests → run decode forward (32-token block + KV cache)
        3. Threshold unmask + update scheduling state
        """
        if not batch:
            return 0.0

        config = self.config
        device = self.device
        block_length = config.block_length

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # ─── Phase 1: Prefill any new requests ────────────────────
        need_prefill = [r for r in batch if not r.prefilled]
        if need_prefill:
            self._prefill_batch(need_prefill)

        # ─── Phase 2: Decode one iteration ────────────────────────
        bs = len(batch)

        # Build decode input: only the current 32-token block per request
        decode_input = torch.full(
            (bs, block_length), config.mask_id,
            dtype=torch.long, device=device)
        decode_pos = torch.zeros(
            (bs, block_length), dtype=torch.long, device=device)

        # Find the common cache_length for this batch (align to captured sizes)
        cache_lengths = []
        for r in batch:
            cache_lengths.append(r.block_start)

        # All requests must share the same cache_length for CUDA graph replay.
        # We use the max and pad shorter ones.
        max_cache_len = max(cache_lengths) if cache_lengths else 0
        # Align to captured cache sizes
        runner_cache_lengths = sorted(self.model_runner.graph_runner.cache_lengths)
        idx = bisect.bisect_left(runner_cache_lengths, max_cache_len)
        if idx < len(runner_cache_lengths):
            aligned_cache_len = runner_cache_lengths[idx]
        else:
            aligned_cache_len = runner_cache_lengths[-1]

        # Build batched KV from pool
        slots = [self._get_kv_slot(r.id) for r in batch]
        # Extract per-request KV and stack into batch dim
        kv_batch = torch.zeros(
            self._kv_pool.shape[0], 2, bs, self._kv_pool.shape[3],
            aligned_cache_len, self._kv_pool.shape[5],
            dtype=torch.bfloat16, device=device)

        for i, (r, slot) in enumerate(zip(batch, slots)):
            cl = min(r.block_start, aligned_cache_len)
            decode_input[i] = r.x[r.block_start:r.block_end]
            decode_pos[i] = torch.arange(
                r.block_start, r.block_start + block_length, device=device)
            if cl > 0:
                kv_batch[:, :, i, :, :cl] = self._kv_pool[:, :, slot, :, :cl]

        # Forward: decode block with KV cache
        with torch.inference_mode():
            output = self.model_runner(
                decode_input, position_ids=decode_pos,
                use_cache=True, past_key_values=kv_batch)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        # ─── Phase 3: Update per-request state ────────────────────
        # Extract output KV and write back to pool for completed-block requests
        out_kv = output.past_key_values
        if isinstance(out_kv, (list, tuple)):
            # Stack list format to tensor
            num_layers = len(out_kv) // 2
            inner = out_kv[0].shape
            out_kv = torch.stack(out_kv, dim=0).reshape(
                num_layers, 2, *inner)

        for i, r in enumerate(batch):
            # Threshold unmask (same as no-KV path)
            self._update_request_kv(r, output, i)

            # If block just completed, write decode KV back to pool
            slot = slots[i]
            bl_s = r.block_start  # NOTE: this is post-advance if block completed
            if isinstance(out_kv, torch.Tensor) and out_kv.dim() == 6:
                # out_kv shape: [layers, 2, bs, heads, cache_len+block_len, dim]
                # We need to write the block_length columns back
                src_start = aligned_cache_len - block_length
                src_end = aligned_cache_len
                # The new KV for this block is at the end of the output
                if out_kv.shape[4] >= aligned_cache_len:
                    self._kv_pool[:, :, slot, :, src_start:src_end] = \
                        out_kv[:, :, i, :, src_start:src_end]

        return elapsed

    def _prefill_batch(self, reqs: List[DiffuseRequest]):
        """Batch-prefill prompt tokens into KV cache for new requests.

        Groups requests by aligned prompt length and processes each group
        as a single batched forward pass — O(groups) forwards instead of
        O(n_reqs). For HumanEval (same prompt template), all requests land
        in the same group → single forward for all prefills.
        """
        if not reqs:
            return

        config = self.config
        device = self.device
        block_length = config.block_length

        # Group by aligned prompt length for batched prefill
        from collections import defaultdict
        groups = defaultdict(list)
        for r in reqs:
            prompt_aligned = (r.prompt_len // block_length) * block_length
            prompt_aligned = max(prompt_aligned, block_length)
            groups[prompt_aligned].append(r)

        for prompt_aligned, group in groups.items():
            # Process in chunks of max_batch_size to fit CUDA graph
            max_bs = config.max_batch_size
            for chunk_start in range(0, len(group), max_bs):
                chunk = group[chunk_start:chunk_start + max_bs]
                bs = len(chunk)
                n_blocks_pf = prompt_aligned // block_length

                # Build batched inputs
                prefill_x = torch.full(
                    (bs, prompt_aligned), config.mask_id,
                    dtype=torch.long, device=device)
                prefill_pos = torch.arange(
                    prompt_aligned, device=device).unsqueeze(0).expand(bs, -1)

                for i, r in enumerate(chunk):
                    fill_len = min(r.prompt_len, prompt_aligned)
                    prefill_x[i, :fill_len] = r.x[:fill_len]

                # Block-diagonal attention mask (shared for same prompt length)
                block_mask = torch.tril(torch.ones(
                    n_blocks_pf, n_blocks_pf, device=device, dtype=torch.bool))
                attn_mask = block_mask.repeat_interleave(
                    block_length, 0).repeat_interleave(
                    block_length, 1).unsqueeze(0).expand(bs, -1, -1)

                with torch.inference_mode():
                    output = self.model_runner(
                        prefill_x, position_ids=prefill_pos,
                        use_cache=True, attention_mask=attn_mask)

                # Extract KV and store into pool per-request
                out_kv = output.past_key_values
                if isinstance(out_kv, (list, tuple)):
                    num_layers = len(out_kv) // 2
                    inner = out_kv[0].shape
                    out_kv = torch.stack(out_kv, dim=0).reshape(
                        num_layers, 2, *inner)

                for i, r in enumerate(chunk):
                    slot = self._get_kv_slot(r.id)
                    if isinstance(out_kv, torch.Tensor):
                        kv_len = min(out_kv.shape[4], self._kv_pool.shape[4])
                        self._kv_pool[:, :, slot, :, :kv_len] = \
                            out_kv[:, :, i, :, :kv_len]

                    r.prefilled = True
                    r.kv_seq_len = prompt_aligned
                    r.total_fwds += 1

    # ═══════════════════════════════════════════════════════════════
    # No-KV Path: full-sequence forward (v3 style)
    # ═══════════════════════════════════════════════════════════════

    def _run_batched_iter(self, batch: List[DiffuseRequest]) -> float:
        """Run one batched denoising iteration WITHOUT KV cache.

        Full-sequence forward each time. Simple but more compute.
        """
        if not batch:
            return 0.0

        config = self.config
        device = self.device
        mask_id = config.mask_id

        max_len = max(r.block_end for r in batch)
        bs = len(batch)

        # Construct batched input
        x_batch = torch.full(
            (bs, max_len), mask_id, dtype=torch.long, device=device)
        pos_batch = torch.arange(
            max_len, device=device).unsqueeze(0).expand(bs, -1)

        for i, r in enumerate(batch):
            x_batch[i, :r.block_end] = r.x[:r.block_end]

        # Construct block-diagonal attention mask
        attn_mask = torch.zeros(
            bs, max_len, max_len, device=device, dtype=torch.bool)
        for i, r in enumerate(batch):
            n_blk = r.cur_block + 1
            for b1 in range(n_blk):
                s1 = (r.prompt_len + b1 * config.block_length
                       if b1 > 0 else 0)
                e1 = min(
                    r.prompt_len + (b1 + 1) * config.block_length
                    if b1 > 0 else r.prompt_len + config.block_length,
                    r.block_end)
                for b2 in range(b1 + 1):
                    s2 = (r.prompt_len + b2 * config.block_length
                           if b2 > 0 else 0)
                    e2 = min(
                        r.prompt_len + (b2 + 1) * config.block_length
                        if b2 > 0 else r.prompt_len + config.block_length,
                        r.block_end)
                    attn_mask[i, s1:e1, s2:e2] = True
            attn_mask[i, :r.prompt_len, :r.prompt_len] = True

        # Forward pass
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            output = self.model_runner(
                x_batch, position_ids=pos_batch,
                use_cache=False, attention_mask=attn_mask)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000

        # Post-process
        for i, r in enumerate(batch):
            self._update_request(r, output, i)

        return elapsed

    # ═══════════════════════════════════════════════════════════════
    # Shared: threshold unmask + confidence tracking
    # ═══════════════════════════════════════════════════════════════

    def _update_request(self, r: DiffuseRequest, output, batch_idx: int):
        """Update request after no-KV forward. Logits cover full sequence."""
        bl_s, bl_e = r.block_start, r.block_end
        logits = output.logits[batch_idx, bl_s:bl_e]
        self._apply_threshold_unmask(r, logits, bl_s, bl_e)

    def _update_request_kv(self, r: DiffuseRequest, output, batch_idx: int):
        """Update request after KV-cache forward. Logits cover only the block."""
        logits = output.logits[batch_idx]  # [block_length, vocab]
        bl_s, bl_e = r.block_start, r.block_end
        block_len = bl_e - bl_s
        logits = logits[:block_len]
        self._apply_threshold_unmask(r, logits, bl_s, bl_e)

    def _apply_threshold_unmask(
        self, r: DiffuseRequest, logits: torch.Tensor,
        bl_s: int, bl_e: int,
    ):
        """Core threshold unmasking + confidence tracking.

        v2 optimizations:
        1. Adaptive threshold: high mean_confidence -> lower threshold -> faster convergence
        2. Minimum unmask guarantee: at least 2 tokens per iter
        3. Convergence acceleration: stalled iterations -> threshold decay
        Target: reduce fwds/req from ~62 to ~25 -> cut TPOT P90 by >60%.
        """
        config = self.config
        mask_id = config.mask_id
        eos_id = config.eos_id

        block_tokens = r.x[bl_s:bl_e]
        mask_idx = (block_tokens == mask_id)
        n_masked_before = mask_idx.sum().item()

        if not mask_idx.any():
            if (block_tokens == eos_id).any():      r.mark_early_stop()
            else:
                r.advance_block()
            return

        # Argmax prediction
        x0 = logits.argmax(dim=-1)
        probs = F.softmax(logits.float(), dim=-1)
        x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        valid_mask = mask_idx & (x0 != mask_id)
        x0 = torch.where(mask_idx, x0, block_tokens)
        confidence = torch.where(
            valid_mask, x0_p,
            torch.tensor(-float('inf'), device=r.x.device))

        # Track confidence for CW-SRPT
        valid_conf = confidence[valid_mask]
        mean_conf = valid_conf.mean().item() if len(valid_conf) > 0 else 0.0
        r.last_confidence_sum = (
            valid_conf.sum().item() if len(valid_conf) > 0 else 0.0)

        # v2: Adaptive threshold
        base_th = r.threshold  # 0.9

        # (a) Confidence-adaptive: high mean_conf -> lower threshold
        conf_boost = 0.5 * max(0.0, mean_conf - 0.4)
        adaptive_th = base_th * (1.0 - conf_boost)

        # (b) Stall decay: stuck iterations -> more aggressive
        stall_decay = 0.02 * r.iters_since_progress
        adaptive_th = max(0.3, adaptive_th - stall_decay)

        # Apply threshold
        max_conf = confidence.max()
        actual_th = min(adaptive_th, max_conf.item() - 1e-5)
        transfer = confidence >= actual_th

        # (c) Minimum unmask: at least min_k tokens per iter
        min_k = min(2, n_masked_before)
        if transfer.sum().item() < min_k and n_masked_before >= min_k:
            _, topk_idx = torch.topk(confidence, k=min_k)
            transfer[topk_idx] = True

        r.x[bl_s:bl_e] = torch.where(transfer, x0, block_tokens)

        # Update convergence statistics
        n_masked_after = (r.x[bl_s:bl_e] == mask_id).sum().item()
        n_unmasked = n_masked_before - n_masked_after
        r.last_n_unmasked = n_unmasked

        # EMA convergence rate
        alpha = 0.3
        r.convergence_rate = (
            alpha * n_unmasked + (1 - alpha) * r.convergence_rate)

        # Starvation detection
        if n_unmasked > 0:
            r.iters_since_progress = 0
        else:
            r.iters_since_progress += 1

        # Check block completion
        if n_masked_after == 0:
            if (r.x[bl_s:bl_e] == eos_id).any():
                r.mark_early_stop()
            else:
                r.advance_block()

        r.total_fwds += 1

