"""
FoundryGraphPool: Foundry-backed CUDA graph pool for DiffServe.

Replaces dInfer's CudaGraphRunner with Foundry's template-based context
materialization. Key differences:

  CudaGraphRunner (old):
    - Captures [1,2,4,8] batch sizes → 4 graphs per cache_length
    - Uses torch.cuda.CUDAGraph → 60s cold start
    - 7% graph miss on non-power-of-2 batch sizes

  FoundryGraphPool (new):
    - Captures [1..max_batch] dense → every bs has a graph
    - Uses foundry.CUDAGraph → save to disk, load in ~1s
    - 0% graph miss, 0 padding waste from batch sizing

SAVE phase (offline, once):
    pool = FoundryGraphPool(model_runner, config)
    pool.capture_and_save("/path/to/archive")

SERVE phase (online, every startup):
    pool = FoundryGraphPool.load_from_archive(model_runner, config, "/path/to/archive")
    output = pool.replay(input_ids, position_ids, past_key_values, ...)
"""

import bisect
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class FoundryGraphPool:
    """Dense CUDA graph pool backed by Foundry for instant restore.

    Mirrors CudaGraphRunner's interface so it can be a drop-in replacement
    in ModelRunner.forward().
    """

    def __init__(self, model_runner, config=None):
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)

        # Graph index: (bs, is_decode, length, cache_length) → (graph, output_buf)
        self.graphs: Dict[tuple, Any] = {}
        self.output_buffers: Dict[tuple, Any] = {}

        # Dense batch sizes: [1, 2, 3, ..., max_batch]
        max_bs = max(model_runner.supported_batch_sizes)
        self.capture_bs = list(range(1, max_bs + 1))
        self.max_bs = max_bs

        # Inherit from model_runner
        self.block_length = model_runner.block_length
        self.prefill_lengths = model_runner.prefill_lengths
        self.cache_lengths = model_runner.cache_lengths
        self.decoding_lengths = model_runner.decoding_lengths
        self.max_num_token = self.max_bs * max(
            self.block_length * 2, max(self.prefill_lengths))
        self.disable_padding = False

        from sglang.srt.layers.dp_attention import get_attention_tp_size
        self.tp_size = get_attention_tp_size()
        self.enable_compile = model_runner.enable_compile

        # Static buffers (same layout as CudaGraphRunner)
        cfg = model_runner.model.config
        num_layers = cfg.num_hidden_layers
        num_kv_heads = cfg.num_key_value_heads
        num_heads = cfg.num_attention_heads
        head_dim = cfg.hidden_size // num_heads

        with torch.device(self.device):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.position_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.past_key_values = torch.zeros(
                (num_layers, 2, self.max_bs,
                 max(1, num_kv_heads // self.tp_size),
                 model_runner.max_length, head_dim),
                dtype=torch.bfloat16)
            self.attention_mask = torch.ones(
                (self.max_bs, model_runner.max_length, model_runner.max_length),
                dtype=torch.bool)
            self.attention_mask[0, 0, 0] = False

        self._archive_dir: Optional[str] = None

    # ─── SAVE Phase ───────────────────────────────────────────────

    def capture_and_save(self, archive_dir: str):
        """Capture all graphs and save to disk via Foundry.

        This is the SAVE phase — run once offline for each model.
        Creates a Foundry archive that can be loaded in ~1s.
        """
        try:
            import foundry
            from foundry.graph import CUDAGraph as FoundryCUDAGraph
        except ImportError:
            logger.warning(
                "Foundry not available — falling back to torch.cuda.CUDAGraph. "
                "Install foundry for instant graph restore.")
            self._capture_torch_fallback()
            return

        os.makedirs(archive_dir, exist_ok=True)
        self._archive_dir = archive_dir

        from sglang.srt.distributed.parallel_state import graph_capture
        from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode
        from sglang.srt.utils import get_available_gpu_memory

        logger.info(f"[FoundryGraphPool] SAVE phase: capturing dense graphs to {archive_dir}")
        t0 = time.perf_counter()
        graph_index = 0

        with model_capture_mode():
            with graph_capture() as ctx:
                self.stream = ctx.stream

                # Decode graphs: all bs × cache_lengths × lengths
                for bs in sorted(self.capture_bs, reverse=True):
                    for cache_length in self.cache_lengths:
                        for length in [self.block_length]:
                            graph, output = self._capture_one(
                                bs, True, length, cache_length,
                                use_mask=False, use_foundry=True)
                            key = (bs, True, length, cache_length)
                            self.graphs[key] = graph
                            self.output_buffers[key] = output

                            # Save via Foundry
                            save_path = os.path.join(
                                archive_dir,
                                f"graph_{graph_index}_bs{bs}_d_l{length}_c{cache_length}.json")
                            graph.save(save_path, output.logits if hasattr(output, 'logits') else output)
                            graph_index += 1

                        # Cross-block (2x block_length) if enabled
                        if self.model_runner.use_cross_block:
                            length = self.block_length * 2
                            graph, output = self._capture_one(
                                bs, True, length, cache_length,
                                use_mask=True, use_foundry=True)
                            key = (bs, True, length, cache_length)
                            self.graphs[key] = graph
                            self.output_buffers[key] = output

                            save_path = os.path.join(
                                archive_dir,
                                f"graph_{graph_index}_bs{bs}_d_l{length}_c{cache_length}.json")
                            graph.save(save_path, output.logits if hasattr(output, 'logits') else output)
                            graph_index += 1

                # Prefill graphs: max_bs × prefill_lengths
                for length in self.prefill_lengths:
                    bs = self.max_bs
                    graph, output = self._capture_one(
                        bs, False, length, 0,
                        use_mask=True, use_foundry=True)
                    key = (bs, False, length, 0)
                    self.graphs[key] = graph
                    self.output_buffers[key] = output

                    save_path = os.path.join(
                        archive_dir,
                        f"graph_{graph_index}_bs{bs}_p_l{length}.json")
                    graph.save(save_path, output.logits if hasattr(output, 'logits') else output)
                    graph_index += 1

        # Save manifest
        try:
            foundry.save_graph_manifest(archive_dir)
        except Exception as e:
            logger.warning(f"Could not save manifest: {e}")

        elapsed = time.perf_counter() - t0
        logger.info(
            f"[FoundryGraphPool] SAVE complete: {graph_index} graphs "
            f"in {elapsed:.1f}s → {archive_dir}")

    # ─── LOAD Phase ───────────────────────────────────────────────

    @classmethod
    def load_from_archive(cls, model_runner, config, archive_dir: str) -> "FoundryGraphPool":
        """Load graphs from Foundry archive (~1s cold start).

        This is the SERVE phase — run at every startup.
        """
        pool = cls(model_runner, config)
        pool._archive_dir = archive_dir

        try:
            from foundry.graph import CUDAGraph as FoundryCUDAGraph
        except ImportError:
            logger.warning("Foundry not available — falling back to capture")
            pool._capture_torch_fallback()
            return pool

        # Scan archive for available graphs
        pattern = re.compile(
            r"graph_(\d+)_bs(\d+)_(d|p)_l(\d+)(?:_c(\d+))?\.json")
        graph_files = []
        for f in sorted(os.listdir(archive_dir)):
            m = pattern.match(f)
            if m:
                idx, bs, phase, length = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
                cache_len = int(m.group(5)) if m.group(5) else 0
                is_decode = (phase == 'd')
                graph_files.append({
                    'path': os.path.join(archive_dir, f),
                    'key': (bs, is_decode, length, cache_len),
                    'index': idx,
                })

        if not graph_files:
            logger.warning(f"No graph files found in {archive_dir} — will capture from scratch")
            pool._capture_torch_fallback()
            return pool

        # Load via Foundry's batched template-sharing loader
        paths = [g['path'] for g in graph_files]
        logger.info(f"[FoundryGraphPool] Loading {len(paths)} graphs from {archive_dir}...")
        t0 = time.perf_counter()

        pending = FoundryCUDAGraph.start_graph_builds(paths, num_threads=8)
        results = FoundryCUDAGraph.finish_graph_loads(pending)

        for i, gf in enumerate(graph_files):
            graph, output_tensors = results[i]
            pool.graphs[gf['key']] = graph
            pool.output_buffers[gf['key']] = output_tensors

        elapsed = time.perf_counter() - t0
        logger.info(
            f"[FoundryGraphPool] Loaded {len(pool.graphs)} graphs "
            f"in {elapsed:.2f}s (template sharing)")

        return pool

    # ─── Fallback: torch.cuda.CUDAGraph ──────────────────────────

    def _capture_torch_fallback(self):
        """Fall back to standard CUDA graph capture (like CudaGraphRunner)."""
        from sglang.srt.distributed.parallel_state import graph_capture
        from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode

        logger.info("[FoundryGraphPool] Falling back to torch.cuda.CUDAGraph capture")
        with model_capture_mode():
            with graph_capture() as ctx:
                self.stream = ctx.stream
                for bs in sorted(self.capture_bs, reverse=True):
                    for cache_length in self.cache_lengths:
                        length = self.block_length
                        graph, output = self._capture_one(
                            bs, True, length, cache_length,
                            use_mask=False, use_foundry=False)
                        self.graphs[(bs, True, length, cache_length)] = graph
                        self.output_buffers[(bs, True, length, cache_length)] = output

                        if self.model_runner.use_cross_block:
                            length = self.block_length * 2
                            graph, output = self._capture_one(
                                bs, True, length, cache_length,
                                use_mask=True, use_foundry=False)
                            self.graphs[(bs, True, length, cache_length)] = graph
                            self.output_buffers[(bs, True, length, cache_length)] = output

                for length in self.prefill_lengths:
                    bs = self.max_bs
                    graph, output = self._capture_one(
                        bs, False, length, 0,
                        use_mask=True, use_foundry=False)
                    self.graphs[(bs, False, length, 0)] = graph
                    self.output_buffers[(bs, False, length, 0)] = output

        logger.info(f"[FoundryGraphPool] Captured {len(self.graphs)} graphs (torch fallback)")

    # ─── Graph Capture (shared by SAVE and fallback) ──────────────

    def _capture_one(
        self, bs: int, is_decode: bool, length: int,
        cache_length: int = 0, use_mask: bool = False,
        use_foundry: bool = False,
    ) -> Tuple[Any, Any]:
        """Capture a single CUDA graph for a specific shape.

        Mirrors CudaGraphRunner.capture_one_batch_size() but supports
        both Foundry and standard torch graphs.
        """
        from diffserve.model_loader import setup_vllm_mocks
        from sglang.srt.distributed import get_tp_group

        if use_foundry:
            try:
                from foundry.graph import CUDAGraph as FoundryCUDAGraph
                graph = FoundryCUDAGraph()
            except ImportError:
                graph = torch.cuda.CUDAGraph()
        else:
            graph = torch.cuda.CUDAGraph()

        num_tokens = bs * length
        input_ids = self.input_ids[:num_tokens].view(bs, length)
        position_ids = self.position_ids[:num_tokens].view(bs, length)

        past_kv = (self.past_key_values[:, :, :bs, :, :cache_length]
                    if is_decode else None)
        attn_mask = None
        if use_mask:
            if not is_decode:
                attn_mask = self.attention_mask[:bs, :length, :length]
            else:
                attn_mask = self.attention_mask[:bs, :length, :cache_length]

        tp_group = get_tp_group()

        # Get compiled or raw forward
        from diffserve.model_loader import setup_vllm_mocks
        forward = self.model_runner.model.forward

        def run_once():
            return forward(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=None,
                pp_proxy_tensors=None,
                past_key_values=past_kv,
                replace_position=(0, 0),
                use_cache=True,
                attention_mask=attn_mask,
            )

        # Warmup
        for _ in range(2):
            self.device_module.synchronize()
            tp_group.barrier()
            run_once()

        # Capture
        pool = self.device_module.graph_pool_handle()
        with self.device_module.graph(graph, pool=pool, stream=self.stream):
            output = run_once()

        return graph, output

    # ─── Runtime: can_run / replay ────────────────────────────────

    def can_run(self, input_ids, position_ids, past_key_values,
                is_decode_phase=True, length=0, cache_length=0):
        """Check if we have a graph for this shape."""
        bs = input_ids.shape[0]
        key = (bs, is_decode_phase, length, cache_length)
        if key in self.graphs:
            return True
        # Try max_bs for non-decode (prefill) or padding
        if not self.disable_padding:
            if (self.max_bs, is_decode_phase, length, cache_length) in self.graphs:
                return True
        return False

    def replay_prepare(self, input_ids, position_ids, past_key_values,
                       is_decode_phase, length, attention_mask, cache_length):
        """Stage inputs into static buffers before replay.

        Same logic as CudaGraphRunner.replay_prepare().
        """
        from dinfer.decoding.utils import KVCache

        raw_bs = input_ids.shape[0]
        raw_num_token = raw_bs * length

        # Find the nearest supported bs
        idx = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[min(idx, len(self.capture_bs) - 1)]

        self.input_ids[:raw_num_token].copy_(input_ids.flatten())
        self.position_ids[:raw_num_token].copy_(position_ids.flatten())

        if is_decode_phase and past_key_values is not None:
            pkv = past_key_values._data if isinstance(past_key_values, KVCache) else past_key_values
            if isinstance(pkv, torch.Tensor):
                min_len = min(self.past_key_values.shape[4], pkv.shape[4], cache_length)
                self.past_key_values[:, :, :, :, min_len:].fill_(0)
                self.past_key_values[:, :, :raw_bs, :, :min_len].copy_(
                    pkv[:, :, :, :, :min_len])

        if attention_mask is not None:
            min_len = min(self.attention_mask.shape[2], attention_mask.shape[2])
            self.attention_mask[:raw_bs, :length, :min_len].copy_(
                attention_mask[:, :, :min_len])

        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs
        self.is_decode_phase = is_decode_phase
        self.length = length

    def replay(self, input_ids, position_ids, past_key_values,
               is_decode_phase, length, attention_mask, cache_length):
        """Stage inputs and replay the graph."""
        self.replay_prepare(
            input_ids, position_ids, past_key_values,
            is_decode_phase, length, attention_mask, cache_length)

        key = (self.bs, is_decode_phase, length, cache_length)
        self.graphs[key].replay()
        return self.output_buffers[key]

    # ─── Stats ────────────────────────────────────────────────────

    @property
    def coverage(self) -> dict:
        """Return graph coverage statistics."""
        decode_graphs = {k for k in self.graphs if k[1] is True}
        prefill_graphs = {k for k in self.graphs if k[1] is False}
        bs_covered = sorted(set(k[0] for k in decode_graphs))
        return {
            "total_graphs": len(self.graphs),
            "decode_graphs": len(decode_graphs),
            "prefill_graphs": len(prefill_graphs),
            "bs_covered": bs_covered,
            "archive_dir": self._archive_dir,
        }
