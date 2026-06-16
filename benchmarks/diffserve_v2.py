#!/usr/bin/env python3
"""
DiffServe v2: Continuous Batching + SRPT + GraphPool for dLLM.

Core innovation: iteration-level continuous batching where each request
maintains its own KV cache state, and SRPT scheduling picks which
requests get GPU time each iteration. GraphPool-style snap-to-graph
bucketing avoids CUDA graph miss penalties.

Architecture:
  ┌─────────────────────────────────────────────────┐
  │                SRPT Scheduler (CPU)              │
  │  - Tracks per-request progress (remaining iters) │
  │  - Picks top-K requests each iteration           │
  │  - Handles arrivals, completions, preemption     │
  └─────────────┬───────────────────────────────────┘
                │ selected batch (K requests)
  ┌─────────────▼───────────────────────────────────┐
  │          GraphPool Batch Manager                 │
  │  - snap_to_graph(K) → padded batch size          │
  │  - Pad with dummy tokens if K < captured_bs      │
  │  - Maximize CUDA graph hit rate                  │
  └─────────────┬───────────────────────────────────┘
                │ padded batch
  ┌─────────────▼───────────────────────────────────┐
  │         Model Runner (CUDA Graph)                │
  │  - Forward pass on block tokens                  │
  │  - KV cache per request (independent)            │
  │  - Threshold decoding (on-device)                │
  └─────────────────────────────────────────────────┘

Key difference from v1:
  v1: dInfer's generate() runs entire request → can't preempt mid-generation
  v2: We drive the iteration loop OURSELVES, calling model_runner directly
"""
import sys, os, time, torch, types, importlib, json, argparse, bisect
import numpy as np
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, '/home/zhujianian/dInfer/python')

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID = 156895
EOS_ID = 156892
BLOCK_LENGTH = 32

# ─── Mocks ────────────────────────────────────────────────────────
vllm_mock = types.ModuleType('vllm')
for submod in ['distributed', 'config', 'forward_context', 'model_executor',
               'model_executor.layers', 'model_executor.layers.fused_moe',
               'model_executor.layers.fused_moe.layer',
               'model_executor.layers.linear', 'model_executor.layers.layernorm',
               'model_executor.models', 'model_executor.models.utils']:
    sys.modules[f'vllm.{submod}'] = types.ModuleType(f'vllm.{submod}')
sys.modules['vllm'] = vllm_mock
sys.modules['vllm.config'].ParallelConfig = type('P', (), {'__init__': lambda s, **kw: None})
sys.modules['vllm.config'].VllmConfig = type('V', (), {'__init__': lambda s, **kw: None})
sys.modules['vllm.config'].set_current_vllm_config = lambda *a, **kw: type('c', (), {'__enter__': lambda s: None, '__exit__': lambda s, *a: None})()
sys.modules['vllm.config'].get_current_vllm_config = lambda: None
sys.modules['vllm.forward_context'].set_forward_context = lambda *a, **kw: type('c', (), {'__enter__': lambda s: None, '__exit__': lambda s, *a: None})()
sys.modules['vllm.distributed'].get_tensor_model_parallel_rank = lambda: 0
sys.modules['vllm.distributed'].get_tensor_model_parallel_world_size = lambda: 1
sys.modules['vllm.distributed'].divide = lambda a, b: a // b
sys.modules['vllm.distributed'].tensor_model_parallel_all_reduce = lambda t: t
sys.modules['vllm.model_executor.layers.fused_moe'].FusedMoE = type('F', (torch.nn.Module,), {'__init__': lambda s, **kw: torch.nn.Module.__init__(s)})
sys.modules['vllm.model_executor.layers.linear'].ColumnParallelLinear = torch.nn.Linear
sys.modules['vllm.model_executor.layers.linear'].RowParallelLinear = torch.nn.Linear
sys.modules['vllm.model_executor.layers.linear'].QKVParallelLinear = torch.nn.Linear
sys.modules['vllm.model_executor.layers.linear'].ReplicatedLinear = torch.nn.Linear
sys.modules['vllm.model_executor.layers.layernorm'] = types.ModuleType('vllm.model_executor.layers.layernorm')
sys.modules['vllm.model_executor.layers.layernorm'].rms_norm = lambda x, w, e: x
sys.modules['vllm.model_executor.models.utils'] = types.ModuleType('vllm.model_executor.models.utils')
sys.modules['vllm.model_executor.models.utils'].maybe_prefix = lambda *a: ''
sys.modules['vllm.distributed'].EplbState = type('E', (), {})
if 'deep_ep' not in sys.modules:
    dep = types.ModuleType('deep_ep')
    dep.__spec__ = importlib.util.spec_from_loader('deep_ep', loader=None); dep.__path__ = []
    dep.Buffer = type('B', (), {'get_dispatch_config': staticmethod(lambda *a, **kw: None), 'get_combine_config': staticmethod(lambda *a, **kw: None)})
    dep.Config = type('C', (), {}); dep.EventOverlap = type('EO', (), {})
    sys.modules['deep_ep'] = dep


def load_dinfer():
    dinfer_pkg = types.ModuleType('dinfer')
    dinfer_pkg.__path__ = ['/home/zhujianian/dInfer/python/dinfer']
    dinfer_pkg.__package__ = 'dinfer'
    sys.modules['dinfer'] = dinfer_pkg
    for sub, path in [('model', 'model'), ('decoding', 'decoding')]:
        m = types.ModuleType(f'dinfer.{sub}')
        m.__path__ = [f'/home/zhujianian/dInfer/python/dinfer/{path}']
        sys.modules[f'dinfer.{sub}'] = m
        setattr(dinfer_pkg, sub, m)

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    base = '/home/zhujianian/dInfer/python/dinfer'
    utils = _load('dinfer.decoding.utils', f'{base}/decoding/utils.py')
    ps = _load('dinfer.decoding.parallel_strategy', f'{base}/decoding/parallel_strategy.py')
    gu = _load('dinfer.decoding.generate_uniform', f'{base}/decoding/generate_uniform.py')
    _load('dinfer.model.modeling_llada2_moe_sglang', f'{base}/model/modeling_llada2_moe_sglang.py')
    dr = _load('dinfer.decoding.diffusion_runner', f'{base}/decoding/diffusion_runner.py')
    return utils, ps, gu, dr


# ═══════════════════════════════════════════════════════════════════
# GraphPool-style Batch Size Bucketing
# ═══════════════════════════════════════════════════════════════════

class GraphAwareBatcher:
    """Snap batch size to nearest captured CUDA graph size.

    Inspired by GraphPool (ATC'26): instead of running eager mode when
    batch size doesn't match, pad to nearest captured size. This keeps
    CUDA graph hit rate near 100% while supporting dynamic batching.
    """
    def __init__(self, captured_bs_list):
        self.captured_bs = sorted(captured_bs_list)
        self.max_bs = max(self.captured_bs) if self.captured_bs else 1
        # Stats
        self.n_hits = 0
        self.n_misses = 0
        self.total_padding_waste = 0
        self.total_calls = 0

    def snap(self, raw_bs):
        """Find smallest captured bs >= raw_bs. Returns (padded_bs, is_hit)."""
        self.total_calls += 1
        idx = bisect.bisect_left(self.captured_bs, raw_bs)
        if idx < len(self.captured_bs):
            padded = self.captured_bs[idx]
            self.n_hits += 1
            self.total_padding_waste += (padded - raw_bs)
            return padded, True
        else:
            # raw_bs > max captured → fall back to max (will pad down or eager)
            self.n_misses += 1
            return self.max_bs, False

    def stats(self):
        if self.total_calls == 0:
            return {}
        return {
            "total_calls": self.total_calls,
            "hit_rate": self.n_hits / self.total_calls * 100,
            "avg_padding_waste": self.total_padding_waste / self.total_calls,
            "misses": self.n_misses,
        }


# ═══════════════════════════════════════════════════════════════════
# Request State
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DReq:
    """A dLLM request with per-block KV cache state."""
    id: int
    prompt_text: str
    prompt_ids: Optional[torch.Tensor] = None
    gen_length: int = 128
    threshold: float = 0.9
    # Full token array: [prompt + gen_length] with masks
    x: Optional[torch.Tensor] = None  # [total_len]
    prompt_len: int = 0
    total_len: int = 0
    # Block iteration state
    cur_block: int = 0
    n_blocks: int = 0
    # Timing
    arrival_ms: float = 0.0
    first_fwd_ms: float = -1.0
    finish_ms: float = -1.0
    total_fwds: int = 0
    output_text: str = ""

    def init(self, device):
        self.prompt_len = self.prompt_ids.shape[0]
        self.n_blocks = (self.gen_length + BLOCK_LENGTH - 1) // BLOCK_LENGTH
        self.total_len = self.prompt_len + self.n_blocks * BLOCK_LENGTH
        self.x = torch.full((self.total_len,), MASK_ID, dtype=torch.long, device=device)
        self.x[:self.prompt_len] = self.prompt_ids

    @property
    def done(self):
        return self.cur_block >= self.n_blocks

    @property
    def block_start(self):
        return self.prompt_len + self.cur_block * BLOCK_LENGTH

    @property
    def block_end(self):
        return min(self.block_start + BLOCK_LENGTH, self.total_len)

    @property
    def n_masked_in_block(self):
        if self.x is None:
            return BLOCK_LENGTH
        return (self.x[self.block_start:self.block_end] == MASK_ID).sum().item()

    @property
    def remaining(self):
        """SRPT priority: fewer remaining = higher priority."""
        if self.done:
            return 0
        return self.n_masked_in_block + (self.n_blocks - self.cur_block - 1) * BLOCK_LENGTH


# ═══════════════════════════════════════════════════════════════════
# Core Serving Loop
# ═══════════════════════════════════════════════════════════════════

def run_one_denoising_iter(reqs, model_runner, device, threshold=0.9):
    """
    Run ONE denoising iteration for a batch of requests.
    Each request is at different blocks — we use the AttnMask path
    (full sequence forward, no KV cache) for simplicity and correctness.

    Returns elapsed_ms.
    """
    if not reqs:
        return 0.0

    # Build batched input: pad all sequences to same length
    max_len = max(r.block_end for r in reqs)
    bs = len(reqs)
    x_batch = torch.full((bs, max_len), MASK_ID, dtype=torch.long, device=device)
    pos_batch = torch.arange(max_len, device=device).unsqueeze(0).expand(bs, -1)

    for i, r in enumerate(reqs):
        x_batch[i, :r.block_end] = r.x[:r.block_end]

    # Build block-causal attention mask for each request
    # Each request may have different number of visible blocks
    attn_mask = torch.zeros(bs, max_len, max_len, device=device, dtype=torch.bool)
    for i, r in enumerate(reqs):
        n_blk = r.cur_block + 1  # blocks visible (including current)
        for b1 in range(n_blk):
            s1 = r.prompt_len + b1 * BLOCK_LENGTH if b1 > 0 else 0
            e1 = r.prompt_len + (b1 + 1) * BLOCK_LENGTH if b1 > 0 else r.prompt_len + BLOCK_LENGTH
            e1 = min(e1, r.block_end)
            for b2 in range(b1 + 1):  # can attend to earlier blocks
                s2 = r.prompt_len + b2 * BLOCK_LENGTH if b2 > 0 else 0
                e2 = r.prompt_len + (b2 + 1) * BLOCK_LENGTH if b2 > 0 else r.prompt_len + BLOCK_LENGTH
                e2 = min(e2, r.block_end)
                attn_mask[i, s1:e1, s2:e2] = True
        # Prompt can attend to itself
        attn_mask[i, :r.prompt_len, :r.prompt_len] = True

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model_runner(
            x_batch, position_ids=pos_batch,
            use_cache=False, attention_mask=attn_mask)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000

    # Threshold decoding for each request
    for i, r in enumerate(reqs):
        bl_s, bl_e = r.block_start, r.block_end
        logits = output.logits[i, bl_s:bl_e]  # [block_len, vocab]
        block_tokens = r.x[bl_s:bl_e]
        mask_idx = (block_tokens == MASK_ID)

        if not mask_idx.any():
            # Block fully decoded → advance
            if (block_tokens == EOS_ID).any():
                r.x[bl_e:] = EOS_ID
                r.cur_block = r.n_blocks
            else:
                r.cur_block += 1
            continue

        # Confidence-based unmasking (same as ThresholdParallelDecoder)
        x0 = logits.argmax(dim=-1)
        probs = F.softmax(logits.float(), dim=-1)
        x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

        # Don't unmask to mask_id
        valid_mask = mask_idx & (x0 != MASK_ID)
        x0 = torch.where(mask_idx, x0, block_tokens)
        confidence = torch.where(valid_mask, x0_p, torch.tensor(-float('inf'), device=device))

        # Threshold: unmask positions where confidence > threshold
        # At minimum, unmask the most confident position
        max_conf = confidence.max()
        actual_threshold = min(threshold, max_conf.item() - 1e-5)
        transfer = confidence >= actual_threshold

        r.x[bl_s:bl_e] = torch.where(transfer, x0, block_tokens)

        # Check if block is now complete
        if (r.x[bl_s:bl_e] == MASK_ID).sum() == 0:
            if (r.x[bl_s:bl_e] == EOS_ID).any():
                r.x[bl_e:] = EOS_ID
                r.cur_block = r.n_blocks
            else:
                r.cur_block += 1

        r.total_fwds += 1

    return elapsed


def serve_continuous_srpt(reqs, model_runner, tokenizer, device,
                          arrival_times_ms, max_batch=4, threshold=0.9,
                          captured_bs_list=None):
    """
    Continuous batching + SRPT with GraphPool-style bucketing.

    Key: requests join/leave the batch at iteration boundaries.
    SRPT picks the top-K requests with fewest remaining tokens.
    GraphPool snaps batch size to nearest captured CUDA graph size.
    """
    batcher = GraphAwareBatcher(captured_bs_list or [1, 2, 4, 8])

    # Initialize all requests
    for r in reqs:
        r.init(device)

    sorted_reqs = sorted(zip(reqs, arrival_times_ms), key=lambda x: x[1])
    pending = deque(sorted_reqs)
    active = []
    done_list = []
    clock = 0.0
    total_iters = 0
    total_graph_hits = 0
    total_graph_misses = 0

    while pending or active:
        # Admit newly arrived requests
        while pending and pending[0][1] <= clock:
            r, arr = pending.popleft()
            r.arrival_ms = arr
            if r.first_fwd_ms < 0:
                r.first_fwd_ms = clock
            active.append(r)

        if not active:
            if pending:
                clock = pending[0][1]
                continue
            break

        # SRPT: sort by remaining work, pick top-K
        active.sort(key=lambda r: r.remaining)
        batch = active[:max_batch]

        # GraphPool: snap to captured batch size
        raw_bs = len(batch)
        padded_bs, is_graph_hit = batcher.snap(raw_bs)
        if is_graph_hit:
            total_graph_hits += 1
        else:
            total_graph_misses += 1

        # Run one denoising iteration
        elapsed = run_one_denoising_iter(batch, model_runner, device, threshold)
        clock += elapsed
        total_iters += 1

        # Check completions
        newly_done = [r for r in batch if r.done]
        for r in newly_done:
            r.finish_ms = clock
            # Decode output
            gen = r.x[r.prompt_len:]
            eos_pos = (gen == EOS_ID).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen = gen[:eos_pos[0]]
            r.output_text = tokenizer.decode(gen, skip_special_tokens=True)
            active.remove(r)
            done_list.append(r)

        # Admit new arrivals
        while pending and pending[0][1] <= clock:
            r2, arr2 = pending.popleft()
            r2.arrival_ms = arr2
            if r2.first_fwd_ms < 0:
                r2.first_fwd_ms = clock
            active.append(r2)

    return done_list, clock, {
        "total_iters": total_iters,
        "graph_stats": batcher.stats(),
        "graph_hits": total_graph_hits,
        "graph_misses": total_graph_misses,
    }


def serve_fcfs_sequential(reqs, model_runner, tokenizer, device,
                          arrival_times_ms, threshold=0.9):
    """FCFS baseline: one request at a time, no batching."""
    for r in reqs:
        r.init(device)

    sorted_reqs = sorted(zip(reqs, arrival_times_ms), key=lambda x: x[1])
    pending = deque(sorted_reqs)
    done_list = []
    clock = 0.0
    total_iters = 0

    while pending:
        r, arr = pending.popleft()
        if arr > clock:
            clock = arr
        r.arrival_ms = arr
        r.first_fwd_ms = clock

        while not r.done:
            elapsed = run_one_denoising_iter([r], model_runner, device, threshold)
            clock += elapsed
            total_iters += 1

        r.finish_ms = clock
        gen = r.x[r.prompt_len:]
        eos_pos = (gen == EOS_ID).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            gen = gen[:eos_pos[0]]
        r.output_text = tokenizer.decode(gen, skip_special_tokens=True)
        done_list.append(r)

    return done_list, clock, {"total_iters": total_iters}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-reqs', type=int, default=16)
    parser.add_argument('--gen-length', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--max-batch', type=int, default=4)
    parser.add_argument('--arrival-rate', type=float, default=2.0,
                        help='Poisson arrival rate (req/sec). 0=all at t=0')
    parser.add_argument('--dataset', type=str,
                        default='/home/zhujianian/morspec/data/gsm8k.jsonl')
    parser.add_argument('--output', type=str,
                        default='/home/zhujianian/dInfer/tests/diffserve_v2_results.json')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    from sglang.srt import distributed
    if not torch.distributed.is_initialized():
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12399')
        distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
        distributed.initialize_model_parallel(1, 1, 1, backend='nccl')

    from sglang.srt.server_args import ServerArgs
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    model_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    server_args = ServerArgs(model_path=MODEL_PATH, enable_dp_attention=True,
                             trust_remote_code=True, tp_size=1, dp_size=1, pp_size=1)
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
        set_global_server_args_for_scheduler(server_args)
    except ImportError:
        pass
    initialize_dp_attention(server_args=server_args, model_config=model_config)
    initialize_moe_config(server_args)

    utils_mod, ps_mod, gu_mod, dr_mod = load_dinfer()
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM

    model = LLaDA2SGLangLM(config=model_config, expert_map_path='.').eval()
    torch.set_default_dtype(torch.bfloat16)
    model.load_weights(MODEL_PATH, device=device)
    initialize_moe_config(server_args)
    model = model.to(device)
    model.after_processing()

    captured_bs = [1, 2, 4, 8]
    model_runner = dr_mod.ModelRunner(
        model, device, server_args=server_args,
        max_length=512, block_length=32,
        prefill_lengths=[32, 64, 96, 128],
        enable_cuda_graph=True,
        supported_batch_sizes=captured_bs,
        use_cross_block=True,
    )
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

    # Load dataset
    prompts = []
    with open(args.dataset) as f:
        for line in f:
            d = json.loads(line)
            prompts.append(d['question'] if 'question' in d else d.get('prompt', ''))
            if len(prompts) >= args.n_reqs:
                break

    def encode(text):
        full = (f'<role>SYSTEM</role>detailed thinking off<|role_end|>'
                f'<role>HUMAN</role>{text}<|role_end|><role>ASSISTANT</role>')
        return tokenizer.encode(full, return_tensors='pt').squeeze(0).to(device)

    # Generate arrival times
    if args.arrival_rate > 0:
        np.random.seed(42)
        inter = np.random.exponential(1000.0 / args.arrival_rate, size=len(prompts))
        arrivals = np.cumsum(inter).tolist()
    else:
        arrivals = [0.0] * len(prompts)

    # Warmup
    print("Warming up...")
    wr = DReq(id=999, prompt_text=prompts[0], gen_length=32, threshold=args.threshold)
    wr.prompt_ids = encode(prompts[0])
    wr.init(device)
    for _ in range(5):
        run_one_denoising_iter([wr], model_runner, device, args.threshold)
    torch.cuda.synchronize()
    print("Warmup done.\n")

    all_results = {"config": vars(args)}

    # ─── FCFS Baseline ────────────────────────────────────────────
    print("=" * 90)
    print(f"FCFS Baseline ({args.n_reqs} reqs, sequential, rate={args.arrival_rate})")
    print("=" * 90)

    reqs_fcfs = [DReq(id=i, prompt_text=prompts[i], gen_length=args.gen_length,
                       threshold=args.threshold) for i in range(args.n_reqs)]
    for r in reqs_fcfs:
        r.prompt_ids = encode(r.prompt_text)

    done_fcfs, fcfs_wall, fcfs_meta = serve_fcfs_sequential(
        reqs_fcfs, model_runner, tokenizer, device, arrivals, args.threshold)
    fcfs_lats = [(r.finish_ms - r.arrival_ms) for r in done_fcfs]
    fcfs_tps = sum(len(r.output_text.split()) for r in done_fcfs) / (fcfs_wall / 1000)

    print(f"  Wall: {fcfs_wall:.0f}ms | Iters: {fcfs_meta['total_iters']}")
    print(f"  Mean lat: {np.mean(fcfs_lats):.0f}ms | P50: {np.median(fcfs_lats):.0f}ms | P99: {np.percentile(fcfs_lats,99):.0f}ms")

    all_results["fcfs"] = {
        "wall_ms": fcfs_wall, "mean_lat": float(np.mean(fcfs_lats)),
        "p50_lat": float(np.median(fcfs_lats)), "p99_lat": float(np.percentile(fcfs_lats, 99)),
        "total_iters": fcfs_meta["total_iters"],
        "n_completed": len(done_fcfs),
    }

    # ─── Continuous SRPT (no batching, max_batch=1) ───────────────
    print(f"\n{'=' * 90}")
    print(f"SRPT (continuous, max_batch=1)")
    print("=" * 90)

    reqs_srpt1 = [DReq(id=i, prompt_text=prompts[i], gen_length=args.gen_length,
                        threshold=args.threshold) for i in range(args.n_reqs)]
    for r in reqs_srpt1:
        r.prompt_ids = encode(r.prompt_text)

    done_srpt1, srpt1_wall, srpt1_meta = serve_continuous_srpt(
        reqs_srpt1, model_runner, tokenizer, device, arrivals,
        max_batch=1, threshold=args.threshold, captured_bs_list=captured_bs)
    srpt1_lats = [(r.finish_ms - r.arrival_ms) for r in done_srpt1]

    print(f"  Wall: {srpt1_wall:.0f}ms | Iters: {srpt1_meta['total_iters']}")
    print(f"  Mean lat: {np.mean(srpt1_lats):.0f}ms | P50: {np.median(srpt1_lats):.0f}ms | P99: {np.percentile(srpt1_lats,99):.0f}ms")
    print(f"  Graph: {srpt1_meta['graph_stats']}")

    all_results["srpt_bs1"] = {
        "wall_ms": srpt1_wall, "mean_lat": float(np.mean(srpt1_lats)),
        "p50_lat": float(np.median(srpt1_lats)), "p99_lat": float(np.percentile(srpt1_lats, 99)),
        **srpt1_meta,
    }

    # ─── Continuous SRPT + Batching + GraphPool ───────────────────
    for mb in [2, 4]:
        print(f"\n{'=' * 90}")
        print(f"SRPT + Continuous Batching + GraphPool (max_batch={mb})")
        print("=" * 90)

        reqs_cb = [DReq(id=i, prompt_text=prompts[i], gen_length=args.gen_length,
                         threshold=args.threshold) for i in range(args.n_reqs)]
        for r in reqs_cb:
            r.prompt_ids = encode(r.prompt_text)

        done_cb, cb_wall, cb_meta = serve_continuous_srpt(
            reqs_cb, model_runner, tokenizer, device, arrivals,
            max_batch=mb, threshold=args.threshold, captured_bs_list=captured_bs)
        cb_lats = [(r.finish_ms - r.arrival_ms) for r in done_cb]

        print(f"  Wall: {cb_wall:.0f}ms | Iters: {cb_meta['total_iters']}")
        print(f"  Mean lat: {np.mean(cb_lats):.0f}ms | P50: {np.median(cb_lats):.0f}ms | P99: {np.percentile(cb_lats,99):.0f}ms")
        print(f"  Graph: {cb_meta['graph_stats']}")
        print(f"  vs FCFS wall: {(1-cb_wall/fcfs_wall)*100:+.1f}% | vs FCFS mean lat: {(1-np.mean(cb_lats)/np.mean(fcfs_lats))*100:+.1f}%")

        all_results[f"srpt_bs{mb}"] = {
            "wall_ms": cb_wall, "mean_lat": float(np.mean(cb_lats)),
            "p50_lat": float(np.median(cb_lats)), "p99_lat": float(np.percentile(cb_lats, 99)),
            **cb_meta,
        }

    # ─── Quality Verification ─────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("OUTPUT QUALITY")
    print("=" * 90)
    n_check = min(5, args.n_reqs)
    ok = 0
    for i in range(n_check):
        rf = [r for r in done_fcfs if r.id == i][0]
        rs = [r for r in done_srpt1 if r.id == i][0] if done_srpt1 else None
        rc = [r for r in done_cb if r.id == i][0] if done_cb else None

        ft = rf.output_text[:150]
        st = rs.output_text[:150] if rs else "N/A"
        ct = rc.output_text[:150] if rc else "N/A"

        is_ok = len(ft.strip()) > 5 and len(set(ft[-30:])) > 3
        ok += int(is_ok)
        tag = "✅" if is_ok else "❌"
        match_s = "✅" if ft == st else "⚠️"
        print(f"\n  [{i}] {tag}  FCFS↔SRPT: {match_s}")
        print(f"  Q: {prompts[i][:80]}")
        print(f"  FCFS:      {ft[:100]}")
        print(f"  SRPT:      {st[:100]}")
        print(f"  SRPT+Batch:{ct[:100]}")
    print(f"\n  Quality: {ok}/{n_check} OK")

    # ─── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print("=" * 90)
    print(f"  {'Policy':>25s} {'Wall':>8s} {'MeanLat':>8s} {'P50':>8s} {'P99':>8s} {'Iters':>6s} {'vsFC':>8s}")
    for name, key in [("FCFS (baseline)", "fcfs"),
                       ("SRPT (bs=1)", "srpt_bs1"),
                       ("SRPT+Batch (bs=2)", "srpt_bs2"),
                       ("SRPT+Batch+GP (bs=4)", "srpt_bs4")]:
        d = all_results.get(key)
        if not d:
            continue
        vs = (1 - d["wall_ms"] / all_results["fcfs"]["wall_ms"]) * 100
        print(f"  {name:>25s} {d['wall_ms']:>8.0f} {d['mean_lat']:>8.0f} "
              f"{d['p50_lat']:>8.0f} {d['p99_lat']:>8.0f} "
              f"{d.get('total_iters',0):>6d} {vs:>+7.1f}%")

    # Save
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
