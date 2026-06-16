#!/usr/bin/env python3
"""
DiffServe v3: dLLM-Native Scheduling Beyond SRPT.

Tests scheduling policies that exploit dLLM's unique properties:

1. FCFS: baseline, sequential
2. SRPT: shortest remaining first (generic)
3. CW-SRPT: confidence-weighted remaining work prediction (dLLM-specific)
4. BAB-SRPT: block-aligned batching + SRPT (dLLM-specific)
   Groups requests by block index → zero padding waste
5. APPC: aggressive preemption with progress credit (dLLM-specific)
   Exploits free preemption to do fine-grained round-robin weighted by convergence rate

Uses Azure trace for realistic arrival patterns.
"""
import sys, os, time, torch, types, importlib, json, argparse, bisect, csv
import numpy as np
import torch.nn.functional as F
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
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
    for sub in ['model', 'decoding']:
        m = types.ModuleType(f'dinfer.{sub}')
        m.__path__ = [f'/home/zhujianian/dInfer/python/dinfer/{sub}']
        sys.modules[f'dinfer.{sub}'] = m
        setattr(dinfer_pkg, sub, m)
    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    base = '/home/zhujianian/dInfer/python/dinfer'
    _load('dinfer.decoding.utils', f'{base}/decoding/utils.py')
    _load('dinfer.decoding.parallel_strategy', f'{base}/decoding/parallel_strategy.py')
    _load('dinfer.decoding.generate_uniform', f'{base}/decoding/generate_uniform.py')
    _load('dinfer.model.modeling_llada2_moe_sglang', f'{base}/model/modeling_llada2_moe_sglang.py')
    dr = _load('dinfer.decoding.diffusion_runner', f'{base}/decoding/diffusion_runner.py')
    return dr


# ═══════════════════════════════════════════════════════════════════
# Request
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DReq:
    id: int
    prompt_ids: torch.Tensor = None
    gen_length: int = 128
    threshold: float = 0.9
    x: Optional[torch.Tensor] = None
    prompt_len: int = 0
    total_len: int = 0
    cur_block: int = 0
    n_blocks: int = 0
    arrival_ms: float = 0.0
    first_fwd_ms: float = -1.0
    finish_ms: float = -1.0
    total_fwds: int = 0
    output_text: str = ""
    # dLLM-specific scheduling state
    last_confidence_sum: float = 0.0  # sum of confidence at masked positions
    last_n_unmasked: int = 0          # tokens unmasked in last iteration
    convergence_rate: float = 0.0     # EMA of tokens_unmasked / iteration
    iters_since_progress: int = 0     # starvation detector

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
    def n_masked(self):
        if self.x is None or self.done:
            return 0
        return (self.x[self.block_start:self.block_end] == MASK_ID).sum().item()

    @property
    def remaining_naive(self):
        """SRPT: simple mask count."""
        return self.n_masked + (self.n_blocks - self.cur_block - 1) * BLOCK_LENGTH

    @property
    def remaining_confidence(self):
        """CW-SRPT: confidence-weighted remaining work.
        Low confidence positions need more iterations than high confidence ones."""
        base = (self.n_blocks - self.cur_block - 1) * BLOCK_LENGTH
        if self.last_confidence_sum > 0:
            # Weight current block by inverse confidence
            return (self.n_masked * (1.0 - self.last_confidence_sum / max(self.n_masked, 1))) + base
        return self.n_masked + base

    @property
    def progress_priority(self):
        """APPC: progress-credit priority. Fast convergers get priority."""
        if self.convergence_rate > 0:
            # Higher convergence rate → lower priority number → runs first
            return self.n_masked / max(self.convergence_rate, 0.1)
        return self.remaining_naive


# ═══════════════════════════════════════════════════════════════════
# Core Iteration (with confidence tracking)
# ═══════════════════════════════════════════════════════════════════

def run_iter(reqs, model_runner, device, threshold=0.9):
    """One denoising iteration with confidence tracking."""
    if not reqs:
        return 0.0
    max_len = max(r.block_end for r in reqs)
    bs = len(reqs)
    x_batch = torch.full((bs, max_len), MASK_ID, dtype=torch.long, device=device)
    pos_batch = torch.arange(max_len, device=device).unsqueeze(0).expand(bs, -1)
    for i, r in enumerate(reqs):
        x_batch[i, :r.block_end] = r.x[:r.block_end]

    attn_mask = torch.zeros(bs, max_len, max_len, device=device, dtype=torch.bool)
    for i, r in enumerate(reqs):
        n_blk = r.cur_block + 1
        for b1 in range(n_blk):
            s1 = r.prompt_len + b1 * BLOCK_LENGTH if b1 > 0 else 0
            e1 = min(r.prompt_len + (b1+1)*BLOCK_LENGTH if b1 > 0 else r.prompt_len+BLOCK_LENGTH, r.block_end)
            for b2 in range(b1+1):
                s2 = r.prompt_len + b2*BLOCK_LENGTH if b2 > 0 else 0
                e2 = min(r.prompt_len + (b2+1)*BLOCK_LENGTH if b2 > 0 else r.prompt_len+BLOCK_LENGTH, r.block_end)
                attn_mask[i, s1:e1, s2:e2] = True
        attn_mask[i, :r.prompt_len, :r.prompt_len] = True

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model_runner(x_batch, position_ids=pos_batch, use_cache=False, attention_mask=attn_mask)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000

    for i, r in enumerate(reqs):
        bl_s, bl_e = r.block_start, r.block_end
        logits = output.logits[i, bl_s:bl_e]
        block_tokens = r.x[bl_s:bl_e]
        mask_idx = (block_tokens == MASK_ID)
        n_masked_before = mask_idx.sum().item()

        if not mask_idx.any():
            if (block_tokens == EOS_ID).any():
                r.x[bl_e:] = EOS_ID
                r.cur_block = r.n_blocks
            else:
                r.cur_block += 1
            continue

        x0 = logits.argmax(dim=-1)
        probs = F.softmax(logits.float(), dim=-1)
        x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        valid_mask = mask_idx & (x0 != MASK_ID)
        x0 = torch.where(mask_idx, x0, block_tokens)
        confidence = torch.where(valid_mask, x0_p, torch.tensor(-float('inf'), device=device))

        # Track confidence for CW-SRPT
        valid_conf = confidence[valid_mask]
        r.last_confidence_sum = valid_conf.sum().item() if len(valid_conf) > 0 else 0.0

        max_conf = confidence.max()
        actual_th = min(threshold, max_conf.item() - 1e-5)
        transfer = confidence >= actual_th
        r.x[bl_s:bl_e] = torch.where(transfer, x0, block_tokens)

        n_masked_after = (r.x[bl_s:bl_e] == MASK_ID).sum().item()
        n_unmasked = n_masked_before - n_masked_after
        r.last_n_unmasked = n_unmasked

        # Update convergence rate (EMA)
        alpha = 0.3
        r.convergence_rate = alpha * n_unmasked + (1 - alpha) * r.convergence_rate

        if n_unmasked > 0:
            r.iters_since_progress = 0
        else:
            r.iters_since_progress += 1

        if n_masked_after == 0:
            if (r.x[bl_s:bl_e] == EOS_ID).any():
                r.x[bl_e:] = EOS_ID
                r.cur_block = r.n_blocks
            else:
                r.cur_block += 1

        r.total_fwds += 1

    return elapsed


# ═══════════════════════════════════════════════════════════════════
# Scheduling Policies
# ═══════════════════════════════════════════════════════════════════

def pick_batch_fcfs(active, max_batch):
    return active[:max_batch]

def pick_batch_srpt(active, max_batch):
    active.sort(key=lambda r: r.remaining_naive)
    return active[:max_batch]

def pick_batch_cw_srpt(active, max_batch):
    """Confidence-Weighted SRPT: use confidence to predict remaining work."""
    active.sort(key=lambda r: r.remaining_confidence)
    return active[:max_batch]

def pick_batch_bab_srpt(active, max_batch):
    """Block-Aligned Batching: group by block index, then SRPT within group.
    Requests at same block have same seq_len → zero padding waste."""
    # Group by current block index
    groups = defaultdict(list)
    for r in active:
        groups[r.cur_block].append(r)

    # Within each group, sort by remaining (SRPT)
    for g in groups.values():
        g.sort(key=lambda r: r.remaining_naive)

    # Pick the group with most requests first (maximize batching)
    # Break ties by lowest remaining (SRPT spirit)
    sorted_groups = sorted(groups.items(),
                           key=lambda kv: (-len(kv[1]), min(r.remaining_naive for r in kv[1])))

    batch = []
    for _, group in sorted_groups:
        for r in group:
            if len(batch) >= max_batch:
                break
            batch.append(r)
        if len(batch) >= max_batch:
            break
    return batch

def pick_batch_appc(active, max_batch):
    """Aggressive Preemption with Progress Credit.
    Fast convergers get priority. Starvation-aware."""
    for r in active:
        if r.iters_since_progress > 20:
            # Starvation boost: if stuck for >20 iters, force priority
            r._appc_key = -1000 + r.iters_since_progress
        else:
            r._appc_key = r.progress_priority
    active.sort(key=lambda r: r._appc_key)
    return active[:max_batch]

POLICIES = {
    "FCFS": pick_batch_fcfs,
    "SRPT": pick_batch_srpt,
    "CW-SRPT": pick_batch_cw_srpt,
    "BAB-SRPT": pick_batch_bab_srpt,
    "APPC": pick_batch_appc,
}


# ═══════════════════════════════════════════════════════════════════
# Serving Engine
# ═══════════════════════════════════════════════════════════════════

def serve(reqs, model_runner, tokenizer, device, arrivals, policy_fn,
          max_batch=4, threshold=0.9):
    for r in reqs:
        r.init(device)

    sorted_reqs = sorted(zip(reqs, arrivals), key=lambda x: x[1])
    pending = deque(sorted_reqs)
    active = []
    done_list = []
    clock = 0.0
    total_iters = 0
    total_padding_waste = 0

    while pending or active:
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

        batch = policy_fn(active, max_batch)

        # Track padding waste (seq_len variation within batch)
        if len(batch) > 1:
            lens = [r.block_end for r in batch]
            max_l = max(lens)
            waste = sum(max_l - l for l in lens)
            total_padding_waste += waste

        elapsed = run_iter(batch, model_runner, device, threshold)
        clock += elapsed
        total_iters += 1

        newly_done = [r for r in batch if r.done]
        for r in newly_done:
            r.finish_ms = clock
            gen = r.x[r.prompt_len:]
            eos_pos = (gen == EOS_ID).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen = gen[:eos_pos[0]]
            r.output_text = tokenizer.decode(gen, skip_special_tokens=True)
            active.remove(r)
            done_list.append(r)

        while pending and pending[0][1] <= clock:
            r2, arr2 = pending.popleft()
            r2.arrival_ms = arr2
            if r2.first_fwd_ms < 0:
                r2.first_fwd_ms = clock
            active.append(r2)

    return done_list, clock, {
        "total_iters": total_iters,
        "padding_waste": total_padding_waste,
    }


# ══════════════════════════════════���════════════════════════════════
# Azure Trace Loading
# ═══════════════════════════════════════════════════════════════════

def load_azure_trace(path, n_reqs=50, scale=0.01, start_hour=2.0):
    """Load Azure trace, extract arrival times scaled to our throughput."""
    import datetime
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.datetime.fromisoformat(row['TIMESTAMP'].replace('+00:00', '+00:00'))
            ctx = int(row['ContextTokens'])
            gen = int(row['GeneratedTokens'])
            rows.append((ts, ctx, gen))
            if len(rows) > 500000:
                break

    # Filter to start_hour window
    t0 = rows[0][0]
    start_offset = datetime.timedelta(hours=start_hour)
    filtered = [(ts, ctx, gen) for ts, ctx, gen in rows
                if ts >= t0 + start_offset]

    # Scale: take every 1/scale-th request
    step = max(1, int(1.0 / scale))
    sampled = filtered[::step][:n_reqs]

    if not sampled:
        return [0.0] * n_reqs, [128] * n_reqs

    # Convert to relative arrival times in ms
    base_ts = sampled[0][0]
    arrivals = [(ts - base_ts).total_seconds() * 1000 for ts, _, _ in sampled]
    gen_lengths = [min(max(gen, 32), 256) for _, _, gen in sampled]

    return arrivals, gen_lengths


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-reqs', type=int, default=32)
    parser.add_argument('--gen-length', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--max-batch', type=int, default=8)
    parser.add_argument('--use-azure-trace', action='store_true')
    parser.add_argument('--arrival-rate', type=float, default=5.0)
    parser.add_argument('--output', type=str,
                        default='/home/zhujianian/dInfer/tests/diffserve_v3_results.json')
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

    dr_mod = load_dinfer()
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM

    model = LLaDA2SGLangLM(config=model_config, expert_map_path='.').eval()
    torch.set_default_dtype(torch.bfloat16)
    model.load_weights(MODEL_PATH, device=device)
    initialize_moe_config(server_args)
    model = model.to(device)
    model.after_processing()

    model_runner = dr_mod.ModelRunner(
        model, device, server_args=server_args,
        max_length=512, block_length=32,
        prefill_lengths=[32, 64, 96, 128],
        enable_cuda_graph=True,
        supported_batch_sizes=[1, 2, 4, 8],
        use_cross_block=True,
    )
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

    # Load prompts
    prompts = []
    with open('/home/zhujianian/morspec/data/humaneval.jsonl') as f:
        for line in f:
            d = json.loads(line)
            prompts.append(d.get('prompt', ''))
    # Repeat if needed
    while len(prompts) < args.n_reqs:
        prompts = prompts + prompts
    prompts = prompts[:args.n_reqs]

    def encode(text):
        full = (f'<role>SYSTEM</role>detailed thinking off<|role_end|>'
                f'<role>HUMAN</role>{text}<|role_end|><role>ASSISTANT</role>')
        return tokenizer.encode(full, return_tensors='pt').squeeze(0).to(device)

    # Get arrival times
    if args.use_azure_trace:
        azure_path = '/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv'
        arrivals, gen_lens = load_azure_trace(azure_path, args.n_reqs, scale=0.001)
        print(f"Azure trace: {len(arrivals)} arrivals, span={arrivals[-1]:.0f}ms")
    else:
        np.random.seed(42)
        inter = np.random.exponential(1000.0 / args.arrival_rate, size=args.n_reqs)
        arrivals = np.cumsum(inter).tolist()
        gen_lens = [args.gen_length] * args.n_reqs

    # Warmup
    print("Warming up...")
    wr = DReq(id=999, prompt_ids=encode(prompts[0]), gen_length=32, threshold=args.threshold)
    wr.init(device)
    for _ in range(5):
        run_iter([wr], model_runner, device, args.threshold)
    torch.cuda.synchronize()
    print("Warmup done.\n")

    all_results = {"config": vars(args)}

    # ─── Run all policies ─────────────────────────────────────────
    print("=" * 100)
    print(f"DiffServe v3: {args.n_reqs} reqs, max_batch={args.max_batch}, threshold={args.threshold}")
    print(f"Arrival: {'Azure trace' if args.use_azure_trace else f'Poisson rate={args.arrival_rate}'}")
    print("=" * 100)

    header = f"  {'Policy':>12s} {'Wall':>8s} {'MnLat':>8s} {'P50':>8s} {'P99':>8s} {'Iters':>6s} {'Waste':>6s} {'vsFC':>8s} {'vsSRPT':>8s}"
    print(header)

    fcfs_wall = None
    srpt_wall = None

    for pname, pfn in POLICIES.items():
        max_b = 1 if pname == "FCFS" else args.max_batch
        reqs = [DReq(id=i, prompt_ids=encode(prompts[i]),
                      gen_length=gen_lens[i] if i < len(gen_lens) else args.gen_length,
                      threshold=args.threshold)
                for i in range(args.n_reqs)]

        done, wall, meta = serve(reqs, model_runner, tokenizer, device,
                                 arrivals, pfn, max_b, args.threshold)

        lats = [(r.finish_ms - r.arrival_ms) for r in done]
        mean_lat = np.mean(lats) if lats else 0
        p50 = np.median(lats) if lats else 0
        p99 = np.percentile(lats, 99) if lats else 0

        if fcfs_wall is None:
            fcfs_wall = wall
        if pname == "SRPT":
            srpt_wall = wall

        vs_fcfs = (1 - wall / fcfs_wall) * 100 if fcfs_wall else 0
        vs_srpt = (1 - wall / srpt_wall) * 100 if srpt_wall else 0

        print(f"  {pname:>12s} {wall:>8.0f} {mean_lat:>8.0f} {p50:>8.0f} {p99:>8.0f} "
              f"{meta['total_iters']:>6d} {meta['padding_waste']:>6d} "
              f"{vs_fcfs:>+7.1f}% {vs_srpt:>+7.1f}%")

        all_results[pname] = {
            "wall_ms": wall, "mean_lat": float(mean_lat),
            "p50_lat": float(p50), "p99_lat": float(p99),
            "n_completed": len(done), **meta,
        }

    # ─── Quality check ────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("OUTPUT QUALITY (first 3)")
    print("=" * 100)
    # Compare FCFS vs best policy
    for pname in ["FCFS", "CW-SRPT", "BAB-SRPT", "APPC"]:
        reqs = [DReq(id=i, prompt_ids=encode(prompts[i]),
                      gen_length=args.gen_length, threshold=args.threshold)
                for i in range(3)]
        max_b = 1 if pname == "FCFS" else args.max_batch
        done, _, _ = serve(reqs, model_runner, tokenizer, device,
                           [0, 0, 0], POLICIES[pname], max_b, args.threshold)
        print(f"\n  {pname}:")
        for r in sorted(done, key=lambda r: r.id):
            print(f"    [{r.id}] {r.output_text[:100]}")

    # Save
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
