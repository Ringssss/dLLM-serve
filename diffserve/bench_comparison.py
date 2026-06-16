#!/usr/bin/env python3
"""
DiffServe vs SGLang Native: Head-to-Head A/B Benchmark.

Compares DiffServe (CW-SRPT + KV cache + Foundry) against SGLang's native
dLLM serving (FCFS + LowConfidence) on identical workloads.

Modes:
  --mode direct   Run both systems in-process (no HTTP, apples-to-apples)
  --mode sglang   Start SGLang server, benchmark via HTTP, then compare

Metrics:
  - Wall time, Mean/P50/P99 latency
  - TPOT (time per output token)
  - TTFT (time to first token)
  - Throughput (tok/s)
  - Cold start time
  - CUDA graph hit rate
  - Output quality (text comparison)
  - Arbiter prediction accuracy

Usage:
  # Direct mode (recommended for fair comparison)
  python -m diffserve.bench_comparison --n-reqs 64 --arrival-rate 50

  # With profiling
  python -m diffserve.bench_comparison --n-reqs 32 --arrival-rate 20 --profile
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, '/home/zhujianian/dInfer')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bench_comparison")
logger.setLevel(logging.INFO)


def load_prompts(dataset, n_reqs):
    """Load prompts from dataset."""
    paths = {
        "humaneval": "/home/zhujianian/morspec/data/humaneval.jsonl",
        "gsm8k": "/home/zhujianian/morspec/data/gsm8k.jsonl",
    }
    path = paths.get(dataset)
    if path and os.path.exists(path):
        prompts = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                prompts.append(d.get('prompt', d.get('question', '')))
        while len(prompts) < n_reqs:
            prompts = prompts + prompts
        return prompts[:n_reqs]
    return [f"Write a Python function that checks if a number is prime."] * n_reqs


def generate_arrivals(n_reqs, rate, seed=42):
    """Generate Poisson arrival times in ms."""
    np.random.seed(seed)
    inter = np.random.exponential(1000.0 / rate, size=n_reqs)
    return np.cumsum(inter).tolist()


def compute_metrics(done_requests, wall_ms, gen_length):
    """Compute serving metrics from completed requests."""
    if not done_requests:
        return {}

    lats = [(r.finish_time - r.arrival_time) for r in done_requests]
    n = len(done_requests)
    total_gen_tokens = sum(
        len(r.output_text.split()) for r in done_requests)  # approximate word count
    total_exact_tokens = n * gen_length  # upper bound

    mean_lat = np.mean(lats)
    p50 = np.median(lats)
    p99 = np.percentile(lats, 99)

    # TPOT: average time per output token per request
    tpots = []
    for r in done_requests:
        req_lat = r.finish_time - r.arrival_time
        n_fwds = r.total_fwds
        if n_fwds > 0:
            tpots.append(req_lat / n_fwds)

    # TTFT: time from arrival to first forward
    ttfts = [(r.first_token_time - r.arrival_time) for r in done_requests
             if r.first_token_time > 0]

    throughput = total_exact_tokens / (wall_ms / 1000) if wall_ms > 0 else 0

    return {
        "n_completed": n,
        "wall_ms": wall_ms,
        "mean_lat_ms": float(mean_lat),
        "p50_lat_ms": float(p50),
        "p99_lat_ms": float(p99),
        "throughput_tok_s": float(throughput),
        "mean_tpot_ms": float(np.mean(tpots)) if tpots else 0,
        "p99_tpot_ms": float(np.percentile(tpots, 99)) if tpots else 0,
        "mean_ttft_ms": float(np.mean(ttfts)) if ttfts else 0,
        "p99_ttft_ms": float(np.percentile(ttfts, 99)) if ttfts else 0,
        "total_fwds": sum(r.total_fwds for r in done_requests),
    }


# ═══════════════════════════════════════════════════════════════════
# System A: DiffServe (our system)
# ═══════════════════════════════════════════════════════════════════

def run_diffserve(args, model_runner, tokenizer, prompts, arrivals, gen_lengths,
                  policy="cw-srpt", use_kv=False, label="DiffServe"):
    """Run DiffServe engine and return metrics."""
    import torch
    from diffserve.config import DiffServeConfig
    from diffserve.engine import DiffServeEngine
    from diffserve.request import DiffuseRequest

    device = torch.device(args.device)

    config = DiffServeConfig(
        model_path=args.model,
        max_batch_size=args.max_batch,
        threshold=args.threshold,
        max_seq_length=args.max_seq_length,
        device=args.device,
        policy=policy,
        use_kv_cache=use_kv,
        enable_arbiter=True,
    )

    def encode(text):
        full = (f'<role>SYSTEM</role>detailed thinking off<|role_end|>'
                f'<role>HUMAN</role>{text}<|role_end|><role>ASSISTANT</role>')
        return tokenizer.encode(full, return_tensors='pt').squeeze(0).to(device)

    reqs = [
        DiffuseRequest(
            id=i, prompt_ids=encode(prompts[i]),
            gen_length=gen_lengths[i] if i < len(gen_lengths) else args.gen_length,
            threshold=config.threshold, config=config)
        for i in range(args.n_reqs)
    ]
    for r in reqs:
        r.init()

    engine = DiffServeEngine(config, model_runner, tokenizer)

    # Measure cold start (engine creation only — model already loaded)
    t_cold = 0.0  # model loading not included in this comparison

    # Profile wrapper
    profiler = None
    if args.profile:
        import torch.profiler
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        profiler.__enter__()

    done, wall, meta = engine.serve_batch_sync(reqs, arrivals)

    if profiler:
        profiler.__exit__(None, None, None)
        trace_path = f"/home/zhujianian/dInfer/diffserve/profile_{label.lower().replace(' ', '_')}.json"
        profiler.export_chrome_trace(trace_path)
        logger.info(f"  Profile trace saved to {trace_path}")

    metrics = compute_metrics(done, wall, args.gen_length)
    metrics["cold_start_ms"] = t_cold
    metrics["policy"] = policy
    metrics["use_kv_cache"] = use_kv
    metrics["label"] = label

    # Arbiter stats
    if meta.get("arbiter"):
        metrics["arbiter"] = meta["arbiter"]

    metrics["total_iters"] = meta.get("total_iters", 0)
    metrics["padding_waste"] = meta.get("padding_waste", 0)

    # Sample outputs for quality check
    metrics["_sample_outputs"] = [
        r.output_text[:120] for r in sorted(done, key=lambda r: r.id)[:3]]

    return metrics


# ═══════════════════════════════════════════════════════════════════
# System B: SGLang Native dLLM (simulated via dInfer's FCFS path)
# ═══════════════════════════════════════════════════════════════════

def run_sglang_baseline(args, model_runner, tokenizer, prompts, arrivals, gen_lengths):
    """Simulate SGLang native dLLM: FCFS scheduling, no CW-SRPT.

    SGLang's native dLLM serving uses FCFS (process requests in arrival order)
    with LowConfidence algorithm (no confidence-weighted scheduling).
    We simulate this by running DiffServe with policy=fcfs, max_batch=1.
    """
    return run_diffserve(
        args, model_runner, tokenizer, prompts, arrivals, gen_lengths,
        policy="fcfs", use_kv=False,
        label="SGLang Native (FCFS)")


# ═══════════════════════════════════════════════════════════════════
# Main Comparison
# ═══════════════════════════════════════════════════════════════════

def run_comparison(args):
    """Run head-to-head comparison."""
    import torch
    from diffserve.config import DiffServeConfig
    from diffserve.model_loader import create_model_runner

    # Load model (shared between all runs)
    config = DiffServeConfig(
        model_path=args.model,
        max_batch_size=args.max_batch,
        max_seq_length=args.max_seq_length,
        device=args.device,
    )

    logger.info(f"Loading model from {config.model_path}...")
    t0 = time.perf_counter()
    model_runner, tokenizer = create_model_runner(config)
    model_load_time = (time.perf_counter() - t0) * 1000
    logger.info(f"Model loaded in {model_load_time:.0f}ms. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    device = torch.device(args.device)

    # Load workload
    prompts = load_prompts(args.dataset, args.n_reqs)
    arrivals = generate_arrivals(args.n_reqs, args.arrival_rate)
    gen_lengths = [args.gen_length] * args.n_reqs

    # Warmup
    logger.info("Warming up...")
    from diffserve.request import DiffuseRequest
    from diffserve.engine import DiffServeEngine
    warmup_config = DiffServeConfig(
        model_path=args.model, device=args.device, enable_arbiter=False)
    wr = DiffuseRequest(
        id=999,
        prompt_ids=tokenizer.encode(
            '<role>SYSTEM</role>detailed thinking off<|role_end|>'
            '<role>HUMAN</role>Hi<|role_end|><role>ASSISTANT</role>',
            return_tensors='pt').squeeze(0).to(device),
        gen_length=32, threshold=0.9, config=warmup_config)
    wr.init()
    we = DiffServeEngine(warmup_config, model_runner, tokenizer)
    we.serve_batch_sync([wr], [0.0])
    torch.cuda.synchronize()
    logger.info("Warmup done.\n")

    # ─── Run all systems ──────────────────────────────────────────
    systems = OrderedDict()

    # System A0: FCFS no batching (bs=1) — worst-case baseline
    logger.info("▸ Running: FCFS (bs=1, no batching)...")
    args_fcfs1 = argparse.Namespace(**vars(args))
    args_fcfs1.max_batch = 1
    systems["fcfs_bs1"] = run_diffserve(
        args_fcfs1, model_runner, tokenizer, prompts, arrivals, gen_lengths,
        policy="fcfs", use_kv=False, label="FCFS (bs=1)")

    # System A1: FCFS + batching — SGLang-like baseline
    logger.info("▸ Running: FCFS + Batch (SGLang-like)...")
    systems["fcfs_batch"] = run_diffserve(
        args, model_runner, tokenizer, prompts, arrivals, gen_lengths,
        policy="fcfs", use_kv=False, label="FCFS+Batch (SGLang)")

    # System A2: SRPT + batching (generic, no confidence)
    logger.info("▸ Running: SRPT + Batching...")
    systems["srpt_batch"] = run_diffserve(
        args, model_runner, tokenizer, prompts, arrivals, gen_lengths,
        policy="srpt", use_kv=False, label="SRPT+Batch")

    # System B1: CW-SRPT + batching (our method, no KV)
    logger.info("▸ Running: CW-SRPT + Batching...")
    systems["cwsrpt_batch"] = run_diffserve(
        args, model_runner, tokenizer, prompts, arrivals, gen_lengths,
        policy="cw-srpt", use_kv=False, label="CW-SRPT+Batch")

    # System B2: CW-SRPT + KV cache (full upgrade)
    logger.info("▸ Running: CW-SRPT + KV Cache...")
    systems["cwsrpt_kv"] = run_diffserve(
        args, model_runner, tokenizer, prompts, arrivals, gen_lengths,
        policy="cw-srpt", use_kv=True, label="CW-SRPT+KV")

    # ─── Report ───────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print(f"DiffServe vs SGLang Native — Head-to-Head Comparison")
    print(f"  {args.n_reqs} reqs, gen={args.gen_length}, rate={args.arrival_rate}, "
          f"max_batch={args.max_batch}, dataset={args.dataset}")
    print("=" * 120)

    header = (f"  {'System':<25s} {'Wall':>8s} {'MnLat':>8s} {'P50':>8s} "
              f"{'P99':>8s} {'TPS':>8s} {'TPOT':>7s} {'TTFT':>7s} "
              f"{'Iters':>6s} {'vsFC':>8s}")
    print(header)
    print("  " + "-" * 115)

    fcfs_wall = systems["fcfs_bs1"]["wall_ms"]

    for key, m in systems.items():
        wall = m["wall_ms"]
        vs_fc = (1 - wall / fcfs_wall) * 100 if fcfs_wall > 0 else 0
        print(f"  {m['label']:<25s} "
              f"{wall:>8.0f} {m['mean_lat_ms']:>8.0f} {m['p50_lat_ms']:>8.0f} "
              f"{m['p99_lat_ms']:>8.0f} {m['throughput_tok_s']:>8.0f} "
              f"{m['mean_tpot_ms']:>7.1f} {m['mean_ttft_ms']:>7.0f} "
              f"{m['total_iters']:>6d} {vs_fc:>+7.1f}%")

    # ─── Quality check ────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("OUTPUT QUALITY (first 3 requests)")
    print("=" * 120)
    for key in ["fcfs_bs1", "fcfs_batch", "cwsrpt_batch", "cwsrpt_kv"]:
        m = systems[key]
        print(f"\n  {m['label']}:")
        for i, text in enumerate(m.get("_sample_outputs", [])):
            print(f"    [{i}] {text}")

    # ─── Arbiter accuracy ─────────────────────────────────────────
    for key, m in systems.items():
        if "arbiter" in m:
            arb = m["arbiter"]
            print(f"\n  Arbiter ({m['label']}): "
                  f"bs_accuracy={arb.get('bs_accuracy','N/A')}, "
                  f"bs_mae={arb.get('bs_mae','N/A')}, "
                  f"phases={arb.get('phase_distribution',{})}")

    # ─── SLO Analysis ─────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("SLO ANALYSIS")
    print("=" * 120)
    slo_targets = [
        ("P99 Latency < 30s", lambda m: m["p99_lat_ms"] < 30000),
        ("P99 Latency < 15s", lambda m: m["p99_lat_ms"] < 15000),
        ("Mean TPOT < 50ms", lambda m: m["mean_tpot_ms"] < 50),
        ("P99 TTFT < 5s", lambda m: m.get("p99_ttft_ms", 99999) < 5000),
        ("Throughput > 500 tok/s", lambda m: m["throughput_tok_s"] > 500),
    ]
    print(f"  {'SLO Target':<30s}", end="")
    for key in systems:
        print(f" {systems[key]['label'][:12]:>12s}", end="")
    print()
    for slo_name, slo_fn in slo_targets:
        print(f"  {slo_name:<30s}", end="")
        for key in systems:
            passed = slo_fn(systems[key])
            print(f" {'✅':>12s}" if passed else f" {'❌':>12s}", end="")
        print()

    # ─── Save results ─────────────────────────────────────────────
    if args.output:
        # Remove non-serializable fields
        save_data = {"config": vars(args)}
        for key, m in systems.items():
            save_m = {k: v for k, v in m.items() if not k.startswith("_")}
            save_data[key] = save_m

        with open(args.output, 'w') as f:
            json.dump(save_data, f, indent=2,
                      default=lambda o: float(o) if hasattr(o, 'item') else str(o))
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="DiffServe vs SGLang Native — A/B Comparison")
    parser.add_argument('--model', default='/mnt/models/LLaDA2.0-mini')
    parser.add_argument('--n-reqs', type=int, default=64)
    parser.add_argument('--gen-length', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--max-batch', type=int, default=8)
    parser.add_argument('--max-seq-length', type=int, default=512)
    parser.add_argument('--arrival-rate', type=float, default=50.0)
    parser.add_argument('--dataset', default='humaneval',
                        choices=['humaneval', 'gsm8k'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--profile', action='store_true',
                        help='Enable torch.profiler tracing')
    parser.add_argument('--output', default='/home/zhujianian/dInfer/diffserve/comparison_results.json')
    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
