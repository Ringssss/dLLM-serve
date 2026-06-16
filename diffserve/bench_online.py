"""
DiffServe Online Benchmark.

Drives realistic traffic against the DiffServe engine using Azure traces
or synthetic Poisson arrivals. Supports two modes:

  - **direct**: Bypass HTTP, call engine.serve_batch_sync() directly.
    Best for fast iteration and apples-to-apples policy comparison.

  - **http**: Send real HTTP requests to a running DiffServe server.
    Measures end-to-end latency including network and serialization.

Usage:
  # Direct mode (no server needed)
  python -m diffserve.bench_online --mode direct --policy cw-srpt --n-reqs 32

  # HTTP mode (server must be running)
  python -m diffserve.bench_online --mode http --target http://localhost:8000 --n-reqs 32

  # Sweep all policies
  python -m diffserve.bench_online --mode direct --sweep --n-reqs 24

  # Use Azure traces
  python -m diffserve.bench_online --mode direct --trace azure --n-reqs 64
"""

import argparse
import asyncio
import csv
import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Azure Trace Loading
# ═══════════════════════════════════════════════════════════════════

AZURE_CONV_PATH = "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv"
AZURE_CODE_PATH = "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_code_1week.csv"


def load_azure_trace(
    path: str = AZURE_CONV_PATH,
    n_reqs: int = 50,
    scale: float = 0.01,
    start_hour: float = 2.0,
) -> Tuple[List[float], List[int]]:
    """Load Azure LLM Inference Trace and extract arrival times + gen lengths.

    Args:
        path: Path to Azure trace CSV.
        n_reqs: Number of requests to extract.
        scale: Sampling rate (1/scale-th of requests are kept).
        start_hour: Skip first N hours of trace (warm-up period).

    Returns:
        Tuple of (arrival_times_ms, gen_lengths).
    """
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.datetime.fromisoformat(
                row['TIMESTAMP'].replace('+00:00', '+00:00'))
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

    # Subsample
    step = max(1, int(1.0 / scale))
    sampled = filtered[::step][:n_reqs]

    if not sampled:
        return [0.0] * n_reqs, [128] * n_reqs

    # Convert to relative arrival times in ms
    base_ts = sampled[0][0]
    arrivals = [(ts - base_ts).total_seconds() * 1000 for ts, _, _ in sampled]
    gen_lengths = [min(max(gen, 32), 256) for _, _, gen in sampled]

    return arrivals, gen_lengths


def generate_poisson_arrivals(
    n_reqs: int,
    rate: float,
    seed: int = 42,
) -> List[float]:
    """Generate Poisson arrival times.

    Args:
        n_reqs: Number of requests.
        rate: Arrival rate (requests per second).
        seed: Random seed.

    Returns:
        List of arrival times in milliseconds.
    """
    np.random.seed(seed)
    inter = np.random.exponential(1000.0 / rate, size=n_reqs)
    return np.cumsum(inter).tolist()


# ═══════════════════════════════════════════════════════════════════
# Prompt Loading
# ═══════════════════════════════════════════════════════════════════

def load_prompts(dataset: str = "humaneval", n_reqs: int = 32) -> List[str]:
    """Load prompts from a dataset file.

    Args:
        dataset: Dataset name ("humaneval" or "gsm8k").
        n_reqs: Number of prompts needed.

    Returns:
        List of prompt strings.
    """
    paths = {
        "humaneval": "/home/zhujianian/morspec/data/humaneval.jsonl",
        "gsm8k": "/home/zhujianian/morspec/data/gsm8k.jsonl",
    }
    path = paths.get(dataset)
    if path is None or not os.path.exists(path):
        # Fallback: generate simple prompts
        return [
            f"Write a Python function that {task}."
            for task in [
                "checks if a number is prime",
                "sorts a list of integers",
                "finds the longest common subsequence",
                "implements binary search",
                "calculates the fibonacci sequence",
                "reverses a linked list",
                "validates parentheses",
                "finds duplicates in an array",
            ] * ((n_reqs + 7) // 8)
        ][:n_reqs]

    prompts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            prompts.append(d.get('prompt', d.get('question', '')))

    # Repeat if needed
    while len(prompts) < n_reqs:
        prompts = prompts + prompts
    return prompts[:n_reqs]


# ═══════════════════════════════════════════════════════════════════
# Direct Mode Benchmark
# ═══════════════════════════════════════════════════════════════════

def run_direct_benchmark(args):
    """Run benchmark in direct mode (no HTTP server)."""
    import torch
    from .config import DiffServeConfig
    from .model_loader import create_model_runner
    from .engine import DiffServeEngine
    from .request import DiffuseRequest
    from .scheduler import SchedulingPolicy

    config = DiffServeConfig(
        model_path=args.model,
        max_batch_size=args.max_batch,
        threshold=args.threshold,
  _seq_length=args.max_seq_length,
        device=args.device,
        use_kv_cache=getattr(args, 'use_kv_cache', False),
        use_foundry=getattr(args, 'use_foundry', False),
        foundry_archive_dir=getattr(args, 'foundry_archive_dir', ''),
    )

    # Load model
    print(f"Loading model from {config.model_path}...")
    model_runner, tokenizer = create_model_runner(config)
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    device = torch.device(config.device)

    # Load prompts
    prompts = load_prompts(args.dataset, args.n_reqs)

    def encode(text):
        full = (f'<role>SYSTEM</role>detailed thinking off<|role_end|>'
                f'<role>HUMAN</role>{text}<|role_end|><role>ASSISTANT</role>')
        return tokenizer.encode(full, return_tensors='pt').squeeze(0).to(device)

    # Get arrival times
    if args.trace == "azure":
        trace_path = args.trace_path or AZURE_CONV_PATH
        arrivals, gen_lengths = load_azure_trace(
            trace_path, args.n_reqs, scale=0.001)
        print(f"Azure trace: {len(arrivals)} arrivals, span={arrivals[-1]:.0f}ms")
    else:
        arrivals = generate_poisson_arrivals(
            args.n_reqs, args.arrival_rate)
        gen_lengths = [args.gen_length] * args.n_reqs

    # Warmup
    print("Warming up...")
    warmup_config = DiffServeConfig(**{**vars(config.__class__()),
                                       **{k: v for k, v in vars(config).items()}})
    warmup_req = DiffuseRequest(
        id=999, prompt_ids=encode(prompts[0]),
        gen_length=32, threshold=config.threshold, config=config)
    warmup_req.init()

    engine = DiffServeEngine(config, model_runner, tokenizer)
    warmup_done, _, _ = engine.serve_batch_sync(
        [warmup_req], [0.0])
    if warmup_done:
        print(f"  Warmup output: {warmup_done[0].output_text[:80]}")
    torch.cuda.synchronize()
    print("Warmup done.\n")

    # Determine which policies to run
    if args.sweep:
        policies = ["fcfs", "srpt", "cw-srpt", "bab-srpt", "appc"]
    else:
        policies = [args.policy]

    # ─── Run benchmark ────────────────────────────────────────────
    print("=" * 100)
    print(f"DiffServe Online Benchmark: {args.n_reqs} reqs, "
          f"max_batch={args.max_batch}, threshold={args.threshold}")
    print(f"Arrival: {'Azure trace' if args.trace == 'azure' else f'Poisson rate={args.arrival_rate}'}")
    print(f"Dataset: {args.dataset}")
    print("=" * 100)

    header = (f"  {'Policy':>12s} {'Wall':>8s} {'MnLat':>8s} {'P50':>8s} "
              f"{'P99':>8s} {'Iters':>6s} {'Waste':>6s} {'vsFC':>8s} {'vsSRPT':>8s}")
    print(header)

    all_results = {"config": vars(args)}
    fcfs_wall = None
    srpt_wall = None

    for pname in policies:
        config.policy = pname
        max_b = 1 if pname == "fcfs" else args.max_batch

        # Create fresh requests
        reqs = [
            DiffuseRequest(
                id=i,
                prompt_ids=encode(prompts[i]),
                gen_length=gen_lengths[i] if i < len(gen_lengths) else args.gen_length,
                threshold=config.threshold,
                config=config,
            )
            for i in range(args.n_reqs)
        ]
        for r in reqs:
            r.init()

        # Override max_batch for this policy
        orig_max_batch = config.max_batch_size
        config.max_batch_size = max_b
        engine_run = DiffServeEngine(config, model_runner, tokenizer)

        done, wall, meta = engine_run.serve_batch_sync(reqs, arrivals)
        config.max_batch_size = orig_max_batch

        lats = [(r.finish_time - r.arrival_time) for r in done]
        mean_lat = np.mean(lats) if lats else 0
        p50 = np.median(lats) if lats else 0
        p99 = np.percentile(lats, 99) if lats else 0

        if fcfs_wall is None:
            fcfs_wall = wall
        if pname == "srpt":
            srpt_wall = wall

        vs_fc = (1 - wall / fcfs_wall) * 100 if fcfs_wall else 0
        vs_sr = (1 - wall / srpt_wall) * 100 if srpt_wall else 0

        print(f"  {pname:>12s} {wall:>8.0f} {mean_lat:>8.0f} {p50:>8.0f} "
              f"{p99:>8.0f} {meta['total_iters']:>6d} "
              f"{meta['padding_waste']:>6d} {vs_fc:>+7.1f}% {vs_sr:>+7.1f}%")

        all_results[pname] = {
            "wall_ms": wall,
            "mean_lat": float(mean_lat),
            "p50_lat": float(p50),
            "p99_lat": float(p99),
            "n_completed": len(done),
            **meta,
        }

    # ─── Quality check ────────────────────────────────────────────
    if args.sweep and len(policies) > 1:
        print(f"\n{'=' * 100}")
        print("OUTPUT QUALITY (first 3 requests)")
        print("=" * 100)

        for pname in ["fcfs", "cw-srpt"]:
            config.policy = pname
            max_b = 1 if pname == "fcfs" else args.max_batch
            config.max_batch_size = max_b

            reqs = [
                DiffuseRequest(
                    id=i, prompt_ids=encode(prompts[i]),
                    gen_length=args.gen_length, threshold=config.threshold,
                    config=config,
                )
                for i in range(3)
            ]
            for r in reqs:
                r.init()

            engine_q = DiffServeEngine(config, model_runner, tokenizer)
            done, _, _ = engine_q.serve_batch_sync(reqs, [0, 0, 0])

            print(f"\n  {pname}:")
            for r in sorted(done, key=lambda r: r.id):
                print(f"    [{r.id}] {r.output_text[:100]}")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2,
                      default=lambda o: float(o) if hasattr(o, 'item') else str(o))
        print(f"\nResults saved to {args.output}")


# ═══════════════════════════════════════════════════════════════════
# HTTP Mode Benchmark
# ═══════════════════════════════════════════════════════════════════

async def run_http_benchmark(args):
    """Run benchmark by sending real HTTP requests to a running server."""
    try:
        import aiohttp
    except ImportError:
        print("aiohttp required for HTTP mode: pip install aiohttp")
        return

    target = args.target.rstrip('/')
    prompts = load_prompts(args.dataset, args.n_reqs)

    # Get arrival times
    if args.trace == "azure":
        trace_path = args.trace_path or AZURE_CONV_PATH
        arrivals, gen_lengths = load_azure_trace(
            trace_path, args.n_reqs, scale=0.001)
    else:
        arrivals = generate_poisson_arrivals(
            args.n_reqs, args.arrival_rate)
        gen_lengths = [args.gen_length] * args.n_reqs

    results = []
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = []

        for i in range(args.n_reqs):
            # Schedule request at arrival time
            delay = arrivals[i] / 1000.0  # convert ms to seconds
            task = asyncio.create_task(
                _send_request(
                    session, target, prompts[i],
                    gen_lengths[i] if i < len(gen_lengths) else args.gen_length,
                    delay, i))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    total_wall = (time.perf_counter() - start_time) * 1000

    # Report
    latencies = [r['latency_ms'] for r in results if r['success']]
    n_ok = len(latencies)
    n_fail = len(results) - n_ok

    print(f"\n{'=' * 80}")
    print(f"HTTP Benchmark Results: {target}")
    print(f"{'=' * 80}")
    print(f"  Total wall: {total_wall:.0f} ms")
    print(f"  Completed:  {n_ok}/{args.n_reqs} ({n_fail} failures)")
    if latencies:
        print(f"  Mean lat:   {np.mean(latencies):.0f} ms")
        print(f"  P50 lat:    {np.median(latencies):.0f} ms")
        print(f"  P99 lat:    {np.percentile(latencies, 99):.0f} ms")


async def _send_request(
    session, target: str, prompt: str, gen_length: int,
    delay: float, req_id: int,
) -> dict:
    """Send a single HTTP request after waiting for the arrival delay."""
    await asyncio.sleep(delay)
    t0 = time.perf_counter()

    try:
        async with session.post(
            f"{target}/v1/completions",
            json={"prompt": prompt, "max_tokens": gen_length},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            latency = (time.perf_counter() - t0) * 1000
            return {
                "id": req_id,
                "success": True,
                "latency_ms": latency,
                "output": data.get("choices", [{}])[0].get("text", ""),
            }
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return {
            "id": req_id,
            "success": False,
            "latency_ms": latency,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DiffServe Online Benchmark")

    parser.add_argument('--mode', choices=['direct', 'http'], default='direct',
                        help='Benchmark mode')
    parser.add_argument('--model', default='/mnt/models/LLaDA2.0-mini',
                        help='Model path (direct mode)')
    parser.add_argument('--target', default='http://localhost:8000',
                        help='Server URL (http mode)')
    parser.add_argument('--n-reqs', type=int, default=32)
    parser.add_argument('--gen-length', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--max-batch', type=int, default=8)
    parser.add_argument('--max-seq-length', type=int, default=512)
    parser.add_argument('--policy', default='cw-srpt',
                        choices=['fcfs', 'srpt', 'cw-srpt', 'bab-srpt', 'appc'])
    parser.add_argument('--sweep', action='store_true',
                        help='Run all policies for comparison')
    parser.add_argument('--trace', choices=['poisson', 'azure'], default='poisson',
                        help='Arrival pattern')
    parser.add_argument('--trace-path', default=None,
                        help='Path to Azure trace CSV')
    parser.add_argument('--arrival-rate', type=float, default=5.0,
                        help='Poisson arrival rate (reqs/sec)')
    parser.add_argument('--dataset', default='humaneval',
                        choices=['humaneval', 'gsm8k'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--use-kv-cache', action='store_true',
                        help='Enable KV cache for decode iterations')
    parser.add_argument('--output', default=None,
                        help='Path to save results JSON')
    args = parser.parse_args()

    if args.output is None:
        args.output = f'/home/zhujianian/dInfer/diffserve/bench_results.json'

    if args.mode == 'direct':
        run_direct_benchmark(args)
    elif args.mode == 'http':
        asyncio.run(run_http_benchmark(args))


if __name__ == "__main__":
    main()
