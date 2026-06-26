#!/usr/bin/env python3
"""
Benchmark: CW-SRPT V3 (three-pronged) vs LowConfidence vs CW-SRPT V2.

Runs real SGLang HTTP serving. Measures P90/P99 latency, TPS, TPOT.
Also runs output quality check on 4 datasets.

Usage:
  python codex_coding/src/cwsrpt_v3/bench_v3.py --tp 1 --reqs 64 --rates 5,10,20
  python codex_coding/src/cwsrpt_v3/bench_v3.py --tp 2 --reqs 64
  python codex_coding/src/cwsrpt_v3/bench_v3.py --quality-only
"""
import argparse
import concurrent.futures
import json
import numpy as np
import os
import requests
import signal
import subprocess
import sys
import threading
import time

PYTHON = "/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"
MODEL = "/mnt/models/LLaDA2.0-mini"
PORT = 30100
RESULTS_DIR = "/home/zhujianian/dInfer/codex_coding/results/cwsrpt_v3"

# ─── Helpers ──────────────────────────────────────────────────

def mk_prompt(text):
    return (
        '<role>SYSTEM</role>detailed thinking off<|role_end|>'
        '<role>HUMAN</role>' + text + '<|role_end|><role>ASSISTANT</role>'
    )

def send_one(prompt_text, max_tokens=128):
    t0 = time.perf_counter()
    r = requests.post(
        f"http://localhost:{PORT}/v1/completions",
        json={"model": MODEL, "prompt": mk_prompt(prompt_text),
              "max_tokens": max_tokens, "temperature": 0},
        timeout=120)
    lat = (time.perf_counter() - t0) * 1000
    data = r.json()
    tok = data.get("usage", {}).get("completion_tokens", 0)
    txt = data.get("choices", [{}])[0].get("text", "")
    return lat, tok, txt

def start_server(algo, tp):
    gpus = ",".join(str(i) for i in range(tp))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    proc = subprocess.Popen(
        [PYTHON, "-m", "sglang.launch_server",
         "--model-path", MODEL,
         "--dllm-algorithm", algo,
         "--max-running-requests", "8",
         "--disable-radix-cache",
         "--trust-remote-code",
         "--port", str(PORT),
         "--tp-size", str(tp)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            if requests.get(f"http://localhost:{PORT}/v1/models", timeout=2).status_code == 200:
                return proc
        except:
            pass
        time.sleep(2)
    raise RuntimeError(f"Server {algo} TP={tp} failed to start")

def stop_server(proc):
    os.kill(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
    time.sleep(3)

def load_prompts(path, n):
    prompts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            prompts.append(d.get('prompt', d.get('question', d.get('turns', [''])[0] if 'turns' in d else '')))
    while len(prompts) < n:
        prompts = prompts + prompts
    return prompts[:n]

def bench_sustained(prompts, n_reqs, rate):
    results = []
    np.random.seed(42)
    delays = np.cumsum(np.random.exponential(1/rate, n_reqs)).tolist()
    def go(idx, delay):
        time.sleep(delay)
        lat, tok, txt = send_one(prompts[idx % len(prompts)])
        results.append((lat, tok))
    threads = [threading.Thread(target=go, args=(i, delays[i])) for i in range(n_reqs)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = (time.perf_counter() - t0) * 1000
    lats = [r[0] for r in results]
    toks = [r[1] for r in results]
    return {
        "wall": wall,
        "tps": sum(toks) / (wall / 1000) if wall > 0 else 0,
        "p50": float(np.median(lats)),
        "p90": float(np.percentile(lats, 90)),
        "p99": float(np.percentile(lats, 99)),
        "mean": float(np.mean(lats)),
    }

# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--reqs", type=int, default=64)
    parser.add_argument("--rates", default="5,10,20")
    parser.add_argument("--algos", default="LowConfidence,CW_SRPT_V2,CW_SRPT_V3")
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rates = [int(r) for r in args.rates.split(",")]
    algos = args.algos.split(",")

    prompts = load_prompts("/home/zhujianian/morspec/data/humaneval.jsonl", args.reqs)

    all_results = {}

    for algo in algos:
        tag = f"TP{args.tp}_{algo}"
        print(f"\n{'='*80}")
        print(f"  {tag}: starting server...")
        print(f"{'='*80}")

        try:
            proc = start_server(algo, args.tp)
            print(f"  Server ready.")

            # Warmup
            send_one(prompts[0])
            send_one(prompts[1])

            # Single request TPOT
            lat, tok, txt = send_one(prompts[0])
            tpot = lat / max(tok, 1)
            print(f"  Single: {lat:.0f}ms, TPOT={tpot:.1f}ms | {txt[:50]}")

            algo_results = {"single_tpot": tpot, "single_lat": lat}

            if not args.quality_only:
                for rate in rates:
                    print(f"  rate={rate}/s...", end=" ", flush=True)
                    r = bench_sustained(prompts, args.reqs, rate)
                    print(f"TPS={r['tps']:.0f} P50={r['p50']:.0f} P90={r['p90']:.0f} P99={r['p99']:.0f}")
                    algo_results[f"rate_{rate}"] = r

            # Quality check
            if args.quality_only or algo == algos[-1]:
                print(f"\n  Quality check...")
                datasets = {
                    "humaneval": "/home/zhujianian/morspec/data/humaneval.jsonl",
                    "gsm8k": "/home/zhujianian/morspec/data/gsm8k.jsonl",
                    "mgsm": "/home/zhujianian/morspec/data/mgsm.jsonl",
                    "mt_bench": "/home/zhujianian/morspec/data/mt_bench.jsonl",
                }
                quality = {}
                for ds_name, ds_path in datasets.items():
                    ds_prompts = load_prompts(ds_path, 4)
                    outputs = []
                    for p in ds_prompts:
                        _, _, txt = send_one(p)
                        outputs.append(txt[:100])
                    readable = sum(1 for o in outputs if len(o.strip()) > 10)
                    quality[ds_name] = {"readable": f"{readable}/4", "samples": outputs[:2]}
                    print(f"    {ds_name}: {readable}/4 readable | {outputs[0][:60]}")
                algo_results["quality"] = quality

            all_results[tag] = algo_results
            stop_server(proc)

        except Exception as e:
            print(f"  ERROR: {e}")
            try:
                stop_server(proc)
            except:
                pass

    # ─── Comparison table ─────────────────────────────────────
    if len(all_results) >= 2 and not args.quality_only:
        print(f"\n{'='*100}")
        print(f"  COMPARISON: TP={args.tp}, {args.reqs} reqs")
        print(f"{'='*100}")

        baseline_tag = f"TP{args.tp}_LowConfidence"
        for rate in rates:
            rk = f"rate_{rate}"
            print(f"\n  rate={rate}/s:")
            for tag, res in all_results.items():
                if rk in res:
                    r = res[rk]
                    base = all_results.get(baseline_tag, {}).get(rk, {})
                    tps_d = f"({(r['tps']/base['tps']-1)*100:+.0f}%)" if base.get('tps') else ""
                    p90_d = f"({(1-r['p90']/base['p90'])*100:+.0f}%)" if base.get('p90') else ""
                    print(f"    {tag:<30s} TPS={r['tps']:>6.0f} {tps_d:>8s}  P90={r['p90']:>7.0f} {p90_d:>8s}  P99={r['p99']:>7.0f}")

    # Save
    out_path = f"{RESULTS_DIR}/bench_tp{args.tp}_{'quality' if args.quality_only else 'full'}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
