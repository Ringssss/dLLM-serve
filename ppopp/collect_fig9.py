#!/usr/bin/env python3
"""
Fig. 9 data collection: 30-min trace replay for throughput-over-time.

Runs SGLang (LowConfidence) and Ours (CW_SRPT_V3) under Kimi and Azure traces
for 30 minutes each, recording per-minute throughput (completed req/s).

Uses 4 GPU slots in parallel:
  GPU 0: SGLang + Kimi trace (port 30100)
  GPU 1: Ours + Kimi trace (port 30200)
  GPU 2: SGLang + Azure trace (port 30300)
  GPU 3: Ours + Azure trace (port 30400)
"""
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Thread

import numpy as np
import requests as http_req

PYTHON = "/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"
MODEL = "/mnt/models/LLaDA2.0-mini"
OUT_DIR = Path("/home/zhujianian/sglang/ppopp")

KIMI_TRACE = "/mnt/models/kimik25/kimi-k25-trace/kimi_k25_conv_1day.csv"
AZURE_TRACE = "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv"

GSM8K_PATH = "/home/zhujianian/morspec/data/gsm8k.jsonl"
HUMANEVAL_PATH = "/home/zhujianian/morspec/data/humaneval.jsonl"
MT_BENCH_PATH = "/home/zhujianian/morspec/data/mt_bench.jsonl"

GEN_LENGTH = 128
DURATION_MIN = 30


def load_prompts(n=2000):
    """Load a large pool of prompts for 30-min replay."""
    prompts = []
    for path in [GSM8K_PATH, HUMANEVAL_PATH, MT_BENCH_PATH]:
        if not Path(path).exists():
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                text = d.get("question", d.get("prompt", ""))
                if not text:
                    turns = d.get("turns", [])
                    if turns:
                        text = turns[0]
                if text:
                    prompts.append(text[:512])
    # Cycle to get enough
    while len(prompts) < n:
        prompts.extend(prompts[:n - len(prompts)])
    return prompts[:n]


def load_trace_arrivals(trace_path, duration_s=1800, target_rate=None):
    """Load inter-arrival times from trace CSV for `duration_s` seconds.

    If target_rate is specified, scale trace to achieve that mean rate.
    Returns list of (relative_time_s, prompt_index) tuples.
    """
    timestamps = []
    with open(trace_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["TIMESTAMP"].replace("+00:00", "+00:00"))
                timestamps.append(ts.timestamp())
            except Exception:
                continue
            if len(timestamps) > 500000:
                break

    timestamps.sort()

    # Find a busy 30-min window
    window_s = duration_s
    best_start = 0
    best_count = 0
    j = 0
    for i in range(len(timestamps)):
        while j < len(timestamps) and timestamps[j] - timestamps[i] <= window_s:
            j += 1
        if j - i > best_count:
            best_count = j - i
            best_start = i

    # Extract relative times
    window = timestamps[best_start:best_start + best_count]
    t0 = window[0]
    arrivals = [(t - t0, idx % 2000) for idx, t in enumerate(window)]

    # Scale if needed to hit target rate
    if target_rate and len(arrivals) > 1:
        actual_rate = len(arrivals) / (arrivals[-1][0] - arrivals[0][0] + 1)
        if actual_rate > 0:
            scale = actual_rate / target_rate
            arrivals = [(t * scale, idx) for t, idx in arrivals]
            # Trim to duration
            arrivals = [(t, idx) for t, idx in arrivals if t <= duration_s]

    return arrivals


def mk_prompt(text):
    return (
        '<role>SYSTEM</role>detailed thinking off<|role_end|>'
        '<role>HUMAN</role>' + text + '<|role_end|><role>ASSISTANT</role>'
    )


def wait_server(port, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = http_req.get(f"http://127.0.0.1:{port}/health", timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(3)
    return False


def start_server(algo, gpu, port):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ["http_proxy", "https_proxy", "all_proxy", "ALL_PROXY"]:
        env.pop(k, None)
    env["no_proxy"] = "127.0.0.1,localhost"

    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL,
           "--dllm-algorithm", algo,
           "--max-running-requests", "8",
           "--disable-radix-cache", "--trust-remote-code",
           "--port", str(port)]

    log = OUT_DIR / f"server_{algo}_gpu{gpu}.log"
    proc = subprocess.Popen(cmd, env=env,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT)
    print(f"  [{algo} GPU{gpu} port={port}] starting (PID {proc.pid})...")
    if not wait_server(port):
        print(f"  [{algo} GPU{gpu}] FAILED to start!")
        proc.kill()
        return None
    print(f"  [{algo} GPU{gpu}] ready.")
    return proc


def run_trace_replay(port, prompts, arrivals, duration_s):
    """Replay trace arrivals, record per-minute completed requests.

    Returns: dict with per-minute throughput array (req/s per minute).
    """
    n_minutes = duration_s // 60
    completions_per_minute = [0] * n_minutes
    completion_times = []

    t_start = time.time()

    def send_one(prompt_text, scheduled_time):
        # Wait until scheduled time
        now = time.time() - t_start
        if scheduled_time > now:
            time.sleep(scheduled_time - now)

        payload = {"model": "default", "prompt": mk_prompt(prompt_text),
                   "max_tokens": GEN_LENGTH, "temperature": 0}
        t0 = time.time()
        try:
            r = http_req.post(f"http://127.0.0.1:{port}/v1/completions",
                              json=payload, timeout=300)
            if r.status_code == 200:
                finish_time = time.time() - t_start
                minute_bucket = min(int(finish_time / 60), n_minutes - 1)
                completions_per_minute[minute_bucket] += 1
                completion_times.append(finish_time)
        except:
            pass

    # Submit all requests following trace timing
    with ThreadPoolExecutor(max_workers=128) as pool:
        futures = []
        for arr_time, prompt_idx in arrivals:
            if arr_time > duration_s:
                break
            futures.append(pool.submit(send_one, prompts[prompt_idx], arr_time))

        # Wait for all to complete (with timeout)
        for f in as_completed(futures, timeout=duration_s + 120):
            pass

    # Convert to req/s per minute
    throughput_per_min = [c / 60.0 for c in completions_per_minute]
    return {
        "throughput_per_min": throughput_per_min,
        "total_completed": sum(completions_per_minute),
        "total_submitted": min(len(arrivals), len([a for a, _ in arrivals if a <= duration_s])),
    }


def run_experiment(algo, gpu, port, trace_name, trace_path, prompts, target_rate=10.0):
    """Run one full 30-min experiment."""
    print(f"\n  [{algo} | {trace_name}] Loading trace (target ~{target_rate} req/s)...")
    arrivals = load_trace_arrivals(trace_path, duration_s=DURATION_MIN * 60,
                                    target_rate=target_rate)
    print(f"  [{algo} | {trace_name}] {len(arrivals)} arrivals over {DURATION_MIN} min")

    # Start server
    proc = start_server(algo, gpu, port)
    if proc is None:
        return None

    # Warmup
    try:
        http_req.post(f"http://127.0.0.1:{port}/v1/completions",
                      json={"model": "default", "prompt": mk_prompt(prompts[0]),
                            "max_tokens": 32, "temperature": 0}, timeout=60)
    except:
        pass

    print(f"  [{algo} | {trace_name}] Starting {DURATION_MIN}-min replay...")
    result = run_trace_replay(port, prompts, arrivals,
                               duration_s=DURATION_MIN * 60)

    # Kill server
    os.system(f"kill {proc.pid} 2>/dev/null")
    time.sleep(2)

    result["algo"] = algo
    result["trace"] = trace_name
    result["target_rate"] = target_rate
    print(f"  [{algo} | {trace_name}] Done. "
          f"Completed {result['total_completed']}/{result['total_submitted']} reqs. "
          f"Mean throughput: {np.mean(result['throughput_per_min']):.2f} req/s")
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(2000)
    print(f"Loaded {len(prompts)} prompts")

    # Target rate: scale traces to ~10 req/s (moderate load for 30 min)
    TARGET_RATE = 10.0

    # Run all 4 experiments in parallel on different GPUs
    experiments = [
        ("LowConfidence", 0, 30100, "Kimi trace", KIMI_TRACE),
        ("CW_SRPT_V3",    1, 30200, "Kimi trace", KIMI_TRACE),
        ("LowConfidence", 2, 30300, "Azure trace", AZURE_TRACE),
        ("CW_SRPT_V3",    3, 30400, "Azure trace", AZURE_TRACE),
    ]

    print(f"\n{'='*60}")
    print(f"Starting 4 parallel experiments ({DURATION_MIN} min each)")
    print(f"Target arrival rate: {TARGET_RATE} req/s")
    print(f"{'='*60}")

    results = {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for algo, gpu, port, trace_name, trace_path in experiments:
            key = f"{algo}_{trace_name}"
            futures[pool.submit(run_experiment, algo, gpu, port,
                                trace_name, trace_path, prompts, TARGET_RATE)] = key

        for f in as_completed(futures):
            key = futures[f]
            try:
                result = f.result()
                if result:
                    results[key] = result
            except Exception as e:
                print(f"  ERROR in {key}: {e}")

    # Save raw results
    raw_path = OUT_DIR / "fig9_raw_data.json"
    # Convert numpy arrays to lists for JSON
    save_data = {}
    for k, v in results.items():
        save_data[k] = {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
                        for kk, vv in v.items()}
    with open(raw_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✅ Raw data: {raw_path}")

    # Generate plot data format
    time_min_arr = list(range(DURATION_MIN))
    plot_data = {"Kimi trace": {}, "Azure trace": {}}

    for key, res in results.items():
        algo = res["algo"]
        trace = res["trace"]
        tp = res["throughput_per_min"]
        # Pad to 30 if needed
        while len(tp) < DURATION_MIN:
            tp.append(0)
        method_name = "Ours" if "V3" in algo else "SGLang"
        plot_data[trace][method_name] = tp[:DURATION_MIN]

    # Save plot-ready data
    plot_path = OUT_DIR / "fig9_plot_data.json"
    with open(plot_path, "w") as f:
        json.dump({"time_min": time_min_arr, "data": plot_data}, f, indent=2)
    print(f"✅ Plot data: {plot_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for trace in ["Kimi trace", "Azure trace"]:
        print(f"\n{trace}:")
        for method in ["SGLang", "Ours"]:
            if method in plot_data[trace]:
                tp = plot_data[trace][method]
                print(f"  {method}: mean={np.mean(tp):.2f} req/s, "
                      f"peak={np.max(tp):.2f} req/s")
        if "SGLang" in plot_data[trace] and "Ours" in plot_data[trace]:
            sg = np.mean(plot_data[trace]["SGLang"])
            ou = np.mean(plot_data[trace]["Ours"])
            if sg > 0:
                print(f"  Gain: +{(ou/sg - 1)*100:.0f}%")


if __name__ == "__main__":
    main()
