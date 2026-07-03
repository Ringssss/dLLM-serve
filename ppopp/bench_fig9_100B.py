#!/usr/bin/env python3
"""
Fig. 9 for 100B: 30-min trace replay on LLaDA2.0-flash, TP=4.
Sequential: SGLang+Kimi → SGLang+Azure → Ours+Kimi → Ours+Azure
Total ~2 hours.
"""
import csv, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import numpy as np
import requests as http_req

PYTHON = "/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"
MODEL = "/mnt/models/LLaDA2.0-flash"
PORT = 30100
TP = 4
DURATION_MIN = 30
OUT = Path("/home/zhujianian/sglang/ppopp")

KIMI_TRACE = "/mnt/models/kimik25/kimi-k25-trace/kimi_k25_conv_1day.csv"
AZURE_TRACE = "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv"
GSM8K = "/home/zhujianian/morspec/data/gsm8k.jsonl"
HUMANEVAL = "/home/zhujianian/morspec/data/humaneval.jsonl"
GEN_LENGTH = 128


def load_prompts(n=2000):
    prompts = []
    for path in [GSM8K, HUMANEVAL]:
        if not Path(path).exists(): continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                t = d.get("question", d.get("prompt", ""))
                if t: prompts.append(t[:512])
    while len(prompts) < n:
        prompts.extend(prompts[:n - len(prompts)])
    return prompts[:n]


def load_trace(trace_path, duration_s, target_rate=3.0):
    """Load trace, scale to target_rate for 100B (lower than 16B)."""
    timestamps = []
    with open(trace_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["TIMESTAMP"].replace("+00:00", "+00:00"))
                timestamps.append(ts.timestamp())
            except: continue
            if len(timestamps) > 200000: break

    timestamps.sort()
    # Find densest window
    best_start, best_count, j = 0, 0, 0
    for i in range(len(timestamps)):
        while j < len(timestamps) and timestamps[j] - timestamps[i] <= duration_s:
            j += 1
        if j - i > best_count:
            best_count = j - i
            best_start = i

    window = timestamps[best_start:best_start + best_count]
    t0 = window[0]
    arrivals = [(t - t0, idx % 2000) for idx, t in enumerate(window)]

    # Scale to target rate
    if arrivals:
        actual_rate = len(arrivals) / (arrivals[-1][0] + 1)
        if actual_rate > 0:
            scale = actual_rate / target_rate
            arrivals = [(t * scale, idx) for t, idx in arrivals]
            arrivals = [(t, idx) for t, idx in arrivals if t <= duration_s]

    return arrivals


def mk_prompt(t):
    return '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>' + t + '<|role_end|><role>ASSISTANT</role>'


def wait_server(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = http_req.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
            if r.status_code == 200: return True
        except: pass
        time.sleep(5)
    return False


def kill_server():
    os.system(f"pkill -f 'sglang.launch_server.*{PORT}' 2>/dev/null")
    time.sleep(5)


def start_server(algo):
    kill_server()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(TP))
    for k in ["http_proxy", "https_proxy", "all_proxy", "ALL_PROXY"]:
        env.pop(k, None)
    env["no_proxy"] = "127.0.0.1,localhost"
    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL, "--dllm-algorithm", algo,
           "--max-running-requests", "8", "--tp-size", str(TP),
           "--disable-radix-cache", "--trust-remote-code", "--port", str(PORT)]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  Starting {algo} TP={TP}...")
    if not wait_server():
        proc.kill()
        return None
    print(f"  Ready.")
    return proc


def run_replay(prompts, arrivals):
    """Replay trace, record per-minute completions."""
    n_min = DURATION_MIN
    completions_per_min = [0] * n_min
    t_start = time.time()

    def send_one(prompt_text, scheduled_time):
        now = time.time() - t_start
        if scheduled_time > now:
            time.sleep(scheduled_time - now)
        t0 = time.time()
        try:
            r = http_req.post(f"http://127.0.0.1:{PORT}/v1/completions",
                              json={"model": "default", "prompt": mk_prompt(prompt_text),
                                    "max_tokens": GEN_LENGTH, "temperature": 0}, timeout=300)
            if r.status_code == 200:
                finish = time.time() - t_start
                bucket = min(int(finish / 60), n_min - 1)
                completions_per_min[bucket] += 1
        except: pass

    duration_s = DURATION_MIN * 60
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = []
        for arr_time, pidx in arrivals:
            if arr_time > duration_s: break
            futures.append(pool.submit(send_one, prompts[pidx], arr_time))
        for f in as_completed(futures, timeout=duration_s + 180):
            pass

    return [c / 60.0 for c in completions_per_min]


def main():
    prompts = load_prompts(2000)
    print(f"Loaded {len(prompts)} prompts")

    # Target rate for 100B: ~3 req/s (much lower than 16B's ~10)
    TARGET_RATE = 3.0

    experiments = [
        ("LowConfidence", "Kimi trace", KIMI_TRACE),
        ("LowConfidence", "Azure trace", AZURE_TRACE),
        ("CW_SRPT_V3", "Kimi trace", KIMI_TRACE),
        ("CW_SRPT_V3", "Azure trace", AZURE_TRACE),
    ]

    results = {}

    for algo, trace_name, trace_path in experiments:
        print(f"\n{'='*60}")
        print(f"100B | {algo} | {trace_name} | {DURATION_MIN} min")
        print(f"{'='*60}")

        arrivals = load_trace(trace_path, DURATION_MIN * 60, target_rate=TARGET_RATE)
        print(f"  {len(arrivals)} arrivals at ~{TARGET_RATE} req/s")

        proc = start_server(algo)
        if proc is None:
            print("  FAILED")
            continue

        # Warmup
        try:
            http_req.post(f"http://127.0.0.1:{PORT}/v1/completions",
                          json={"model": "default", "prompt": mk_prompt(prompts[0]),
                                "max_tokens": 32, "temperature": 0}, timeout=120)
        except: pass

        print(f"  Running {DURATION_MIN}-min replay...")
        tp_per_min = run_replay(prompts, arrivals)

        key = f"{algo}_{trace_name}"
        method = "Ours" if "V3" in algo else "SGLang"
        results.setdefault(trace_name, {})[method] = tp_per_min

        mean_tp = np.mean([v for v in tp_per_min if v > 0])
        print(f"  Done. Mean throughput: {mean_tp:.2f} req/s")
        print(f"  Per-min: {[round(v,1) for v in tp_per_min]}")

        kill_server()

    # Save
    out_path = OUT / "fig9_100B_plot_data.json"
    with open(out_path, "w") as f:
        json.dump({"time_min": list(range(DURATION_MIN)), "data": results}, f, indent=2)
    print(f"\n✅ Saved: {out_path}")

    # Summary
    for trace_name, trace_data in results.items():
        print(f"\n{trace_name}:")
        for method, tp in trace_data.items():
            print(f"  {method}: mean={np.mean(tp):.2f} req/s, peak={max(tp):.2f}")


if __name__ == "__main__":
    main()
