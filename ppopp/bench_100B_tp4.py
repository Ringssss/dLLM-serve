#!/usr/bin/env python3
"""
Fig. 8: LLaDA2.0-flash (100B MoE) load sweep, TP=4.
Tests SGLang (LowConfidence) and Ours (CW_SRPT_V3) at multiple rates.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import requests as http_req

PYTHON = "/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"
MODEL = "/mnt/models/LLaDA2.0-flash"
PORT = 30100
TP = 4
OUT = Path("/home/zhujianian/sglang/ppopp")
N_REQS = 48  # fewer reqs for 100B (slower)
GEN_LENGTH = 128
RATES = [1, 2, 3, 4, 5, 6, 8]  # lower rates for bigger model

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis step by step.",
    "Write a detailed comparison between TCP and UDP protocols.",
    "Explain how a transformer neural network works.",
    "Describe the history of the Roman Empire from founding to fall.",
    "Write a short essay about the impact of AI on society.",
    "Explain functional vs object-oriented programming.",
    "What are the key principles of quantum computing?",
    "Implement a binary search tree in Python.",
    "Explain the CAP theorem with real-world examples.",
    "Describe the architecture of a modern OS kernel.",
    "Write a haiku about the ocean.",
    "Tell me a joke about programming.",
    "Describe a sunset over the mountains in vivid detail.",
    "The Industrial Revolution transformed manufacturing.",
    "Machine learning uses statistical methods to learn.",
    "List the planets in order from the sun.",
    "What is the Pythagorean theorem?",
    "Explain how vaccines protect against diseases.",
    "Describe the water cycle and its importance.",
    "What is the difference between TCP and UDP?",
    "Explain the concept of recursion in programming.",
    "Describe the structure of DNA.",
    "What causes earthquakes?",
    "Explain how solar panels convert sunlight to electricity.",
    "What is the theory of evolution?",
    "Describe the process of fermentation.",
    "Explain how the internet works at a high level.",
    "What is blockchain technology?",
    "Describe the differences between RAM and ROM.",
    "Explain the greenhouse effect.",
    "What is the significance of the Turing test?",
    "Describe how a compiler works.",
    "Explain the concept of supply and demand.",
    "What is the Doppler effect?",
    "Describe the process of nuclear fission.",
    "Explain how GPS navigation works.",
    "What are the fundamental forces of nature?",
    "Describe the life cycle of a star.",
    "Explain the concept of entropy.",
    "What is the difference between AC and DC current?",
    "Describe how antibiotics work.",
    "Explain the concept of machine learning overfitting.",
    "What is the halting problem?",
    "Describe how a neural network learns.",
    "Explain the concept of time complexity in algorithms.",
]


def mk_prompt(text):
    return ('<role>SYSTEM</role>detailed thinking off<|role_end|>'
            '<role>HUMAN</role>' + text + '<|role_end|><role>ASSISTANT</role>')


def wait_server(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = http_req.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(5)
    return False


def kill_server():
    os.system(f"pkill -f 'sglang.launch_server.*{PORT}' 2>/dev/null")
    time.sleep(5)


def start_server(algo):
    kill_server()
    gpus = ",".join(str(i) for i in range(TP))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    for k in ["http_proxy", "https_proxy", "all_proxy", "ALL_PROXY"]:
        env.pop(k, None)
    env["no_proxy"] = "127.0.0.1,localhost"

    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL,
           "--dllm-algorithm", algo,
           "--max-running-requests", "8",
           "--tp-size", str(TP),
           "--disable-radix-cache", "--trust-remote-code",
           "--port", str(PORT)]

    log = OUT / f"server_100B_{algo}.log"
    proc = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    print(f"  Starting {algo} TP={TP} (PID {proc.pid})...")
    print(f"  Waiting (100B model takes ~3-5 min to load)...")
    if not wait_server(timeout=600):
        print(f"  FAILED! Check {log}")
        proc.kill()
        return None
    print(f"  {algo} ready.")
    return proc


def run_rate(prompts, rate):
    interval = 1.0 / rate
    latencies = []
    t_start = time.time()

    def send(idx):
        payload = {"model": "default", "prompt": mk_prompt(prompts[idx % len(prompts)]),
                   "max_tokens": GEN_LENGTH, "temperature": 0}
        t0 = time.time()
        try:
            r = http_req.post(f"http://127.0.0.1:{PORT}/v1/completions",
                              json=payload, timeout=300)
            if r.status_code == 200:
                return time.time() - t0
        except:
            pass
        return None

    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = []
        for i in range(N_REQS):
            futures.append(pool.submit(send, i))
            if i < N_REQS - 1:
                time.sleep(interval)
        for f in as_completed(futures):
            lat = f.result()
            if lat is not None:
                latencies.append(lat)

    wall = time.time() - t_start
    if not latencies:
        return {"p90": 99999, "tps": 0, "ok": 0}
    latencies.sort()
    n = len(latencies)
    return {
        "p50": round(latencies[int(n*0.5)] * 1000, 1),
        "p90": round(latencies[int(n*0.9)] * 1000, 1),
        "p99": round(latencies[int(n*0.99)] * 1000, 1),
        "mean": round(np.mean(latencies) * 1000, 1),
        "tps": round(n * GEN_LENGTH / wall, 1),
        "ok": n,
    }


def main():
    algos = ["LowConfidence", "CW_SRPT_V3"]
    results = {}

    for algo in algos:
        print(f"\n{'='*60}")
        print(f"LLaDA2.0-flash (100B), TP={TP}, Algorithm: {algo}")
        print(f"{'='*60}")

        proc = start_server(algo)
        if proc is None:
            continue

        # Warmup
        print("  Warmup...")
        run_rate(PROMPTS[:4], 1)

        algo_results = {}
        for rate in RATES:
            print(f"  rate={rate} ...", end=" ", flush=True)
            res = run_rate(PROMPTS, rate)
            algo_results[rate] = res
            print(f"P90={res['p90']:.0f}ms TPS={res['tps']:.0f} OK={res['ok']}/{N_REQS}")

        results[algo] = algo_results
        kill_server()

    # Save
    out_path = OUT / "fig8_100B_tp4_measured.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Rate':<6} {'SGLang P90':<14} {'Ours P90':<14} {'Reduction':<10}")
    print("-" * 44)
    for rate in RATES:
        if "LowConfidence" in results and "CW_SRPT_V3" in results:
            sg = results["LowConfidence"].get(rate, {}).get("p90", 0)
            ou = results["CW_SRPT_V3"].get(rate, {}).get("p90", 0)
            red = (1 - ou/sg) * 100 if sg > 0 else 0
            print(f"{rate:<6} {sg:<14.0f} {ou:<14.0f} −{red:.0f}%")


if __name__ == "__main__":
    main()
