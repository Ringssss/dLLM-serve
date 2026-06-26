#!/usr/bin/env python3
"""
Collect data for Fig. 10 (Ablation), Fig. 12 (Sensitivity), Fig. 13 (Quality).

Runs on GPUs 4-7 in parallel while Fig. 9 uses GPUs 0-3.

Fig. 10 ablation configs (rate=10, 64 reqs each):
  A: LowConfidence (FIFO + fixed threshold)               = baseline
  B: CW_SRPT_V3(target_iters=32, early_break=1.0)         = + scheduling only
  C: CW_SRPT_V3(early_break=1.0)                          = + scheduling + stride
  D: CW_SRPT_V3                                           = full PAS

Fig. 12 sensitivity:
  (a) target_iters sweep: 2, 4, 8, 16, 32
  (b) max-running-requests sweep: 4, 8, 16, 32
  (c) min_conf sweep: 0.1, 0.2, 0.3, 0.4, 0.5
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests as http_req

PYTHON = "/home/zhujianian/miniconda3/envs/sglang-bench/bin/python"
MODEL = "/mnt/models/LLaDA2.0-mini"
OUT_DIR = Path("/home/zhujianian/sglang/ppopp")

GSM8K = "/home/zhujianian/morspec/data/gsm8k.jsonl"
HUMANEVAL = "/home/zhujianian/morspec/data/humaneval.jsonl"

N_REQS = 64
GEN_LENGTH = 128
RATE = 10  # Fixed rate for ablation/sensitivity


def load_prompts(path, n=64):
    prompts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d.get("question", d.get("prompt", ""))
            if text:
                prompts.append(text[:512])
            if len(prompts) >= n:
                break
    while len(prompts) < n:
        prompts.extend(prompts[:n - len(prompts)])
    return prompts[:n]


def mk_prompt(text):
    return ('<role>SYSTEM</role>detailed thinking off<|role_end|>'
            '<role>HUMAN</role>' + text + '<|role_end|><role>ASSISTANT</role>')


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


def kill_port(port):
    os.system(f"pkill -f 'sglang.launch_server.*--port {port}' 2>/dev/null")
    time.sleep(3)


def start_server(algo, gpu, port, max_running=8, algo_config=None):
    kill_port(port)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ["http_proxy", "https_proxy", "all_proxy", "ALL_PROXY"]:
        env.pop(k, None)
    env["no_proxy"] = "127.0.0.1,localhost"

    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL,
           "--dllm-algorithm", algo,
           "--max-running-requests", str(max_running),
           "--disable-radix-cache", "--trust-remote-code",
           "--port", str(port)]

    if algo_config:
        # Write temp config file
        cfg_path = f"/tmp/dllm_config_{port}.yaml"
        import yaml
        with open(cfg_path, "w") as f:
            yaml.dump(algo_config, f)
        cmd += ["--dllm-algorithm-config", cfg_path]

    log = OUT_DIR / f"server_{algo}_gpu{gpu}_port{port}.log"
    proc = subprocess.Popen(cmd, env=env,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT)
    if not wait_server(port):
        proc.kill()
        return None
    return proc


def run_rate_test(port, prompts, rate, n_reqs=64):
    interval = 1.0 / rate
    latencies = []
    t_start = time.time()

    def send(idx):
        payload = {"model": "default", "prompt": mk_prompt(prompts[idx]),
                   "max_tokens": GEN_LENGTH, "temperature": 0}
        t0 = time.time()
        try:
            r = http_req.post(f"http://127.0.0.1:{port}/v1/completions",
                              json=payload, timeout=180)
            if r.status_code == 200:
                return time.time() - t0
        except:
            pass
        return None

    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = []
        for i in range(min(n_reqs, len(prompts))):
            futures.append(pool.submit(send, i))
            if i < n_reqs - 1:
                time.sleep(interval)
        for f in as_completed(futures):
            lat = f.result()
            if lat is not None:
                latencies.append(lat)

    wall = time.time() - t_start
    if not latencies:
        return {"p90": 99999, "tps": 0}
    latencies.sort()
    n = len(latencies)
    return {
        "p50": round(latencies[int(n * 0.5)] * 1000, 1),
        "p90": round(latencies[int(n * 0.9)] * 1000, 1),
        "p99": round(latencies[int(n * 0.99)] * 1000, 1),
        "mean": round(np.mean(latencies) * 1000, 1),
        "tps": round(n * GEN_LENGTH / wall, 1),
        "ok": n,
    }


# ═══════════════════════════════════════════════════════════════════
# Fig. 10: Component Ablation (GPU 4)
# ═══════════════════════════════════════════════════════════════════
def run_fig10(gpu=4, port=30500):
    print(f"\n{'='*60}")
    print(f"Fig. 10: Component Ablation (GPU {gpu})")
    print(f"{'='*60}")

    prompts = load_prompts(GSM8K, N_REQS)
    configs = [
        ("SGLang",              "LowConfidence", 8, None),
        ("+ Frontier\nScheduling", "CW_SRPT_V3", 8,
         {"target_iters": 32, "min_unmask": 1, "early_break_ratio": 1.0, "min_conf": 0.0}),
        ("+ Elastic\nStride",     "CW_SRPT_V3", 8,
         {"target_iters": 8, "min_unmask": 2, "early_break_ratio": 1.0, "min_conf": 0.3}),
        ("PAS",                    "CW_SRPT_V3", 8, None),
    ]

    results = {}
    for label, algo, max_run, cfg in configs:
        print(f"\n  [{label.replace(chr(10),' ')}] ...", end=" ", flush=True)
        proc = start_server(algo, gpu, port, max_running=max_run, algo_config=cfg)
        if proc is None:
            print("FAILED")
            continue
        # Warmup
        run_rate_test(port, prompts[:4], 2, 4)
        # Test at multiple rates
        label_results = {}
        for rate in [5, 10, 15, 20]:
            res = run_rate_test(port, prompts, rate, N_REQS)
            label_results[rate] = res
            print(f"r={rate}:P90={res['p90']:.0f}", end=" ", flush=True)
        results[label.replace('\n', ' ')] = label_results
        kill_port(port)
        print()

    return results


# ═══════════════════════════════════════════════════════════════════
# Fig. 12: Sensitivity (GPU 5, 6, 7)
# ═══════════════════════════════════════════════════════════════════
def run_fig12a_target_iters(gpu=5, port=30600):
    """Sweep target_iters (scheduling quantum)."""
    print(f"\n{'='*60}")
    print(f"Fig. 12(a): target_iters sweep (GPU {gpu})")
    print(f"{'='*60}")

    prompts = load_prompts(GSM8K, N_REQS)
    values = [2, 4, 8, 16, 32]
    results = {}

    for ti in values:
        cfg = {"target_iters": ti, "min_unmask": 2, "min_conf": 0.3, "early_break_ratio": 0.5}
        print(f"  target_iters={ti} ...", end=" ", flush=True)
        proc = start_server("CW_SRPT_V3", gpu, port, algo_config=cfg)
        if proc is None:
            print("FAILED")
            continue
        run_rate_test(port, prompts[:4], 2, 4)
        res = run_rate_test(port, prompts, RATE, N_REQS)
        results[ti] = res
        print(f"P90={res['p90']:.0f}ms TPS={res['tps']:.0f}")
        kill_port(port)

    return results


def run_fig12b_batch_cap(gpu=6, port=30700):
    """Sweep max_running_requests (batch capacity)."""
    print(f"\n{'='*60}")
    print(f"Fig. 12(b): batch capacity sweep (GPU {gpu})")
    print(f"{'='*60}")

    prompts = load_prompts(GSM8K, N_REQS)
    values = [4, 8, 16, 32]
    results = {}

    for mr in values:
        print(f"  max_running={mr} ...", end=" ", flush=True)
        proc = start_server("CW_SRPT_V3", gpu, port, max_running=mr)
        if proc is None:
            print("FAILED")
            continue
        run_rate_test(port, prompts[:4], 2, 4)
        res = run_rate_test(port, prompts, RATE, N_REQS)
        results[mr] = res
        print(f"P90={res['p90']:.0f}ms TPS={res['tps']:.0f}")
        kill_port(port)

    return results


def run_fig12c_min_conf(gpu=7, port=30800):
    """Sweep min_conf (confidence guard)."""
    print(f"\n{'='*60}")
    print(f"Fig. 12(c): min_conf sweep (GPU {gpu})")
    print(f"{'='*60}")

    prompts = load_prompts(GSM8K, N_REQS)
    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}

    for mc in values:
        cfg = {"target_iters": 8, "min_unmask": 2, "min_conf": mc, "early_break_ratio": 0.5}
        print(f"  min_conf={mc} ...", end=" ", flush=True)
        proc = start_server("CW_SRPT_V3", gpu, port, algo_config=cfg)
        if proc is None:
            print("FAILED")
            continue
        run_rate_test(port, prompts[:4], 2, 4)
        res = run_rate_test(port, prompts, RATE, N_REQS)
        results[mc] = res
        print(f"P90={res['p90']:.0f}ms TPS={res['tps']:.0f}")
        kill_port(port)

    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run all experiments in parallel on different GPUs
    with ThreadPoolExecutor(max_workers=4) as pool:
        f10 = pool.submit(run_fig10, gpu=4, port=30500)
        f12a = pool.submit(run_fig12a_target_iters, gpu=5, port=30600)
        f12b = pool.submit(run_fig12b_batch_cap, gpu=6, port=30700)
        f12c = pool.submit(run_fig12c_min_conf, gpu=7, port=30800)

        try:
            all_results["fig10_ablation"] = f10.result()
        except Exception as e:
            print(f"Fig 10 error: {e}")

        try:
            all_results["fig12a_target_iters"] = f12a.result()
        except Exception as e:
            print(f"Fig 12a error: {e}")

        try:
            all_results["fig12b_batch_cap"] = f12b.result()
        except Exception as e:
            print(f"Fig 12b error: {e}")

        try:
            all_results["fig12c_min_conf"] = f12c.result()
        except Exception as e:
            print(f"Fig 12c error: {e}")

    # Save
    out_path = OUT_DIR / "fig10_12_raw_data.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ All data saved: {out_path}")

    # Print summaries
    if "fig10_ablation" in all_results:
        print("\n=== Fig. 10 Ablation (rate=10) ===")
        for label, rates in all_results["fig10_ablation"].items():
            if 10 in rates:
                print(f"  {label:<25s} P90={rates[10]['p90']:.0f}ms")

    for fig, title in [("fig12a_target_iters", "target_iters"),
                       ("fig12b_batch_cap", "batch_cap"),
                       ("fig12c_min_conf", "min_conf")]:
        if fig in all_results:
            print(f"\n=== Fig. 12: {title} ===")
            for k, v in all_results[fig].items():
                print(f"  {k}: P90={v['p90']:.0f}ms TPS={v['tps']:.0f}")


if __name__ == "__main__":
    main()
