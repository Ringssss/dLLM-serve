#!/usr/bin/env python3
"""
Collect ALL runtime trace data for the 3-panel figure:
  Top: GPU SM utilization (via nvidia-smi)
  Mid: Prefill tokens + decode batch size (from request tracking)
  Bot: Queue depth (sent - completed)

Burst pattern: 60s steady(5) → 60s burst(15) → 90s drain(5)
Runs SGLang then PAS sequentially on GPU 4.
"""
import json, os, subprocess, sys, time, threading, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests as http_req

PYTHON = '/home/zhujianian/miniconda3/envs/sglang-bench/bin/python'
MODEL = '/mnt/models/LLaDA2.0-mini'
PORT = 30100
GPU_ID = 4
OUT = Path('/home/zhujianian/sglang/ppopp')
DURATION = 210
AVG_PROMPT_LEN = 45  # average prompt token length for our prompts

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
    "Implement a binary search tree in Python with insert and search.",
    "Explain the CAP theorem with real-world examples.",
] * 8


def mk(t):
    return '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>' + t + '<|role_end|><role>ASSISTANT</role>'


def start_server(algo):
    os.system('pkill -9 -f "sglang.launch_server" 2>/dev/null')
    time.sleep(5)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    for k in ['http_proxy', 'https_proxy', 'all_proxy', 'ALL_PROXY']:
        env.pop(k, None)
    env['no_proxy'] = '127.0.0.1,localhost'
    cmd = [PYTHON, '-m', 'sglang.launch_server',
           '--model-path', MODEL, '--dllm-algorithm', algo,
           '--max-running-requests', '8',
           '--disable-radix-cache', '--trust-remote-code', '--port', str(PORT)]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'  Starting {algo}...', flush=True)
    for i in range(60):
        try:
            if http_req.get(f'http://127.0.0.1:{PORT}/health', timeout=5).status_code == 200:
                print(f'  Ready after {i*3}s', flush=True)
                return proc
        except:
            pass
        time.sleep(3)
    print('  FAILED', flush=True)
    proc.kill()
    return None


def get_gpu_sm_active():
    """Poll nvidia-smi for GPU utilization on GPU_ID."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits',
             f'--id={GPU_ID}'],
            timeout=2).decode().strip()
        return int(out)
    except:
        return 0


def collect_trace(name):
    """Send burst traffic and collect all metrics every 1s."""
    records = []
    stop_flag = threading.Event()
    completion_count = [0]
    cum_sent = [0]
    # Track per-second new arrivals for prefill token estimation
    arrivals_this_sec = [0]
    arrivals_lock = threading.Lock()

    # Background sampler: GPU util + queue every 1s
    def sampler():
        t0 = time.time()
        while not stop_flag.is_set():
            elapsed = time.time() - t0
            gpu_util = get_gpu_sm_active()
            with arrivals_lock:
                new_arrivals = arrivals_this_sec[0]
                arrivals_this_sec[0] = 0

            # prefill tokens ≈ new_arrivals × avg_prompt_length
            prefill_tokens = new_arrivals * AVG_PROMPT_LEN
            # decode batch size ≈ currently running (sent - completed - queued)
            # approximate: min(8, sent-completed) since max_running=8
            in_system = max(0, cum_sent[0] - completion_count[0])
            decode_bs = min(8, in_system)
            queue = max(0, in_system - 8)

            records.append({
                't': round(elapsed, 1),
                'gpu_util': gpu_util,
                'prefill_tokens': prefill_tokens,
                'decode_bs': decode_bs,
                'queue': queue,
                'in_system': in_system,
                'completed': completion_count[0],
                'sent': cum_sent[0],
            })
            time.sleep(1.0)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    # Send requests with burst pattern
    pool = ThreadPoolExecutor(max_workers=64)
    t_start = time.time()

    def send_one(idx):
        try:
            r = http_req.post(f'http://127.0.0.1:{PORT}/v1/completions',
                              json={'model': 'default', 'prompt': mk(PROMPTS[idx % len(PROMPTS)]),
                                    'max_tokens': 128, 'temperature': 0}, timeout=180)
            if r.status_code == 200:
                completion_count[0] += 1
        except:
            pass

    while time.time() - t_start < DURATION:
        elapsed = time.time() - t_start
        if elapsed < 60:
            rate = 5
        elif elapsed < 120:
            rate = 15
        else:
            rate = 5

        pool.submit(send_one, cum_sent[0])
        cum_sent[0] += 1
        with arrivals_lock:
            arrivals_this_sec[0] += 1
        time.sleep(1.0 / rate)

    # Wait for drain
    print(f'  Sent {cum_sent[0]} reqs, waiting 30s for drain...', flush=True)
    time.sleep(30)
    stop_flag.set()
    sampler_thread.join(timeout=5)
    pool.shutdown(wait=False)

    return records


def main():
    all_results = {}

    for algo, name in [('LowConfidence', 'SGLang'), ('CW_SRPT_V3', 'PAS')]:
        print(f'\n{"="*50}', flush=True)
        print(f'{name} ({algo})', flush=True)
        print(f'{"="*50}', flush=True)

        proc = start_server(algo)
        if not proc:
            continue

        # Warmup
        try:
            http_req.post(f'http://127.0.0.1:{PORT}/v1/completions',
                          json={'model': 'default', 'prompt': mk(PROMPTS[0]),
                                'max_tokens': 32, 'temperature': 0}, timeout=60)
        except:
            pass

        print(f'  Collecting trace ({DURATION}s)...', flush=True)
        records = collect_trace(name)
        all_results[name] = records

        qd = [r['queue'] for r in records]
        gpu = [r['gpu_util'] for r in records]
        print(f'  Done. {len(records)} samples', flush=True)
        print(f'  Queue: mean={np.mean(qd):.1f}, max={max(qd)}', flush=True)
        print(f'  GPU util: mean={np.mean(gpu):.0f}%, max={max(gpu)}%', flush=True)
        print(f'  Completed: {records[-1]["completed"]}/{records[-1]["sent"]}', flush=True)

        os.system('pkill -9 -f "sglang.launch_server" 2>/dev/null')
        time.sleep(5)

    # Save
    with open(OUT / 'fig_runtime_trace_full.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n✅ Saved: {OUT}/fig_runtime_trace_full.json', flush=True)


if __name__ == '__main__':
    main()
