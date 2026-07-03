#!/usr/bin/env python3
"""
Collect runtime trace data: queue depth over time for SGLang vs PAS.
Short burst test (~3 min each) to show queueing behavior under load spikes.
"""
import json, os, subprocess, sys, time, threading
from pathlib import Path
import numpy as np
import requests as http_req

PYTHON = '/home/zhujianian/miniconda3/envs/sglang-bench/bin/python'
MODEL = '/mnt/models/LLaDA2.0-mini'
PORT = 30100
OUT = Path('/home/zhujianian/sglang/ppopp')
DURATION = 180  # 3 minutes
SAMPLE_INTERVAL = 1.0  # sample every 1s

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute Fibonacci.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis step by step.",
    "Write a comparison between TCP and UDP protocols.",
    "Explain how a transformer neural network works.",
    "Describe the history of the Roman Empire.",
    "Write a short essay about the impact of AI on society.",
] * 8


def mk(t):
    return '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>' + t + '<|role_end|><role>ASSISTANT</role>'


def start_server(algo):
    os.system(f'pkill -f "sglang.launch_server.*{PORT}" 2>/dev/null')
    time.sleep(5)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '4'
    for k in ['http_proxy', 'https_proxy', 'all_proxy', 'ALL_PROXY']:
        env.pop(k, None)
    env['no_proxy'] = '127.0.0.1,localhost'
    cmd = [PYTHON, '-m', 'sglang.launch_server',
           '--model-path', MODEL, '--dllm-algorithm', algo,
           '--max-running-requests', '8',
           '--disable-radix-cache', '--trust-remote-code', '--port', str(PORT)]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(60):
        try:
            if http_req.get(f'http://127.0.0.1:{PORT}/health', timeout=5).status_code == 200:
                print(f'  Ready', flush=True)
                return proc
        except:
            pass
        time.sleep(3)
    print('  FAILED', flush=True)
    proc.kill()
    return None


def collect_trace(algo_name):
    """Send requests with varying rate (including burst) and sample queue depth."""
    queue_depths = []
    timestamps = []
    running_counts = []
    stop_flag = threading.Event()

    # Background sampler: poll /get_server_info for queue depth
    def sampler():
        t0 = time.time()
        while not stop_flag.is_set():
            try:
                r = http_req.get(f'http://127.0.0.1:{PORT}/get_server_info', timeout=2)
                if r.status_code == 200:
                    info = r.json()
                    # Try different keys
                    q = info.get('num_waiting_reqs', info.get('waiting_queue_len', 0))
                    run = info.get('num_running_reqs', info.get('running_req_count', 0))
                    timestamps.append(time.time() - t0)
                    queue_depths.append(q)
                    running_counts.append(run)
            except:
                pass
            time.sleep(SAMPLE_INTERVAL)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    # Send requests with burst pattern:
    # Phase 1 (0-60s): steady rate=5
    # Phase 2 (60-120s): burst rate=15
    # Phase 3 (120-180s): back to rate=5, queue drains
    t_start = time.time()
    req_idx = 0

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=64)

    def send_one(idx):
        try:
            http_req.post(f'http://127.0.0.1:{PORT}/v1/completions',
                          json={'model': 'default', 'prompt': mk(PROMPTS[idx % len(PROMPTS)]),
                                'max_tokens': 128, 'temperature': 0}, timeout=120)
        except:
            pass

    while time.time() - t_start < DURATION:
        elapsed = time.time() - t_start
        if elapsed < 60:
            rate = 5  # steady
        elif elapsed < 120:
            rate = 15  # burst
        else:
            rate = 5  # drain

        pool.submit(send_one, req_idx)
        req_idx += 1
        time.sleep(1.0 / rate)

    # Wait for queue to drain
    time.sleep(30)
    stop_flag.set()
    sampler_thread.join(timeout=5)
    pool.shutdown(wait=False)

    return {
        'timestamps': timestamps,
        'queue_depths': queue_depths,
        'running_counts': running_counts,
        'total_sent': req_idx,
    }


def main():
    results = {}
    for algo, name in [('LowConfidence', 'SGLang'), ('CW_SRPT_V3', 'PAS')]:
        print(f'\n=== {name} ({algo}) ===', flush=True)
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
        trace = collect_trace(name)
        results[name] = trace
        print(f'  Done. {len(trace["timestamps"])} samples, sent {trace["total_sent"]} reqs', flush=True)
        print(f'  Queue: mean={np.mean(trace["queue_depths"]):.1f}, max={max(trace["queue_depths"]):.0f}', flush=True)

        os.system(f'pkill -f "sglang.launch_server.*{PORT}"')
        time.sleep(5)

    with open(OUT / 'fig_runtime_trace_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✅ Saved: {OUT}/fig_runtime_trace_data.json', flush=True)


if __name__ == '__main__':
    main()
