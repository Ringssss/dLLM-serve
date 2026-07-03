#!/usr/bin/env python3
"""
Runtime trace: real measurement of queue depth for SGLang vs PAS.
~210s per algo. Burst pattern: 60s steady → 60s burst → 90s drain.
Uses /get_server_info to poll queue depth every 1s.
"""
import json, os, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import requests as http_req

PYTHON = '/home/zhujianian/miniconda3/envs/sglang-bench/bin/python'
MODEL = '/mnt/models/LLaDA2.0-mini'
PORT = 30100
OUT = Path('/home/zhujianian/sglang/ppopp')
DURATION = 210

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute Fibonacci.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis step by step.",
    "Write a comparison between TCP and UDP protocols.",
    "Explain how a transformer neural network works.",
    "Describe the history of the Roman Empire.",
    "Write a short essay about the impact of AI on society.",
    "Explain functional vs object-oriented programming.",
    "What are the key principles of quantum computing?",
    "Implement a binary search tree in Python.",
    "Explain the CAP theorem with real-world examples.",
] * 6


def mk(t):
    return '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>' + t + '<|role_end|><role>ASSISTANT</role>'


def start_server(algo):
    os.system(f'pkill -9 -f "sglang.launch_server" 2>/dev/null')
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
    print(f'  Starting {algo} (PID {proc.pid})...', flush=True)
    for i in range(60):
        try:
            if http_req.get(f'http://127.0.0.1:{PORT}/health', timeout=5).status_code == 200:
                print(f'  Ready after {i*3}s', flush=True)
                return proc
        except:
            pass
        time.sleep(3)
    print('  FAILED to start', flush=True)
    proc.kill()
    return None


def discover_server_info():
    """Try all possible endpoints to find queue info."""
    endpoints = ['/get_server_info', '/v1/models', '/metrics']
    for ep in endpoints:
        try:
            r = http_req.get(f'http://127.0.0.1:{PORT}{ep}', timeout=3)
            if r.status_code == 200:
                print(f'  Endpoint {ep}: {str(r.text[:300])}', flush=True)
        except:
            pass


def get_queue_depth():
    """Get number of waiting + running requests from server."""
    try:
        r = http_req.get(f'http://127.0.0.1:{PORT}/get_server_info', timeout=2)
        if r.status_code == 200:
            info = r.json()
            # Try all possible field names
            waiting = 0
            running = 0
            for k, v in info.items():
                kl = k.lower()
                if 'wait' in kl or 'queue' in kl or 'pending' in kl:
                    if isinstance(v, (int, float)):
                        waiting = max(waiting, int(v))
                if 'run' in kl and 'req' in kl:
                    if isinstance(v, (int, float)):
                        running = max(running, int(v))
            return waiting, running, info
    except:
        pass
    return None, None, None


def collect_trace(name):
    """Send burst traffic and collect queue depth."""
    timestamps = []
    queue_depths = []
    running_counts = []
    completions = []
    stop_flag = threading.Event()
    completion_count = [0]

    # Background sampler
    def sampler():
        t0 = time.time()
        first_info_logged = False
        while not stop_flag.is_set():
            waiting, running, info = get_queue_depth()
            elapsed = time.time() - t0
            if waiting is not None:
                timestamps.append(elapsed)
                queue_depths.append(waiting)
                running_counts.append(running)
                completions.append(completion_count[0])
                if not first_info_logged and info:
                    print(f'  Server info keys: {list(info.keys())}', flush=True)
                    first_info_logged = True
            time.sleep(1.0)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    # Send requests with burst pattern
    pool = ThreadPoolExecutor(max_workers=64)
    t_start = time.time()
    req_idx = 0

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
            rate = 15  # burst
        else:
            rate = 5

        pool.submit(send_one, req_idx)
        req_idx += 1
        time.sleep(1.0 / rate)

    # Wait for drain
    print(f'  Sent {req_idx} reqs, waiting 30s for drain...', flush=True)
    time.sleep(30)
    stop_flag.set()
    sampler_thread.join(timeout=5)
    pool.shutdown(wait=False)

    return {
        'timestamps': timestamps,
        'queue_depths': queue_depths,
        'running_counts': running_counts,
        'completions': completions,
        'total_sent': req_idx,
    }


def main():
    results = {}

    for algo, name in [('LowConfidence', 'SGLang'), ('CW_SRPT_V3', 'PAS')]:
        print(f'\n{"="*50}', flush=True)
        print(f'{name} ({algo})', flush=True)
        print(f'{"="*50}', flush=True)

        proc = start_server(algo)
        if not proc:
            continue

        # Discover API
        discover_server_info()

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

        qd = trace['queue_depths']
        print(f'  Done. {len(trace["timestamps"])} samples', flush=True)
        print(f'  Queue: mean={np.mean(qd):.1f}, max={max(qd) if qd else 0}, completions={trace["completions"][-1] if trace["completions"] else 0}', flush=True)

        os.system(f'pkill -9 -f "sglang.launch_server" 2>/dev/null')
        time.sleep(5)

    # Save
    with open(OUT / 'fig_runtime_trace_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✅ Saved: {OUT}/fig_runtime_trace_data.json', flush=True)


if __name__ == '__main__':
    main()
