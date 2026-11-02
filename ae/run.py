#!/usr/bin/env python3
"""Sequential paired, open-loop HTTP measurements; every result is freshly measured."""
import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import tempfile

import aiohttp
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_identity():
    if (ROOT / '.git').exists():
        return {
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff'], cwd=ROOT)).hexdigest(),
        }
    return {'archive_source': json.loads((ROOT / 'ae/SOURCE.json').read_text())}


def prompt(text):
    return ('<role>SYSTEM</role>detailed thinking off<|role_end|>'
            '<role>HUMAN</role>' + text + '<|role_end|><role>ASSISTANT</role>')


@contextlib.contextmanager
def server(a, method, out):
    # Serialize startup as well as serving; binding only during preflight leaves
    # a race while workers load the checkpoint and before HTTP begins listening.
    lock_path = Path(tempfile.gettempdir()) / f'pas-ae-{os.getuid()}-{a.port}.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another AE run owns port {a.port}; choose --port')
        with _server(a, method, out):
            yield


@contextlib.contextmanager
def _server(a, method, out):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', a.port))
    config = ROOT / 'ae/configs' / ('sglang.yaml' if method == 'SGLang' else 'pas-paper.yaml')
    cmd = [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(a.model.resolve()),
           '--dllm-algorithm', 'LowConfidence' if method == 'SGLang' else 'CW_SRPT_V3',
           '--dllm-algorithm-config', str(config), '--max-running-requests', str(a.batch),
           '--disable-radix-cache', '--trust-remote-code', '--port', str(a.port),
           '--tp-size', str(a.tp), '--attention-backend', 'triton', '--disable-cuda-graph',
           '--mem-fraction-static', '0.8', '--random-seed', '42']
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'python')
    env['CUDA_VISIBLE_DEVICES'] = a.gpus
    (out / f'{method}-command.json').write_text(json.dumps(cmd, indent=2))
    with (out / f'{method}-server.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + a.start_timeout
            session = requests.Session(); session.trust_env = False
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'{method} exited; see {out}/{method}-server.log')
                try:
                    if session.get(f'http://127.0.0.1:{a.port}/health', timeout=2).ok:
                        info = session.get(f'http://127.0.0.1:{a.port}/server_info', timeout=5).json()
                        args = info.get('server_args', info)
                        expected = 'LowConfidence' if method == 'SGLang' else 'CW_SRPT_V3'
                        if args.get('dllm_algorithm') != expected:
                            raise RuntimeError('Ready endpoint belongs to a different algorithm')
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(f'{method} startup timed out')
            yield
        finally:
            # Kill only the process group this invocation created, including workers.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            except ProcessLookupError:
                pass


async def replay(a, rows, arrivals, max_tokens):
    records = []
    t0 = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=a.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as client:
        async def one(i, scheduled):
            await asyncio.sleep(max(0, t0 + scheduled - time.perf_counter()))
            sent = time.perf_counter() - t0
            record = {'id': i, 'dataset_id': rows[i % len(rows)]['id'], 'scheduled_s': scheduled, 'sent_s': sent}
            try:
                row = rows[i % len(rows)]
                requested_tokens = row.get('output_tokens', max_tokens)
                content = {'input_ids': row['input_ids']} if 'input_ids' in row else {'text': prompt(row['question'])}
                payload = {**content,
                           'sampling_params': {'temperature': 0, 'max_new_tokens': requested_tokens,
                                               'ignore_eos': True}, 'stream': False}
                async with client.post(f'http://127.0.0.1:{a.port}/generate', json=payload) as r:
                    body = await r.json()
                    if r.status != 200:
                        raise RuntimeError(f'HTTP {r.status}: {str(body)[:400]}')
                ids = body.get('output_ids')
                if ids is None:
                    raise RuntimeError('Native response omitted output_ids; cannot verify mask-free output')
                if 156895 in ids:
                    raise RuntimeError('Unresolved mask reached the generated output')
                if not ids or len(ids) != requested_tokens:
                    raise RuntimeError(f'Expected {requested_tokens} tokens, got {len(ids)}')
                if 'input_ids' in row and body['meta_info']['prompt_tokens'] != len(row['input_ids']):
                    raise RuntimeError('Served prompt length differs from trace request')
                record.update(ok=True, response=body)
            except Exception as e:
                record.update(ok=False, error=str(e))
            record['finished_s'] = time.perf_counter() - t0
            record['latency_s'] = record['finished_s'] - scheduled
            records.append(record)
        await asyncio.gather(*(one(i, t) for i, t in enumerate(arrivals)))
    return sorted(records, key=lambda r: r['id'])


def schedule(rate, duration, seed):
    rng = random.Random(seed); t = 0.; arrivals = []
    while True:
        t += rng.expovariate(rate)
        if t >= duration:
            return arrivals or [0.]
        arrivals.append(t)


def summarize(records, duration):
    good = [r for r in records if r['ok']]
    lat = [r['latency_s'] for r in good]
    return {'submitted': len(records), 'completed': len(good), 'failed': len(records) - len(good),
            'p50_s': float(np.percentile(lat, 50)) if lat else None,
            'p90_s': float(np.percentile(lat, 90)) if lat else None,
            'measurement_completions_per_s': sum(r['finished_s'] <= duration for r in good) / duration,
            'drained_requests_per_s': len(good) / max(duration, max((r['finished_s'] for r in records), default=0)),
            'max_client_dispatch_lag_s': max((r['sent_s'] - r['scheduled_s'] for r in records), default=0)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'ae/data/gsm8k.jsonl')
    p.add_argument('--gpus', default='0,1'); p.add_argument('--tp', type=int, default=2)
    p.add_argument('--batch', type=int, default=8); p.add_argument('--port', type=int, default=30174)
    p.add_argument('--profile', choices=['smoke', 'quick', 'full'], default='smoke')
    p.add_argument('--rates', type=float, nargs='+', default=[0.5, 1., 2., 4.])
    p.add_argument('--methods', nargs='+', choices=['SGLang', 'PAS'], default=['SGLang', 'PAS'])
    p.add_argument('--duration', type=float); p.add_argument('--warmup', type=float)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-timeout', type=int, default=900)
    p.add_argument('--request-timeout', type=int, help='Default: 180 s quick/smoke, 1800 s full')
    a = p.parse_args()
    if a.request_timeout is None:
        a.request_timeout = 1800 if a.profile == 'full' else 180
    if len(a.gpus.split(',')) != a.tp: p.error('--gpus must contain exactly --tp devices')
    if a.tokens < 1 or any(r <= 0 for r in a.rates): p.error('tokens/rates must be positive')
    duration, warmup = {'smoke': (4, 1), 'quick': (30, 10), 'full': (600, 120)}[a.profile]
    a.duration = a.duration if a.duration is not None else duration
    a.warmup = a.warmup if a.warmup is not None else warmup
    if a.duration <= 0 or a.warmup < 0: p.error('Invalid measurement/warmup duration')
    a.seeds = a.seeds or ([42, 43, 44] if a.profile == 'full' else [42])
    if a.profile == 'smoke': a.rates = [2.]
    a.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in a.data.read_text().splitlines()]
    if not rows: p.error('Empty dataset')
    manifest = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                'data_sha256': digest(a.data),
                **source_identity(),
                'config_sha256': {q.name: digest(q) for q in (ROOT / 'ae/configs').glob('*.yaml')},
                'source_sha256': {str(q.relative_to(ROOT)): digest(q) for q in [
                    ROOT / 'ae/run.py', ROOT / 'ae/requirements.lock',
                    *sorted((ROOT / 'python/sglang/srt/dllm').rglob('*.py')),
                    ROOT / 'python/sglang/srt/managers/scheduler.py',
                    ROOT / 'python/sglang/srt/layers/moe/topk.py']},
                'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], text=True)}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = []; failures = 0
    for method in a.methods:
        print(f'Starting {method}', flush=True)
        with server(a, method, a.output):
            for seed in a.seeds:
                for rate in a.rates:
                    if a.warmup:
                        warm = asyncio.run(replay(a, rows, schedule(rate, a.warmup, seed + 10000), a.tokens))
                        (a.output / f'{method}-{seed}-{rate}-warmup.json').write_text(json.dumps(warm))
                        if any(not r['ok'] for r in warm):
                            raise RuntimeError('Warmup failed correctness checks; inspect warmup JSON')
                    arrivals = schedule(rate, a.duration, seed)
                    records = asyncio.run(replay(a, rows, arrivals, a.tokens))
                    (a.output / f'{method}-{seed}-{rate}-requests.json').write_text(json.dumps(records, indent=2))
                    result = {'method': method, 'seed': seed, 'rate': rate, **summarize(records, a.duration)}
                    failures += result['failed']; results.append(result)
                    (a.output / 'summary.json').write_text(json.dumps(results, indent=2))
                    print(json.dumps(result), flush=True)
    subprocess.run([sys.executable, str(ROOT / 'ae/plot.py'), str(a.output)], check=True)
    if failures: raise SystemExit(f'{failures} failed requests; these results do not pass validation')


if __name__ == '__main__':
    main()
