#!/usr/bin/env python3
'''Portable load sweep over the bundled public datasets: SGLang vs PAS.

One server per method; every (dataset, rate) cell uses seeded Poisson arrivals,
fixed output length and EOS disabled, so both methods see identical requests.
Latency is measured from the scheduled arrival to response completion.
'''
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import ROOT, prompt, schedule, server  # noqa: E402

MASK_ID = 156895
DATASETS = ['gsm8k', 'mgsm', 'humaneval', 'mt_bench']


async def one(client, port, row, tokens, t0, at):
    await asyncio.sleep(max(0, t0 + at - time.perf_counter()))
    body = {'text': prompt(row['question'][:4000]),
            'sampling_params': {'temperature': 0, 'max_new_tokens': tokens, 'ignore_eos': True}}
    try:
        async with client.post(f'http://127.0.0.1:{port}/generate', json=body) as r:
            d = await r.json()
            if r.status != 200:
                raise RuntimeError(f'HTTP {r.status}')
        ids = d.get('output_ids') or []
        if MASK_ID in ids or len(ids) != tokens:
            raise RuntimeError('unresolved mask or wrong output length')
        now = time.perf_counter() - t0
        return {'ok': True, 'lat': now - at, 'done': now}
    except Exception as e:
        return {'ok': False, 'err': str(e)[:200], 'done': time.perf_counter() - t0}


async def cell(a, rows, rate, duration):
    arrivals = schedule(rate, duration, a.seed)
    t0 = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=a.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as c:
        recs = await asyncio.gather(*(one(c, a.port, rows[i % len(rows)], a.tokens, t0, t)
                                      for i, t in enumerate(arrivals)))
    wall = max(r['done'] for r in recs)
    good = [r for r in recs if r['ok']]
    lat = [r['lat'] for r in good]

    def pct(q):
        return float(np.percentile(lat, q)) if lat else None

    return {'rate': rate, 'offered': len(arrivals), 'completed': len(good),
            'failed': len(recs) - len(good), 'wall_s': round(wall, 2),
            'p50_s': pct(50), 'p90_s': pct(90), 'p99_s': pct(99),
            'throughput_rps': round(len(good) / wall, 3),
            'errors': sorted({r['err'] for r in recs if not r['ok']})[:3]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--methods', nargs='+', choices=['SGLang', 'PAS'], default=['SGLang', 'PAS'])
    p.add_argument('--rates', type=float, nargs='+', default=[0.5, 1, 2, 4, 8, 16, 32])
    p.add_argument('--duration', type=float, default=30)
    p.add_argument('--warmup', type=float, default=10)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--batch', type=int, default=64, help='max running requests')
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--tp', type=int, default=2)
    p.add_argument('--port', type=int, default=30174)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-timeout', type=int, default=900)
    p.add_argument('--request-timeout', type=int, default=1800)
    a = p.parse_args()
    if len(a.gpus.split(',')) != a.tp:
        p.error('--gpus must list exactly --tp devices')
    if a.duration <= 0 or any(r <= 0 for r in a.rates):
        p.error('duration and rates must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    data = {d: [json.loads(x) for x in (ROOT / 'ae/datasets' / f'{d}.jsonl').read_text().splitlines()]
            for d in a.datasets}
    results, failures = [], 0
    for method in a.methods:
        print(f'Starting {method}', flush=True)
        with server(a, method, a.output):
            for ds in a.datasets:
                for rate in a.rates:
                    if a.warmup:
                        asyncio.run(cell(a, data[ds], rate, a.warmup))
                    r = {'method': method, 'dataset': ds, **asyncio.run(cell(a, data[ds], rate, a.duration))}
                    failures += r['failed']
                    results.append(r)
                    (a.output / 'summary.json').write_text(json.dumps(results, indent=2))
                    print(json.dumps(r), flush=True)
    if failures:
        raise SystemExit(f'{failures} failed requests; inspect summary.json and the server logs')


if __name__ == '__main__':
    main()
