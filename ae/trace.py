#!/usr/bin/env python3
"""Paired replay of bundled, provenance-recorded Azure arrival/length windows."""
import argparse
import asyncio
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
from transformers import AutoTokenizer
from run import ROOT, digest, source_identity, server, replay, summarize


def make_input(tokenizer, text, n):
    # Exact token count using real GSM8K text, repeated/truncated as needed.
    # This is a length-matched surrogate, not original production content.
    prefix = tokenizer.encode('<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>', add_special_tokens=False)
    suffix = tokenizer.encode('<|role_end|><role>ASSISTANT</role>', add_special_tokens=False)
    body = tokenizer.encode(text, add_special_tokens=False)
    if n <= len(prefix) + len(suffix):
        return (body * math.ceil(n / len(body)))[:n]
    length = n - len(prefix) - len(suffix)
    return prefix + (body * math.ceil(length / len(body)))[:length] + suffix


def workload(a, name, tokenizer, prompts):
    trace_path = ROOT / 'ae/traces' / f'{name}.csv'
    reference = json.loads((ROOT / 'ae/traces/manifest.json').read_text())
    if digest(trace_path) != reference['windows'][name]['sha256']:
        raise RuntimeError('Trace checksum mismatch')
    source = list(csv.DictReader(trace_path.open()))
    rows = []; discarded = 0; clipped_in = 0; clipped_out = 0
    for i, row in enumerate(source):
        ni, no = int(row['input_tokens']), int(row['output_tokens'])
        if a.mode == 'original':
            ni_run, no_run = ni, no
            if ni + no > a.context_limit:
                discarded += 1; continue
        else:
            ni_scaled = max(1, math.ceil(ni * a.input_scale))
            no_scaled = max(1, math.ceil(no * a.output_scale))
            ni_run, no_run = min(ni_scaled, a.input_cap), min(no_scaled, a.output_cap)
            clipped_in += ni_run != ni_scaled; clipped_out += no_run != no_scaled
        ids = make_input(tokenizer, prompts[i % len(prompts)]['question'], ni_run)
        rows.append({'id': i, 'arrival_s': float(row['arrival_s']), 'original_input_tokens': ni,
                     'original_output_tokens': no, 'input_ids': ids, 'output_tokens': no_run})
        if a.max_requests and len(rows) >= a.max_requests: break
    if len(rows) < 2: raise RuntimeError('Window has fewer than two usable requests')
    if a.rate is not None:
        span = rows[-1]['arrival_s'] - rows[0]['arrival_s']
        if span <= 0: raise RuntimeError('Cannot scale a zero-span window')
        scale = (len(rows) - 1) / a.rate / span
        arrivals = [(r['arrival_s'] - rows[0]['arrival_s']) * scale for r in rows]
        duration = len(rows) / a.rate
    else:
        scale = 1.; arrivals = [r['arrival_s'] for r in rows]
        duration = reference['duration_s']
    meta = {'trace': name, 'mode': a.mode, 'trace_sha256': digest(trace_path),
            'source_window': reference['windows'][name], 'requests': len(rows),
            'excluded_context_limit': discarded, 'input_cap_hits': clipped_in,
            'output_cap_hits': clipped_out, 'time_scale': scale, 'duration_s': duration,
            'prompt_content': 'GSM8K token-length surrogate; repeat/truncate; no original multimodal content or cache',
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}}
    for r, t in zip(rows, arrivals): r['scheduled_s'] = t
    return rows, arrivals, duration, meta


def plot(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), layout='constrained', squeeze=False)
    for col, name in enumerate(['azure']):
        for method in ['SGLang', 'PAS']:
            path = out / f'{name}-{method}-requests.json'
            if not path.exists(): continue
            records = json.loads(path.read_text())
            duration = next(r['duration_s'] for r in summary if r['trace'] == name)
            bin_s = max(1., duration / 12)
            edges = np.linspace(0, duration, 13)
            good = [r['finished_s'] for r in records if r['ok'] and r['finished_s'] < duration]
            hist, _ = np.histogram(good, bins=edges)
            axes[0, col].step(edges[:-1], hist / np.diff(edges), where='post', label=method)
            grid = np.linspace(0, max(duration, max(r['finished_s'] for r in records)), 120)
            outstanding = [sum(r['sent_s'] <= t < r['finished_s'] for r in records) for t in grid]
            axes[1, col].plot(grid, outstanding, label=method)
            axes[1, col].axvline(duration, color='grey', linestyle=':', alpha=.5)
        axes[0, col].set(title=name.capitalize(), ylabel='Completions / s (window only)', xlabel='Replay time (s)')
        axes[1, col].set(ylabel='Client outstanding requests', xlabel='Replay time incl. drain (s)')
        for ax in axes[:, col]: ax.legend(); ax.grid(alpha=.25)
    fig.savefig(out / 'traces.png', dpi=180); fig.savefig(out / 'traces.pdf')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['scaled', 'original'], default='scaled')
    p.add_argument('--max-requests', type=int, default=120, help='0 means all usable records')
    p.add_argument('--rate', type=float, default=2., help='Target mean arrival rate; 0 preserves timestamps')
    p.add_argument('--input-scale', type=float, default=1/32); p.add_argument('--output-scale', type=float, default=.25)
    p.add_argument('--input-cap', type=int, default=2048); p.add_argument('--output-cap', type=int, default=256)
    p.add_argument('--context-limit', type=int, default=32768)
    p.add_argument('--gpus', default='0,1'); p.add_argument('--tp', type=int, default=2)
    p.add_argument('--port', type=int, default=30174); p.add_argument('--batch', type=int, default=8)
    p.add_argument('--start-timeout', type=int, default=900); p.add_argument('--request-timeout', type=int, default=600)
    a = p.parse_args()
    if a.rate == 0: a.rate = None
    if a.max_requests < 0 or (a.rate is not None and a.rate <= 0): p.error('Invalid count/rate')
    if len(a.gpus.split(',')) != a.tp: p.error('GPU count must match TP')
    if min(a.input_scale, a.output_scale, a.input_cap, a.output_cap) <= 0: p.error('Positive scales/caps required')
    a.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    prompts = [json.loads(l) for l in (ROOT / 'ae/data/gsm8k.jsonl').read_text().splitlines()]
    workloads = {n: workload(a, n, tokenizer, prompts) for n in ['azure']}
    for n, (rows, arrivals, duration, meta) in workloads.items():
        (a.output / f'{n}-workload.json').write_text(json.dumps({'metadata': meta, 'requests': rows}, indent=2))
    (a.output / 'source.json').write_text(json.dumps(source_identity(), indent=2))
    summaries = []
    for method in ['SGLang', 'PAS']:
        print('Starting', method, flush=True)
        with server(a, method, a.output):
            for name, (rows, arrivals, duration, meta) in workloads.items():
                warm = asyncio.run(replay(a, rows[:2], [0., 0.5], 128))
                (a.output / f'{name}-{method}-warmup.json').write_text(json.dumps(warm, indent=2))
                if any(not r['ok'] for r in warm): raise RuntimeError('Trace warmup failed')
                records = asyncio.run(replay(a, rows, arrivals, 128))
                (a.output / f'{name}-{method}-requests.json').write_text(json.dumps(records, indent=2))
                result = {'trace': name, 'method': method, 'duration_s': duration, **summarize(records, duration)}
                summaries.append(result)
                (a.output / 'summary.json').write_text(json.dumps(summaries, indent=2))
                print(json.dumps(result), flush=True)
                if result['failed']: raise RuntimeError('Trace failures invalidate measurement')
    plot(a.output, summaries)
    from plot_trace_lengths import plot_lengths
    plot_lengths(a.output)


if __name__ == '__main__': main()
