#!/usr/bin/env python3
'''Plot ae/sweep.py output: P90 latency and throughput versus offered load.

Usage: python ae/plot_sweep.py RESULT_DIR
Reads RESULT_DIR/summary.json only; writes summary.csv and sweep.png/pdf.
'''
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
rows = json.loads((out / 'summary.json').read_text())
fields = ['method', 'dataset', 'rate', 'offered', 'completed', 'failed', 'p50_s', 'p90_s', 'p99_s', 'throughput_rps']
with (out / 'summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)

datasets = sorted({r['dataset'] for r in rows}, key=['gsm8k', 'mgsm', 'humaneval', 'mt_bench'].index)
fig, axes = plt.subplots(2, len(datasets), figsize=(3.6 * len(datasets), 6), squeeze=False, layout='constrained')
for col, ds in enumerate(datasets):
    for method in ['SGLang', 'PAS']:
        pts = sorted((r for r in rows if r['dataset'] == ds and r['method'] == method and r['p90_s'] is not None),
                     key=lambda r: r['rate'])
        if not pts:
            continue
        x = [r['rate'] for r in pts]
        axes[0, col].plot(x, [r['p90_s'] for r in pts], marker='o', label=method)
        axes[1, col].plot(x, [r['throughput_rps'] for r in pts], marker='o', label=method)
    axes[0, col].set(title=ds, xscale='log', yscale='log', ylabel='P90 latency (s)')
    axes[1, col].set(xscale='log', xlabel='Offered load (req/s)', ylabel='Throughput (req/s)')
    for ax in axes[:, col]:
        ax.grid(alpha=.3)
        ax.legend()
fig.savefig(out / 'sweep.png', dpi=160)
fig.savefig(out / 'sweep.pdf')
print(f'Wrote {out}/summary.csv and sweep.png/pdf')
