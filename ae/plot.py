#!/usr/bin/env python3
"""Plot only the supplied measurement directory; no embedded reference numbers."""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path(sys.argv[1])
rows = json.loads((out / 'summary.json').read_text())
with (out / 'summary.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
for method in ['SGLang', 'PAS']:
    selected = [r for r in rows if r['method'] == method and r['failed'] == 0]
    rates = sorted({r['rate'] for r in selected})
    for ax, metric, label in zip(axes, ['p90_s', 'drained_requests_per_s'], ['P90 end-to-end latency (s)', 'Throughput incl. drain (req/s)']):
        means = [np.mean([r[metric] for r in selected if r['rate'] == rate]) for rate in rates]
        ax.plot(rates, means, marker='o', label=method)
        ax.set(xlabel='Offered load (req/s)', ylabel=label)
        ax.grid(alpha=.3); ax.legend()
fig.savefig(out / 'comparison.png', dpi=180)
fig.savefig(out / 'comparison.pdf')
