#!/usr/bin/env python3
"""CDFs from original token counts of the bundled fixed windows (no scaling)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_lengths(out):
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
    stats = {}
    for name in ['azure']:
        source = Path(__file__).parent / 'traces' / f'{name}.csv'
        with source.open() as f: rows = list(csv.DictReader(f))
        stats[name] = {'requests': len(rows)}
        for ax, field in zip(axes, ['input_tokens', 'output_tokens']):
            values = np.sort([int(r[field]) for r in rows])
            stats[name][field] = {'p50': float(np.percentile(values, 50)), 'p90': float(np.percentile(values, 90)), 'max': int(values[-1])}
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=name.capitalize())
            ax.set(xlabel=field.replace('_', ' '), ylabel='Empirical CDF', xscale='log')
            ax.grid(alpha=.25); ax.legend()
    (out / 'trace-length-stats.json').write_text(json.dumps(stats, indent=2))
    fig.savefig(out / 'trace-lengths.pdf'); fig.savefig(out / 'trace-lengths.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    plot_lengths(p.parse_args().output)
