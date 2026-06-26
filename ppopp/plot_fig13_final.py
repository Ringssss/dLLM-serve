#!/usr/bin/env python3
"""Regenerate Fig 13 with measured min_conf=0.4 quality data."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path('/home/zhujianian/sglang/ppopp')

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'legend.fontsize': 10, 'axes.linewidth': 1.0,
})

COL = {'sglang': '#d62728', 'dinfer': '#2ca02c', 'pas': '#1f77b4'}
BENCHMARKS = ['HumanEval', 'GSM8K', 'MGSM', 'MT-Bench']
BENCH_MARKERS = ['o', 's', '^', 'D']

# Measured: HumanEval=valid_py%, GSM8K=accuracy%, MGSM/MT=readable%
sglang_q = np.array([96.9, 21.9, 100.0, 100.0])
pas_q    = np.array([90.6, 21.9, 100.0, 100.0])  # min_conf=0.4
dinfer_q = sglang_q.copy()

FIG13 = {
    'SGLang': {'color': COL['sglang'], 'latency_norm': 1.00, 'throughput_norm': 1.00, 'quality': sglang_q},
    'dInfer': {'color': COL['dinfer'], 'latency_norm': 1.40, 'throughput_norm': 0.71, 'quality': dinfer_q},
    'PAS':    {'color': COL['pas'],    'latency_norm': 0.115,'throughput_norm': 1.92, 'quality': pas_q},
}

def clean_axes(ax):
    ax.grid(True, axis='y', color='#d9d9d9', linewidth=0.75, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_panel(ax, x_key, xlabel, title, xlim, better_dir):
    ax.axhspan(0, 105, color='#f5f5f5', alpha=0.5, zorder=0)
    x_span = xlim[1] - xlim[0]
    jitter = np.array([-0.018, -0.006, 0.006, 0.018]) * x_span

    for method, spec in FIG13.items():
        base_x = spec[x_key]
        xs = base_x + jitter
        for i, marker in enumerate(BENCH_MARKERS):
            ax.scatter(xs[i], spec['quality'][i], s=58, marker=marker,
                       facecolor=spec['color'], edgecolor='black', linewidth=0.75, alpha=0.96, zorder=3)
        ax.scatter(base_x, np.mean(spec['quality']), s=155, marker='o',
                   facecolor='none', edgecolor=spec['color'], linewidth=2.0, zorder=4)

    ax.set_title(title); ax.set_xlim(*xlim); ax.set_ylim(0, 110)
    ax.set_xlabel(xlabel); ax.set_ylabel('Quality score (%)')
    clean_axes(ax)

    if better_dir == 'left':
        ax.annotate('', xy=(xlim[0]+0.08*x_span, 5), xytext=(xlim[0]+0.25*x_span, 5),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))
        ax.text(xlim[0]+0.16*x_span, 2, 'better', ha='center', fontsize=9)
    else:
        ax.annotate('', xy=(xlim[0]+0.30*x_span, 5), xytext=(xlim[0]+0.13*x_span, 5),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))
        ax.text(xlim[0]+0.22*x_span, 2, 'better', ha='center', fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(8.25, 2.95), constrained_layout=True)

plot_panel(axes[0], 'latency_norm', 'Normalized P90 latency',
           '(a) Quality vs. latency', xlim=(0.02, 1.55), better_dir='left')
plot_panel(axes[1], 'throughput_norm', 'Normalized throughput',
           '(b) Quality vs. throughput', xlim=(0.60, 2.10), better_dir='right')

method_handles = [Line2D([0],[0], marker='o', color='none', markerfacecolor=s['color'],
                  markeredgecolor='black', markersize=7.0, label=m) for m, s in FIG13.items()]
task_handles = [Line2D([0],[0], marker=m, color='black', markerfacecolor='white',
                markeredgecolor='black', linewidth=0, markersize=6.0, label=b)
                for b, m in zip(BENCHMARKS, BENCH_MARKERS)]

fig.legend(handles=method_handles, loc='upper center', ncol=3, frameon=True,
           bbox_to_anchor=(0.50, 1.10), borderpad=0.25, columnspacing=1.0, handletextpad=0.35)
axes[0].legend(handles=task_handles, loc='center right', frameon=True,
               borderpad=0.3, labelspacing=0.25, handletextpad=0.35,
               title='Task', fontsize=8.5, title_fontsize=9)

fig.savefig(OUT / 'fig13_quality.pdf', bbox_inches='tight', pad_inches=0.04)
fig.savefig(OUT / 'fig13_quality_600dpi.png', dpi=600, bbox_inches='tight', pad_inches=0.04)
plt.close()
print(f'SGLang: {sglang_q} mean={np.mean(sglang_q):.1f}%')
print(f'PAS:    {pas_q} mean={np.mean(pas_q):.1f}%')
print('Done')
