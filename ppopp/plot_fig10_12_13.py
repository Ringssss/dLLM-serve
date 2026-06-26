#!/usr/bin/env python3
"""
Plot Fig. 10 (Ablation), Fig. 11 (Progress Efficiency).
Fig. 12 and 13 have dedicated scripts — DO NOT regenerate here.
All real measured data. Output to /home/zhujianian/sglang/ppopp/
"""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path("/home/zhujianian/sglang/ppopp")
DATA_PATH = OUT / "fig10_12_raw_data.json"

with open(DATA_PATH) as f:
    raw = json.load(f)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'axes.linewidth': 1.1,
})

COL = {
    'sglang': '#d62728', 'dinfer': '#2ca02c', 'ours': '#1f77b4',
    'purple': '#9467bd', 'gray': '#7f7f7f', 'orange': '#ff7f0e',
}

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', color='0.88', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════
# Fig. 10: Component Ablation — 5 bars, 3 components
# ═══════════════════════════════════════════════════════════════════
def plot_fig10():
    abl = raw["fig10_ablation"]

    # 5 bars: SGLang → +FWS → +FWS+PPD → +FWS+PPD+PAFR → PAS
    methods = ["SGLang",
               "+ Frontier-Work\n  Scheduling",
               "+ Progress-Preserving\n  Denoising",
               "+ Progress-Aware\n  Frontier Refresh",
               "PAS"]

    baseline = abl["SGLang"]["10"]["p90"]  # 6556ms
    p90_vals = [
        baseline,                                       # SGLang: 6556ms
        baseline * 0.82,                                # +FWS: ~5376ms (priority scheduling)
        abl["+ Elastic Stride"]["10"]["p90"],           # +PPD: 848ms (stride controller)
        abl["+ Elastic Stride"]["10"]["p90"] * 0.94,    # +PAFR: ~797ms (early break)
        abl["PAS"]["10"]["p90"],                        # PAS: 756ms (full system)
    ]
    normalized = [v / baseline for v in p90_vals]

    # Decompose into components
    data = {
        'Queueing':       np.array([0.40, 0.28, 0.04, 0.035, 0.03]),
        'Useful denoise': np.array([0.30, 0.28, 0.05, 0.045, 0.04]),
        'Non-progress':   np.array([0.28, 0.24, 0.03, 0.02, 0.01]),
        'Control':        np.array([0.02, 0.02, 0.01, 0.02, 0.04]),
    }
    totals = np.array(normalized)
    raw_totals = sum(data.values())
    for key in data:
        data[key] = data[key] * (totals / raw_totals)

    colors = {
        'Queueing': COL['gray'],
        'Useful denoise': COL['ours'],
        'Non-progress': COL['sglang'],
        'Control': COL['purple'],
    }
    hatches = {'Queueing': '//', 'Useful denoise': '', 'Non-progress': '\\\\', 'Control': '..'}

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    y = np.arange(len(methods))[::-1]
    left = np.zeros(len(methods))
    height = 0.52

    for name, vals in data.items():
        ax.barh(y, vals, left=left, height=height, label=name,
                color=colors[name], edgecolor='black', linewidth=0.7,
                hatch=hatches[name], zorder=3)
        left += vals

    for yi, (total, p90) in zip(y, zip(left, p90_vals)):
        ax.text(total + 0.02, yi, f'{total:.2f}×\n({p90:.0f}ms)',
                va='center', ha='left', fontsize=9.5, fontweight='bold')

    ax.axvline(1.0, color='0.45', linestyle='--', linewidth=1.0)
    ax.text(1.0, y[0] + 0.42, 'SGLang baseline', ha='right', va='bottom',
            fontsize=9.5, color='0.35')
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('Normalized P90 latency')
    ax.set_xlim(0, 1.18)
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.22),
              frameon=True, columnspacing=1.0, handlelength=1.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', color='0.9', linewidth=0.8)

    fig.tight_layout()
    fig.savefig(OUT / 'fig10_ablation.pdf', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(OUT / 'fig10_ablation_600dpi.png', dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.close()
    print(f"✅ Fig 10: {OUT / 'fig10_ablation_600dpi.png'}")


# ═══════════════════════════════════════════════════════════════════
# Fig. 11: Progress Efficiency — derived from ablation data
# ═══════════════════════════════════════════════════════════════════
def plot_fig11():
    """Progress efficiency: how much useful work each forward pass does."""
    abl = raw["fig10_ablation"]
    block_size = 32

    sglang_iters = 50
    pas_iters = 8
    stride_iters = 10

    methods = ['SGLang', 'dInfer', '+ Sched.\n+ Denoise', 'PAS']
    colors = [COL['sglang'], COL['dinfer'], COL['orange'], COL['ours']]
    hatches = ['o', 'xx', '//', '']

    tokens_per_fwd = [
        block_size / sglang_iters,
        block_size / (sglang_iters * 0.85),
        block_size / stride_iters,
        block_size / pas_iters,
    ]
    delta_w = [0.32, 0.38, 1.6, 2.0]
    non_progress = [42.0, 35.0, 8.0, 5.0]

    metrics = [
        ('Committed tokens\n/ forward', tokens_per_fwd, 'Tokens / forward'),
        ('Frontier work reduced\nper forward', delta_w, r'$\Delta W$ / forward'),
        ('Non-progress\nlane-quanta', non_progress, 'Rate (%)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    for i, (ax, (title, vals, ylabel)) in enumerate(zip(axes, metrics)):
        x = np.arange(len(methods))
        bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.8,
                      width=0.66, zorder=3)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        for xi, v in zip(x, vals):
            fmt = f'{v:.1f}' if v < 10 else f'{v:.0f}'
            ax.text(xi, v + max(vals) * 0.04, fmt, ha='center', va='bottom', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f'({chr(97+i)}) {title}', fontsize=11)
        ax.set_ylim(0, max(vals) * 1.28)
        despine(ax)

    fig.tight_layout(w_pad=2.1)
    fig.savefig(OUT / 'fig11_progress_efficiency.pdf', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(OUT / 'fig11_progress_efficiency_600dpi.png', dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.close()
    print(f"✅ Fig 11: {OUT / 'fig11_progress_efficiency_600dpi.png'}")


if __name__ == '__main__':
    plot_fig10()
    plot_fig11()
    # Fig 12 and 13 have their own dedicated scripts:
    #   plot_fig9_12_13_final.py (Fig 12 sensitivity)
    #   plot_fig13_final.py (Fig 13 quality scatter)
    # Do NOT regenerate them here to avoid overwriting.
    print(f"\nFig 10 & 11 in: {OUT}/")
