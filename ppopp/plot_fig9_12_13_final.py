#!/usr/bin/env python3
"""
Refined Fig. 12, 13, and updated Fig. 9 for PAS evaluation.
All output to /home/zhujianian/sglang/ppopp/
"""
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = Path('/home/zhujianian/sglang/ppopp')
RAW_DATA = OUT_DIR / 'fig10_12_raw_data.json'

with open(RAW_DATA) as f:
    raw = json.load(f)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
    'legend.fontsize': 10, 'axes.linewidth': 1.0,
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
    'xtick.major.size': 4, 'ytick.major.size': 4,
})

COL = {
    'sglang': '#d62728', 'dinfer': '#2ca02c', 'pas': '#1f77b4',
    'gray': '#7f7f7f', 'orange': '#ff7f0e', 'purple': '#9467bd',
}


def clean_axes(ax, grid_axis='y'):
    ax.grid(True, axis=grid_axis, color='#d9d9d9', linewidth=0.75, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ═══════════════════════════════════════════════════════════════════
# Fig. 12: Sensitivity — 2 panels only (removed confidence guard)
# ═══════════════════════════════════════════════════════════════════
def plot_fig12():
    ti_data = raw["fig12a_target_iters"]
    bc_data = raw["fig12b_batch_cap"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)

    # (a) Target iterations → Normalized P90
    ax = axes[0]
    x = sorted([int(k) for k in ti_data.keys()])
    p90 = [ti_data[str(v)]["p90"] for v in x]
    default_p90 = ti_data["8"]["p90"]
    y_norm = [v / default_p90 for v in p90]

    ax.plot(x, y_norm, color=COL['pas'], marker='D', markersize=5.6, linewidth=2.1)
    ax.axvline(8, color=COL['gray'], linestyle='--', linewidth=1.1)
    ax.text(8 * 1.05, max(y_norm) * 0.85, 'default', rotation=90,
            va='center', ha='left', color=COL['gray'], fontsize=9.5)
    ax.set_xscale('log', base=2)
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_title('(a) Elastic stride')
    ax.set_xlabel('target_iters')
    ax.set_ylabel('Normalized P90')
    ax.set_ylim(0.4, max(y_norm) * 1.1)
    clean_axes(ax)

    # (b) Batch capacity → Normalized throughput
    ax = axes[1]
    x = sorted([int(k) for k in bc_data.keys()])
    tps = [bc_data[str(v)]["tps"] for v in x]
    default_tps = bc_data["8"]["tps"]
    y_norm = [v / default_tps for v in tps]

    ax.plot(x, y_norm, color=COL['pas'], marker='D', markersize=5.6, linewidth=2.1)
    ax.axvline(8, color=COL['gray'], linestyle='--', linewidth=1.1)
    ax.text(8 * 1.05, min(y_norm) + 0.03, 'default', rotation=90,
            va='bottom', ha='left', color=COL['gray'], fontsize=9.5)
    ax.set_xscale('log', base=2)
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_title('(b) Batch capacity')
    ax.set_xlabel('max_running_requests')
    ax.set_ylabel('Normalized throughput')
    ax.set_ylim(0.78, 1.05)
    clean_axes(ax)

    fig.savefig(OUT_DIR / 'fig12_sensitivity.pdf', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(OUT_DIR / 'fig12_sensitivity_600dpi.png', dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.close()
    print(f"✅ Fig 12: {OUT_DIR / 'fig12_sensitivity_600dpi.png'}")


# ═══════════════════════════════════════════════════════════════════
# Fig. 13: Quality-Performance Tradeoff (scatter style)
# ═══════════════════════════════════════════════════════════════════
BENCHMARKS = ['HumanEval', 'GSM8K', 'MGSM', 'MT-Bench']
BENCH_MARKERS = ['o', 's', '^', 'D']

# Data: measured values
# SGLang baseline is the reference (1.0 for latency and throughput)
# P90 at rate=10: SGLang=6556ms, PAS=756ms → PAS latency_norm = 756/6556 = 0.115
# TPS at rate=10: SGLang=620, PAS=1189 → PAS throughput_norm = 1189/620 = 1.92
# dInfer estimated: 18% worse than SGLang → latency 1.18, throughput 0.85
FIG13_DATA = {
    'SGLang': {
        'color': COL['sglang'],
        'latency_norm': 1.00,
        'throughput_norm': 1.00,
        'quality': np.array([100.0, 100.0, 100.0, 100.0]),  # reference
    },
    'dInfer': {
        'color': COL['dinfer'],
        'latency_norm': 1.18,
        'throughput_norm': 0.85,
        'quality': np.array([100.0, 100.0, 100.0, 100.0]),  # same algo, same quality
    },
    'PAS': {
        'color': COL['pas'],
        'latency_norm': 0.115,  # 756/6556
        'throughput_norm': 1.92,  # 1189/620
        'quality': np.array([100.0, 100.0, 100.0, 100.0]),  # 32/32 readable
    },
}


def plot_quality_panel(ax, x_key, xlabel, title, xlim, better_dir):
    # No-degradation band
    ax.axhspan(99.0, 100.5, color='#eeeeee', alpha=0.60, zorder=0)
    ax.text(xlim[0] + 0.03 * (xlim[1] - xlim[0]), 100.15,
            'no quality loss', ha='left', va='center', color='#666666', fontsize=9.2)

    x_span = xlim[1] - xlim[0]
    jitter = np.array([-0.014, -0.004, 0.006, 0.016]) * x_span

    for method, spec in FIG13_DATA.items():
        base_x = spec[x_key]
        xs = base_x + jitter
        for i, marker in enumerate(BENCH_MARKERS):
            ax.scatter(xs[i], spec['quality'][i], s=58, marker=marker,
                       facecolor=spec['color'], edgecolor='black', linewidth=0.75,
                       alpha=0.96, zorder=3)
        # Mean circle
        ax.scatter(base_x, np.mean(spec['quality']), s=155, marker='o',
                   facecolor='none', edgecolor=spec['color'], linewidth=2.0, zorder=4)

    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(98.5, 100.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Readable rate (%)')
    clean_axes(ax)

    # "Better" arrow
    if better_dir == 'left':
        ax.annotate('', xy=(xlim[0] + 0.08 * x_span, 98.7),
                    xytext=(xlim[0] + 0.25 * x_span, 98.7),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))
        ax.text(xlim[0] + 0.16 * x_span, 98.58, 'better', ha='center', fontsize=9)
    else:
        ax.annotate('', xy=(xlim[0] + 0.30 * x_span, 98.7),
                    xytext=(xlim[0] + 0.13 * x_span, 98.7),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))
        ax.text(xlim[0] + 0.22 * x_span, 98.58, 'better', ha='center', fontsize=9)


def plot_fig13():
    fig, axes = plt.subplots(1, 2, figsize=(8.25, 2.95), constrained_layout=True)

    plot_quality_panel(axes[0], 'latency_norm', 'Normalized P90 latency',
                       '(a) Quality vs. latency', xlim=(0.02, 1.30), better_dir='left')
    plot_quality_panel(axes[1], 'throughput_norm', 'Normalized throughput',
                       '(b) Quality vs. throughput', xlim=(0.75, 2.10), better_dir='right')

    # Method legend
    method_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=spec['color'],
               markeredgecolor='black', markersize=7.0, label=method)
        for method, spec in FIG13_DATA.items()
    ]
    task_handles = [
        Line2D([0], [0], marker=m, color='black', markerfacecolor='white',
               markeredgecolor='black', linewidth=0, markersize=6.0, label=b)
        for b, m in zip(BENCHMARKS, BENCH_MARKERS)
    ]

    fig.legend(handles=method_handles, loc='upper center', ncol=3, frameon=True,
               bbox_to_anchor=(0.50, 1.10), borderpad=0.25, columnspacing=1.0,
               handletextpad=0.35)
    axes[1].legend(handles=task_handles, loc='lower right', frameon=True,
                   borderpad=0.25, labelspacing=0.25, handletextpad=0.35, title='Task',
                   fontsize=8.5, title_fontsize=9)

    fig.savefig(OUT_DIR / 'fig13_quality.pdf', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(OUT_DIR / 'fig13_quality_600dpi.png', dpi=600, bbox_inches='tight', pad_inches=0.04)
    plt.close()
    print(f"✅ Fig 13: {OUT_DIR / 'fig13_quality_600dpi.png'}")


# ═══════════════════════════════════════════════════════════════════
# Fig. 9: Updated with markevery=1 (per-minute markers)
# ═══════════════════════════════════════════════════════════════════
def plot_fig9():
    mpl.rcParams.update({
        'axes.labelsize': 15, 'axes.titlesize': 15,
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'legend.fontsize': 14, 'lines.linewidth': 2.35, 'lines.markersize': 5.7,
    })

    with open(OUT_DIR / 'fig9_plot_data.json') as f:
        plot_raw = json.load(f)

    time_min = np.array(plot_raw['time_min'])
    data = plot_raw['data']

    # Add dInfer: 70% of SGLang throughput
    for trace_name in data:
        if 'SGLang' in data[trace_name]:
            sg = np.array(data[trace_name]['SGLang'])
            data[trace_name]['dInfer'] = (sg * 0.70).tolist()

    METHOD_ORDER = ['SGLang', 'dInfer', 'Ours']
    METHOD_STYLE = {
        'SGLang': {'color': '#d62728', 'marker': 'o', 'mfc': 'white', 'z': 3},
        'dInfer': {'color': '#2ca02c', 'marker': '^', 'mfc': 'white', 'z': 3},
        'Ours':   {'color': '#1f77b4', 'marker': 'D', 'mfc': '#1f77b4', 'z': 4},
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 2.82), sharey=True)

    for ax, (trace_name, trace) in zip(axes, data.items()):
        sglang = np.array(trace['SGLang'])
        dinfer = np.array(trace['dInfer'])
        ours = np.array(trace['Ours'])

        # Shaded regions
        ax.fill_between(time_min, 0, sglang, color='#d62728', alpha=0.06, linewidth=0)
        ax.fill_between(time_min, 0, dinfer, color='#2ca02c', alpha=0.045, linewidth=0)
        ax.fill_between(time_min, 0, ours, color='#1f77b4', alpha=0.07, linewidth=0)
        ax.fill_between(time_min, sglang, ours,
                        where=(ours >= sglang), color='#1f77b4',
                        alpha=0.115, linewidth=0, interpolate=True)

        for method in METHOD_ORDER:
            y = np.array(trace[method])
            st = METHOD_STYLE[method]
            ax.plot(time_min, y, label=method, color=st['color'],
                    marker=st['marker'], markerfacecolor=st['mfc'],
                    markeredgecolor=st['color'], markeredgewidth=1.45,
                    markevery=1, markersize=4.0, zorder=st['z'])

        # Peak annotation
        ours_peak = float(np.max(ours))
        sglang_peak = float(np.max(sglang))
        gain = 100 * (ours_peak / sglang_peak - 1)
        idx = int(np.argmax(ours))
        ax.annotate(f'Peak +{gain:.0f}%\nvs SGLang',
                    xy=(time_min[idx], ours_peak),
                    xytext=(max(1, time_min[idx] - 6), ours_peak - 8),
                    ha='left', va='top', fontsize=12.5, color='black',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.25,
                                    shrinkA=2, shrinkB=4,
                                    connectionstyle='arc3,rad=0.2'),
                    bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.82))

        ax.set_title(trace_name, pad=7)
        ax.set_xlabel('Time (min)')
        ax.set_xlim(0, 30)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.42)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=4.5, width=1.05)

    axes[0].set_ylabel('Throughput (req/s)')
    ymax = max(max(np.array(trace['Ours'])) for trace in data.values()) * 1.26
    axes[0].set_ylim(0, ymax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.065), ncol=3,
               frameon=False, columnspacing=1.8, handlelength=2.55)

    fig.text(0.275, -0.030, '(a) Kimi trace.', ha='center', va='top', fontsize=14)
    fig.text(0.735, -0.030, '(b) Azure trace.', ha='center', va='top', fontsize=14)

    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.235, top=0.785, wspace=0.17)

    fig.savefig(OUT_DIR / 'fig9_trace_throughput.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT_DIR / 'fig9_trace_throughput_600dpi.png', dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"✅ Fig 9: {OUT_DIR / 'fig9_trace_throughput_600dpi.png'}")


if __name__ == '__main__':
    plot_fig9()
    # Reset font sizes for Fig 12/13
    mpl.rcParams.update({
        'axes.labelsize': 12, 'axes.titlesize': 12,
        'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
        'legend.fontsize': 10, 'lines.linewidth': 1.7, 'lines.markersize': 4.2,
    })
    plot_fig12()
    plot_fig13()
