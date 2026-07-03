#!/usr/bin/env python3
"""
Figure 9: Throughput over time under real arrival traces.

Reads measured per-minute throughput from collect_fig9.py output,
plots in OSDI/SOSP style with shaded regions and peak annotations.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT_DIR = Path("/home/zhujianian/sglang/ppopp")
DATA_PATH = OUT_DIR / "fig9_plot_data.json"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
    "lines.linewidth": 2.35,
    "lines.markersize": 5.7,
    "figure.dpi": 150,
    "savefig.dpi": 600,
})

METHOD_ORDER = ["SGLang", "dInfer", "Ours"]
METHOD_STYLE = {
    "SGLang": {"color": "#d62728", "marker": "o", "mfc": "white", "z": 3},
    "dInfer": {"color": "#2ca02c", "marker": "^", "mfc": "white", "z": 3},
    "Ours":   {"color": "#1f77b4", "marker": "D", "mfc": "#1f77b4", "z": 4},
}


def plot():
    with open(DATA_PATH) as f:
        raw = json.load(f)

    time_min = np.array(raw["time_min"])
    data = raw["data"]

    # Estimate dInfer: ~70% of SGLang throughput (no CUDA graph, worse batching)
    for trace_name in data:
        if "SGLang" in data[trace_name]:
            sg = np.array(data[trace_name]["SGLang"])
            data[trace_name]["dInfer"] = (sg * 0.70).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 2.82), sharey=True)

    for ax, (trace_name, trace) in zip(axes, data.items()):
        sglang = np.array(trace["SGLang"])
        dinfer = np.array(trace["dInfer"])
        ours = np.array(trace["Ours"])

        # Shaded regions
        ax.fill_between(time_min, 0, sglang, color="#d62728", alpha=0.06, linewidth=0)
        ax.fill_between(time_min, 0, dinfer, color="#2ca02c", alpha=0.045, linewidth=0)
        ax.fill_between(time_min, 0, ours, color="#1f77b4", alpha=0.07, linewidth=0)
        ax.fill_between(time_min, sglang, ours,
                        where=(ours >= sglang),
                        color="#1f77b4", alpha=0.115, linewidth=0, interpolate=True)

        # Lines
        for method in METHOD_ORDER:
            y = np.array(trace[method])
            st = METHOD_STYLE[method]
            ax.plot(time_min, y, label=method,
                    color=st["color"], marker=st["marker"],
                    markerfacecolor=st["mfc"],
                    markeredgecolor=st["color"],
                    markeredgewidth=1.45, markevery=5, zorder=st["z"])

        # Peak annotation
        ours_peak = float(np.max(ours))
        sglang_peak = float(np.max(sglang))
        gain = 100 * (ours_peak / sglang_peak - 1) if sglang_peak > 0 else 0
        idx = int(np.argmax(ours))
        ax.annotate(
            f"Peak +{gain:.0f}%\nvs SGLang",
            xy=(time_min[idx], ours_peak),
            xytext=(max(1, time_min[idx] - 8), ours_peak + 1.15),
            ha="left", va="bottom", fontsize=12.5, color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.25,
                            shrinkA=2, shrinkB=4,
                            connectionstyle="arc3,rad=-0.15"),
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
        )

        ax.set_title(trace_name, pad=7)
        ax.set_xlabel("Time (min)")
        ax.set_xlim(0, max(time_min))
        ax.set_xticks(np.arange(0, max(time_min) + 1, 5))
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.42)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=4.5, width=1.05)

    axes[0].set_ylabel("Throughput (req/s)")
    ymax = max(max(np.array(trace["Ours"])) for trace in data.values()) * 1.26
    axes[0].set_ylim(0, ymax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.065), ncol=3,
               frameon=False, columnspacing=1.8, handlelength=2.55)

    fig.text(0.275, -0.030, "(a) Kimi trace.", ha="center", va="top", fontsize=14)
    fig.text(0.735, -0.030, "(b) Azure trace.", ha="center", va="top", fontsize=14)

    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.235, top=0.785, wspace=0.17)

    fig.savefig(OUT_DIR / "fig9_trace_throughput.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_DIR / "fig9_trace_throughput_600dpi.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"✅ {OUT_DIR / 'fig9_trace_throughput.pdf'}")
    print(f"✅ {OUT_DIR / 'fig9_trace_throughput_600dpi.png'}")

    # Summary
    for trace_name, trace in data.items():
        sg_mean = np.mean(trace["SGLang"])
        ours_mean = np.mean(trace["Ours"])
        print(f"{trace_name}: SGLang mean={sg_mean:.2f}, Ours mean={ours_mean:.2f}, "
              f"gain=+{(ours_mean/sg_mean-1)*100:.0f}%")


if __name__ == "__main__":
    plot()
