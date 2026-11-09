"""于生宝原文模式的逐频负指数证据图，仅作诊断。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_case(result, targets: np.ndarray, output: Path) -> None:
    """画候选连续位置和最近测点频数，不将其冒充最终分类。"""

    events = [event for group in result.negative_pairs for event in group]
    occupancy = np.zeros((result.frequencies_hz.size, result.station_x_m.size), dtype=bool)
    for frequency_index, group in enumerate(result.negative_pairs):
        for event in group:
            occupancy[frequency_index, event.nearest_station_index] = True
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    if events:
        points = axes[0].scatter(
            [event.candidate_x_m / 1000.0 for event in events],
            [event.frequency_hz for event in events],
            c=[event.alpha for event in events],
            cmap="coolwarm", vmin=min(event.alpha for event in events), vmax=0.0,
            s=22, edgecolors="none", alpha=0.75,
        )
        fig.colorbar(points, ax=axes[0], label="相邻尺度 α")
    else:
        axes[0].text(0.5, 0.5, "无负指数尺度对", transform=axes[0].transAxes,
                     ha="center", va="center")
    for target in targets:
        for ax in axes:
            ax.axvline(float(target) / 1000.0, color="black", ls="--", lw=0.9)
    axes[0].set_yscale("log")
    axes[0].set_ylim(result.frequencies_hz.min() / 1.4, result.frequencies_hz.max() * 1.4)
    axes[0].set_ylabel("频率（Hz）")
    axes[0].set_title("逐频负 α 尺度对的最小尺度位置")
    axes[0].grid(alpha=0.2)
    axes[1].bar(result.station_x_m / 1000.0, occupancy.sum(axis=0),
                width=0.35 if result.station_x_m.size > 8 else 0.8,
                color="#2874a6")
    axes[1].set_ylabel("出现频数")
    axes[1].set_xlabel("测点位置（千米）")
    axes[1].set_title("负 α 证据映射至最近原测点，仅供诊断")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "逐频负指数证据.png", dpi=180)
    plt.close(fig)
