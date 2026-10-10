"""论文 2.2.1 识别过程的诊断绘图，不参与候选判定。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_spatial_profiles(result, rho_a: np.ndarray, output_dir: Path) -> None:
    """展示固定三个频率上的原测点与插值剖面。"""

    indices = sorted(set([0, len(result.frequency_diagnostics) // 2, len(result.frequency_diagnostics) - 1]))
    fig, axes = plt.subplots(len(indices), 1, figsize=(9, 7), sharex=True)
    for ax, index in zip(np.atleast_1d(axes), indices):
        item = result.frequency_diagnostics[index]
        ax.plot(item.sample_x_m / 1000.0, item.signal_profile, color="#175e8a", lw=1.4)
        ax.scatter(item.station_x_m / 1000.0, item.signal_at_stations, s=18, color="#c24b2c", zorder=3)
        ax.set_ylabel("空间信号")
        ax.set_title(f"频率 {item.frequency_hz:.4g} Hz")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("测点位置（千米）")
    fig.suptitle("原始测点与空间插值剖面")
    fig.tight_layout()
    fig.savefig(output_dir / "空间剖面诊断.png", dpi=180)
    plt.close(fig)


def plot_maxima_map(result, output_dir: Path) -> None:
    """展示固定最高频率的极大值、完整传播线与候选线。"""

    item = result.frequency_diagnostics[-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    scales = np.asarray(result.config.scales)
    for row, scale in enumerate(scales):
        indices = np.flatnonzero(item.modulus_maxima[row])
        signs = np.sign(item.coefficients[row, indices])
        ax.scatter(item.sample_x_m[indices][signs > 0] / 1000.0, np.full(np.count_nonzero(signs > 0), scale), s=9, c="#b43735")
        ax.scatter(item.sample_x_m[indices][signs < 0] / 1000.0, np.full(np.count_nonzero(signs < 0), scale), s=9, c="#24659a")
    for line in item.maxima_lines:
        if not line.accepted:
            continue
        nodes = sorted(line.nodes, key=lambda node: node.scale_index)
        ax.plot(
            [node.x_m / 1000.0 for node in nodes],
            [scales[node.scale_index] for node in nodes],
            color="#dd8c12" if line.static_candidate else "0.55",
            lw=1.2 if line.static_candidate else 0.5,
            alpha=0.85 if line.static_candidate else 0.4,
        )
    ax.set_yscale("log", base=2)
    ax.set_yticks(scales, labels=[str(scale) for scale in scales])
    ax.set_xlabel("位置（千米）")
    ax.set_ylabel("小波尺度（采样点）")
    ax.set_title(f"模极大值与传播线｜{item.frequency_hz:.4g} Hz")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_dir / "模极大值传播线.png", dpi=180)
    plt.close(fig)


def plot_alpha_map(result, output_dir: Path) -> None:
    """展示逐频候选线的最小相邻尺度指数。"""

    alpha = np.full(result.per_frequency_votes.shape, np.nan)
    for frequency_index, item in enumerate(result.frequency_diagnostics):
        for line in item.maxima_lines:
            if not line.static_candidate or line.mapped_station_index is None:
                continue
            station_index = line.mapped_station_index
            value = min(line.alpha_pairs)
            current = alpha[frequency_index, station_index]
            alpha[frequency_index, station_index] = value if np.isnan(current) else min(value, current)
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(
        np.ma.masked_invalid(alpha),
        origin="lower",
        aspect="auto",
        extent=[result.station_x_m[0] / 1000.0, result.station_x_m[-1] / 1000.0,
                np.log10(result.frequencies_hz[0]), np.log10(result.frequencies_hz[-1])],
        cmap="coolwarm",
    )
    fig.colorbar(image, ax=ax, label="候选线最小相邻尺度 α")
    ax.set_xlabel("映射测点位置（千米）")
    ax.set_ylabel("频率对数 log10(Hz)")
    ax.set_title("逐频负指数候选诊断")
    fig.tight_layout()
    fig.savefig(output_dir / "逐频alpha诊断.png", dpi=180)
    plt.close(fig)


def plot_candidate_stations(result, known_indices: list[int], output_dir: Path) -> None:
    """展示连续候选位置、最近测点和合成真值位置。"""

    fig, ax = plt.subplots(figsize=(10, 5))
    for index in known_indices:
        ax.axvline(result.station_x_m[index] / 1000.0, color="#555555", ls="--", lw=0.8)
    for item in result.frequency_diagnostics:
        positions = [line.candidate_x_m for line in item.maxima_lines if line.static_candidate]
        if positions:
            ax.scatter(np.asarray(positions) / 1000.0, np.full(len(positions), item.frequency_hz),
                       s=14, c="#bd3e31", alpha=0.65)
    ax.set_yscale("log")
    ax.set_xlabel("候选位置（千米）")
    ax.set_ylabel("频率（Hz）")
    ax.set_title("逐频静态候选位置｜虚线为合成真值测点")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "逐频静态候选.png", dpi=180)
    plt.close(fig)


def plot_case_diagnostics(result, rho_a: np.ndarray, known_indices: list[int], output_dir: Path) -> None:
    """保存四类诊断图；图件不反馈到识别计算。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_spatial_profiles(result, rho_a, output_dir)
    plot_maxima_map(result, output_dir)
    plot_alpha_map(result, output_dir)
    plot_candidate_stations(result, known_indices, output_dir)
