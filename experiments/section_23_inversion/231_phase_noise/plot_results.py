"""绘制论文 2.3.1 的中文反演对比图与相位噪声核对图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "section_23" / "231_phase_noise"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_models(destination: Path) -> None:
    """同一色标下并列展示四个方法恢复的深部电阻率。"""

    with np.load(destination / "input.npz", allow_pickle=False) as source:
        x_nodes = source["mesh_x_nodes_m"] / 1000.0
        z_nodes = source["mesh_z_nodes_m"] / 1000.0
        stations = source["station_x_m"] / 1000.0
        static_indices = source["static_station_indices"]
    methods = (
        ("clean_full", "（a）无浅部体全数据参考"),
        ("static_full", "（b）静态全数据"),
        ("phase_preserving", "（c）相位保留法"),
        ("phase_plus_shape", "（d）相位加形态约束"),
    )
    norm = LogNorm(vmin=10.0, vmax=1000.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), layout="constrained")
    image = None
    for ax, (method, title) in zip(axes.ravel(), methods):
        with np.load(destination / f"{method}.npz", allow_pickle=False) as saved:
            log_conductivity = saved["recovered_log_conductivity"]
        rho = np.exp(-log_conductivity).reshape((x_nodes.size - 1, z_nodes.size - 1), order="F").T
        image = ax.pcolormesh(x_nodes, z_nodes, rho, cmap="jet_r", norm=norm, shading="flat")
        for x_min in (-8.0, 2.0):
            ax.add_patch(Rectangle(
                (x_min, -8.0), 6.0, 6.0, fill=False,
                edgecolor="black", linestyle="--", linewidth=1.1,
            ))
        colors = np.array(["#262626"] * stations.size, dtype=object)
        if method != "clean_full":
            colors[static_indices] = "#d33c35"
        ax.scatter(stations, np.full(stations.size, 0.18), c=colors, marker="v", s=24, clip_on=False)
        ax.set_xlim(-10.5, 10.5)
        ax.set_ylim(-10.0, 0.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("测线位置（千米）")
        ax.set_ylabel("深度（千米）")
        ax.tick_params(labelsize=9)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    colorbar.set_label("电阻率（Ω·m）")
    fig.savefig(destination / "TM_随机相位噪声反演对比.png", dpi=190)
    plt.close(fig)


def plot_phase_noise(destination: Path) -> None:
    """逐站核查只有六个指定频点的相位受到随机干扰。"""

    with np.load(destination / "input.npz", allow_pickle=False) as saved:
        frequencies = saved["frequencies_hz"]
        stations = saved["station_x_m"] / 1000.0
        original = saved["static_phase_display_deg"]
        corrupted = saved["corrupted_phase_deg"] + 180.0
        mask = saved["phase_noise_mask"]
        static_indices = saved["static_station_indices"]
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.2), layout="constrained", sharex=True)
    for station_index, ax in enumerate(axes.ravel()):
        ax.semilogx(frequencies, original[:, station_index], "o-", color="#426f98", ms=3, lw=1, label="原始相位")
        ax.semilogx(frequencies, corrupted[:, station_index], "s-", color="#b84b3f", ms=3, lw=1, label="加噪相位")
        for frequency in frequencies[mask[:, station_index]]:
            ax.axvline(frequency, color="#e0ad37", lw=0.8, alpha=0.45)
        station_kind = "静态点" if station_index in static_indices else "普通点"
        ax.set_title(f"{stations[station_index]:+g} 千米（{station_kind}）", fontsize=10)
        ax.set_xlabel("频率（赫兹）")
        ax.set_ylabel("相位（度）")
        ax.grid(alpha=0.18)
    axes[0, 0].set_xlim(frequencies.max() * 1.2, frequencies.min() / 1.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(destination / "八测点随机相位噪声核对.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """只读取已验证并迁移的结果，不再调用正演或反演。"""

    plot_models(OUTPUT_ROOT)
    plot_phase_noise(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
