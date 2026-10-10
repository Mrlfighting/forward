"""绘制四静态点矩形模型的中文论文对比图与相位核对图。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatterMathtext
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "section_23" / "231_phase_noise" / "four_static_rectangles_v1"
LEGACY_MODELS = PROJECT_ROOT / "data" / "reference" / "section_23" / "231_phase_noise" / "legacy_four_panel_models_v1"
LEGACY_SHA256 = {
    "clean_full": "c94016846fa0d597f0ae4cc9e91c5e7df31ff316b0bef672a7733ae9165eb780",
    "static_full": "010cd5b089cc5411a9ab6065141b527959aeb520b20dbf37f84fd3dcb71071af",
    "phase_preserving": "6899579938eb5997848a4b7b7a3fae2aded429aa0f23ad241e6e8f62fb2d0e22",
    "phase_plus_shape": "d5d675620d932a58eae1c81ab8f7fb12ffe22b7b6122c872a0dfb6cf36e8bb11",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def draw_models(directory: Path, legacy_models: Path | None = None) -> None:
    """沿用旧图四面板、矩形框与统一对数电阻率色标。"""

    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with np.load(directory / "input.npz", allow_pickle=False) as saved:
        x_nodes = saved["mesh_x_nodes_m"] / 1000.0
        z_nodes = saved["mesh_z_nodes_m"] / 1000.0
        stations = saved["station_x_m"] / 1000.0
        static_indices = saved["static_station_indices"]
    panels = (
        ("clean_full", "（a）无浅部体：全数据参考"),
        ("static_full", "（b）四静态点：全数据"),
        ("phase_preserving", "（c）四静态点：相位保留"),
        ("phase_plus_shape", "（d）四静态点：相位与形态"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 5.2), facecolor="white")
    norm = LogNorm(vmin=10.0, vmax=1000.0)
    for axis, (name, title) in zip(axes.ravel(), panels):
        if legacy_models is None:
            with np.load(directory / f"{name}.npz", allow_pickle=False) as saved:
                recovered = saved["recovered_log_conductivity"]
        else:
            path = legacy_models / f"{name}.npy"
            if hashlib.sha256(path.read_bytes()).hexdigest() != LEGACY_SHA256[name]:
                raise ValueError(f"旧图恢复模型哈希不匹配：{name}")
            recovered = np.load(path, allow_pickle=False)
        rho = np.exp(-recovered).reshape((x_nodes.size - 1, z_nodes.size - 1), order="F").T
        artist = axis.pcolormesh(x_nodes, z_nodes, rho, cmap="jet_r", norm=norm, shading="flat")
        for rectangle in report["deep_rectangles"]:
            width = rectangle["width_m"] / 1000.0
            height = rectangle["height_m"] / 1000.0
            x0 = rectangle["center_x_m"] / 1000.0 - width / 2.0
            z0 = rectangle["center_z_m"] / 1000.0 - height / 2.0
            axis.add_patch(Rectangle(
                (x0, z0), width, height, fill=False,
                edgecolor="black", linestyle="--", linewidth=0.95,
            ))
        colors = np.full(stations.size, "#202020", dtype=object)
        if name != "clean_full":
            colors[static_indices] = "#df3030"
        axis.scatter(stations, np.full(stations.size, 0.12), marker="v", s=24, c=colors, clip_on=False, zorder=5)
        axis.set_xlim(-10.5, 10.5)
        axis.set_ylim(-12.0, 0.65)
        axis.set_xticks(np.arange(-10.0, 10.1, 2.5))
        axis.set_yticks([0.0, -2.5, -5.0, -7.5, -10.0])
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("测线位置（千米）", fontsize=9)
        axis.set_ylabel("深度（千米）", fontsize=9)
        axis.tick_params(labelsize=8)
        colorbar = figure.colorbar(
            artist, ax=axis, ticks=(10.0, 100.0, 1000.0),
            format=LogFormatterMathtext(base=10), fraction=0.038, pad=0.026,
        )
        colorbar.set_label("电阻率（Ω·m）", fontsize=8)
        colorbar.ax.tick_params(labelsize=8)
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.11, top=0.94, wspace=0.33, hspace=0.43)
    output_name = "TM_四静态点矩形目标反演对比.png" if legacy_models is None else "TM_旧图历史恢复模型中文重绘.png"
    figure.savefig(directory / output_name, dpi=190, facecolor="white")
    plt.close(figure)


def draw_phase_noise(directory: Path) -> None:
    """逐静态点展示旧图六个随机频点的实际相位干扰。"""

    with np.load(directory / "input.npz", allow_pickle=False) as saved:
        frequencies = saved["frequencies_hz"]
        stations = saved["station_x_m"] / 1000.0
        static_indices = saved["static_station_indices"]
        original = saved["static_full_before_interference"].reshape(24, 2, 8)[:, 1, :]
        corrupted = saved["static_full"].reshape(24, 2, 8)[:, 1, :]
        mask = saved["phase_noise_mask"]
    figure, axes = plt.subplots(2, 2, figsize=(9.8, 6.5), sharex=True, facecolor="white")
    for axis, station_index in zip(axes.ravel(), static_indices):
        axis.semilogx(frequencies, original[:, station_index], "o-", ms=3, lw=1, color="#557d9a", label="干扰前")
        axis.semilogx(frequencies, corrupted[:, station_index], "s-", ms=3, lw=1, color="#c23f37", label="干扰后")
        for frequency in frequencies[mask[:, station_index]]:
            axis.axvline(frequency, color="#dfb34d", lw=0.8, alpha=0.45)
        axis.set_title(f"静态点 {stations[station_index]:+g} 千米")
        axis.set_xlabel("频率（赫兹）")
        axis.set_ylabel("相位（度）")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / "TM_四静态点随机相位噪声核对.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """只使用已完成的数值结果绘图。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    directory = parser.parse_args().input_dir.resolve()
    draw_models(directory)
    draw_models(directory, LEGACY_MODELS)
    draw_phase_noise(directory)


if __name__ == "__main__":
    main()
