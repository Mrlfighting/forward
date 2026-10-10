"""绘制与旧论文图布局一致的复杂层状组合模型中文四面板图。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from matplotlib.ticker import LogFormatterMathtext
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "section_23" / "layered_model.json"
INPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "section_23" / "232_layered_model" / "inputs.npz"
RESULT_DIR = PROJECT_ROOT / "results" / "section_23" / "232_layered_model"
PAPER_DIR = PROJECT_ROOT / "paper" / "figures" / "section_23"
FIGURE_NAME = "TM_复杂层状组合模型反演对比.png"
LEGACY_MODEL_NAMES = {
    "clean_full": ("recovered_clean_reference.npy", "91061e4213a5dd87adf023340a05f5b687cccee463071f8712e52f16a71f9da8"),
    "static_full": ("recovered_moderate_static_full.npy", "d919b5c34fb725e53813b5665e1f5dd9b4a0a5376cf838055421637bdab4881f"),
    "phase_preserving": ("recovered_moderate_phase_only.npy", "25a46acd71db4904e8f276039297a71aedd1a3314f23f05131654fe12192e6d9"),
    "phase_plus_shape": ("recovered_moderate_shape.npy", "9cb2213e75a34396bad3eebbb19c759705daabdb9c0357d54f82bc5caadcd24c"),
}


def main() -> None:
    """在共同 10–1000 Ω·m 色标下绘制四组恢复模型。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--width-px", type=int, default=980)
    parser.add_argument("--height-px", type=int, default=400)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--legacy-model-dir", type=Path)
    args = parser.parse_args()
    if min(args.width_px, args.height_px, args.dpi) <= 0:
        raise ValueError("图像尺寸和 DPI 必须为正")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    summary = None
    if args.legacy_model_dir is None:
        summary = json.loads((RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] not in {"validated", "needs_review"}:
            raise ValueError("四组反演结果尚未完成")
    with np.load(INPUT_PATH, allow_pickle=False) as saved:
        x_nodes = saved["inversion_mesh_x_nodes_m"] / 1000.0
        z_nodes = saved["inversion_mesh_z_nodes_m"] / 1000.0
        stations = saved["station_x_m"] / 1000.0
        static_indices = saved["static_station_indices"]
    plt.rcParams.update({
        "font.family": ["SimSun", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "axes.unicode_minus": False,
    })
    panels = (
        ("clean_full", "(a)", False),
        ("static_full", "(b)", True),
        ("phase_preserving", "(c)", True),
        ("phase_plus_shape", "(d)", True),
    )
    figure, axes = plt.subplots(
        2, 2,
        figsize=(args.width_px / args.dpi, args.height_px / args.dpi),
        dpi=args.dpi,
        facecolor="white",
    )
    norm = LogNorm(vmin=10.0, vmax=1000.0)
    for axis, (method, label, show_static) in zip(axes.ravel(), panels):
        if args.legacy_model_dir is None:
            with np.load(RESULT_DIR / f"{method}.npz", allow_pickle=False) as saved:
                recovered = saved["recovered_log_conductivity"]
        else:
            filename, expected = LEGACY_MODEL_NAMES[method]
            path = args.legacy_model_dir / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"旧恢复模型哈希不一致：{filename}")
            recovered = np.load(path, allow_pickle=False)
        rho = np.exp(-recovered).reshape(
            (x_nodes.size - 1, z_nodes.size - 1), order="F"
        ).T
        artist = axis.pcolormesh(
            x_nodes, z_nodes, rho, shading="flat", cmap="jet_r", norm=norm
        )
        for target in config["targets"]:
            axis.add_patch(Ellipse(
                (target["center_x_m"] / 1000.0, target["center_z_m"] / 1000.0),
                2.0 * target["radius_x_m"] / 1000.0,
                2.0 * target["radius_z_m"] / 1000.0,
                fill=False, edgecolor="black", linestyle="--", linewidth=0.7,
            ))
        axis.axhline(-1.0, color="black", linestyle=":", linewidth=0.55)
        axis.axhline(-8.0, color="black", linestyle=":", linewidth=0.55)
        colors = np.full(stations.size, "black", dtype=object)
        if show_static:
            colors[static_indices] = "red"
        axis.scatter(
            stations, np.full(stations.size, 0.12),
            marker="v", s=11, c=colors, clip_on=False,
            zorder=6, linewidths=0,
        )
        axis.set_xlim(-10.5, 10.5)
        axis.set_ylim(-12.0, 0.65)
        axis.set_xlabel("测线位置（千米）")
        axis.set_ylabel("深度（千米）")
        axis.text(
            0.015, 0.965, label,
            transform=axis.transAxes,
            ha="left", va="top", fontsize=9.0, color="black",
        )
        axis.tick_params(pad=1.0)
        colorbar = figure.colorbar(
            artist, ax=axis, ticks=[10.0, 100.0, 1000.0],
            format=LogFormatterMathtext(base=10),
            fraction=0.038, pad=0.035,
        )
        colorbar.ax.tick_params(labelsize=8.0, width=0.45, length=2.0, pad=1.0)
        colorbar.outline.set_linewidth(0.45)
        colorbar.set_label("电阻率（Ω·m）", fontsize=8.0, labelpad=1.0)
    figure.subplots_adjust(
        left=0.075, right=0.965, bottom=0.095, top=0.98,
        wspace=0.36, hspace=0.24,
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if args.legacy_model_dir is not None:
        filename = "TM_复杂层状组合模型_历史结果中文重绘.png"
    else:
        filename = FIGURE_NAME if summary["status"] == "validated" else FIGURE_NAME.replace(".png", "_待核查.png")
    paths = [RESULT_DIR / filename]
    if summary is not None and summary["status"] == "validated":
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        paths.append(PAPER_DIR / filename)
    for path in paths:
        figure.savefig(path, dpi=args.dpi, bbox_inches=None, facecolor="white")
    plt.close(figure)
    status = "旧数组重绘" if summary is None else summary["status"]
    print(f"状态={status}；已生成 {args.width_px}×{args.height_px} 中文四面板图：{RESULT_DIR / filename}")


if __name__ == "__main__":
    main()
