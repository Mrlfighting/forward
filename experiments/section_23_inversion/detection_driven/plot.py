"""绘制三场景识别结果及可运行场景的反演对照图。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "simpeg"))
from mt_static.inversion.solver import build_inversion_mesh


OUTPUT_ROOT = ROOT / "results/section_23/detection_driven"
CASE_LABELS = {
    "231_two_sites": "随机相位噪声双静态点模型",
    "231_four_sites": "随机相位噪声四静态点模型",
    "232_layered": "复杂层状双静态点模型",
}
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def plot_detection(case_id: str) -> None:
    output = OUTPUT_ROOT / case_id
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    saved = np.load(output / "detection.npz")
    x = saved["station_x_m"] / 1000.0
    frequency = saved["frequencies_hz"]
    votes = saved["per_frequency_votes"]
    counts = saved["vote_counts"]
    truth = np.asarray(summary["truth_indices_evaluation_only"], dtype=int)
    selected = np.asarray(summary["detected_indices"], dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), layout="constrained")
    bars = axes[0].bar(x, counts, width=0.55, color="#4477aa")
    for index in truth:
        bars[index].set_edgecolor("#cc3311")
        bars[index].set_linewidth(2.2)
    axes[0].axhline(summary["minimum_votes"], color="black", ls="--", lw=1,
                    label=f"至少 {summary['minimum_votes']} 个频率")
    axes[0].scatter(x[selected], counts[selected], marker="x", color="#cc3311",
                    s=110, linewidths=2, label="进入反演的检测点")
    axes[0].set_xticks(x, [f"{value:g}" for value in x])
    axes[0].set(xlabel="测点位置（千米）", ylabel="逐频候选次数")
    axes[0].legend()
    xx = np.r_[x[0] - (x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2]
    yy = np.r_[frequency[0]/np.sqrt(frequency[1]/frequency[0]),
               np.sqrt(frequency[:-1]*frequency[1:]),
               frequency[-1]*np.sqrt(frequency[-1]/frequency[-2])]
    axes[1].pcolormesh(xx, yy, votes.astype(int), cmap="Greys", vmin=0, vmax=1)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="测点位置（千米）", ylabel="频率（Hz）")
    for index in truth:
        axes[1].axvline(x[index], color="#cc3311", ls=":", lw=1)
    for index in selected:
        axes[1].axvline(x[index], color="#0077bb", ls="--", lw=1)
    fig.suptitle(f"{CASE_LABELS[case_id]}｜八测点逐频识别（红框及红点线是真值，仅供评价）")
    fig.savefig(output / "识别与真值对照.png", dpi=220, facecolor="white")
    plt.close(fig)


def plot_inversion(case_id: str) -> None:
    output = OUTPUT_ROOT / case_id
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed_with_detected_mask":
        return
    config = json.loads((ROOT / "configs/section_23/detection_driven.json").read_text(encoding="utf-8"))
    oracle_dir = ROOT / config["cases"][case_id]["oracle_result_directory"]
    mesh = build_inversion_mesh()
    model_paths = (
        ("已知真值位置：相位保留", oracle_dir / "phase_preserving.npz"),
        ("检测位置：相位保留", output / "phase_preserving.npz"),
        ("已知真值位置：相位＋形态", oracle_dir / "phase_plus_shape.npz"),
        ("检测位置：相位＋形态", output / "phase_plus_shape.npz"),
    )
    x = mesh.nodes_x / 1000.0
    z = mesh.nodes_y / 1000.0
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), sharex=True, sharey=True,
                             layout="constrained")
    image = None
    for axis, (label, path) in zip(axes.ravel(), model_paths):
        with np.load(path) as saved:
            recovered = saved["recovered_log_conductivity"]
        log_rho = (-recovered / np.log(10.0)).reshape(mesh.shape_cells, order="F")
        image = axis.pcolormesh(x, z, log_rho.T, cmap="turbo", vmin=0.0, vmax=3.0,
                                shading="flat", rasterized=True)
        for location in summary["truth_x_m_evaluation_only"]:
            axis.plot(location/1000.0, -0.15, "v", color="black", ms=6)
        for location in summary["detected_x_m"]:
            axis.plot(location/1000.0, -0.15, "v", color="#cc3311", ms=6)
        axis.set(xlim=(-10, 10), ylim=(-12, 0), title=label,
                 xlabel="测线位置（千米）", ylabel="深度（千米）")
    assert image is not None
    fig.colorbar(image, ax=axes.ravel().tolist(), label="lg［电阻率／(Ω·m)］")
    fig.suptitle("四静态点模型：已知真值掩码与检测掩码反演对照；黑/红三角分别为真值/检测位置")
    fig.savefig(output / "检测驱动与真值掩码反演对照.png", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    for case_id in ("231_two_sites", "231_four_sites", "232_layered"):
        plot_detection(case_id)
        plot_inversion(case_id)
    print("三个场景的识别对照图和可运行场景的反演对照图已保存", flush=True)


if __name__ == "__main__":
    main()
