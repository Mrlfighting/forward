"""教学案例的模型网格、正演曲线与反演结果绘图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np

from grid_model import build_true_model


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _grid(mesh, values: np.ndarray) -> np.ndarray:
    """将 SimPEG 的 Fortran 顺序模型改成绘图所需的二维数组。"""

    return np.asarray(values).reshape(mesh.shape_cells, order="F").T


def _outline(axis, config: dict) -> None:
    """用虚线标出单低阻体的真实边界。"""

    body = config["anomaly"]
    axis.add_patch(Rectangle(
        (body["x_min_m"] / 1000.0, body["z_min_m"] / 1000.0),
        (body["x_max_m"] - body["x_min_m"]) / 1000.0,
        (body["z_max_m"] - body["z_min_m"]) / 1000.0,
        fill=False, edgecolor="black", linestyle="--", linewidth=1.2,
    ))


def plot_model_grid(forward_mesh, inversion_mesh, true_model: np.ndarray,
                    stations: np.ndarray, config: dict, output: Path) -> None:
    """展示正演真模型及正反演网格差异。"""

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), layout="constrained")
    rho = np.exp(-true_model)
    artist = axes[0].pcolormesh(
        forward_mesh.nodes_x / 1000.0, forward_mesh.nodes_y / 1000.0,
        _grid(forward_mesh, rho), cmap="jet_r", norm=LogNorm(vmin=10.0, vmax=100.0), shading="flat",
    )
    axes[0].set_title("正演真模型与测点")
    axes[0].scatter(stations / 1000.0, np.full(stations.size, 0.08), marker="v", s=22,
                    c="#202020", clip_on=False, zorder=5)
    axes[1].set_title("正演细网格与反演粗网格")
    for axis, mesh, color in ((axes[1], forward_mesh, "#88a8bf"),
                              (axes[1], inversion_mesh, "#313d4d")):
        for position in mesh.nodes_x:
            if -5000.0 <= position <= 5000.0:
                axis.axvline(position / 1000.0, color=color, lw=0.32 if mesh is forward_mesh else 0.72,
                             alpha=0.55 if mesh is forward_mesh else 0.8)
        for depth in mesh.nodes_y:
            if -4500.0 <= depth <= 0.0:
                axis.axhline(depth / 1000.0, color=color, lw=0.32 if mesh is forward_mesh else 0.72,
                             alpha=0.55 if mesh is forward_mesh else 0.8)
    for axis in axes:
        _outline(axis, config)
        axis.set_xlim(-5.0, 5.0)
        axis.set_ylim(-4.5, 0.25)
        axis.set_xlabel("测线位置（千米）")
        axis.set_ylabel("深度（千米）")
    axes[1].text(0.02, 0.03, "浅蓝：正演网格｜深灰：反演网格", transform=axes[1].transAxes,
                 fontsize=8, ha="left", va="bottom", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    figure.colorbar(artist, ax=axes[0], label="电阻率（Ω·m）", shrink=0.88)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _curve_axes(axes) -> None:
    """按参考图固定周期、视电阻率和相位坐标范围。"""

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlim(1e-3, 1e3)
        axis.grid(which="major", color="#a0a0a0", alpha=0.45, linewidth=0.8)
        axis.grid(which="minor", color="#a0a0a0", alpha=0.25, linewidth=0.5)
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e1, 1e4)
    axes[0].set_ylabel("视电阻率（Ω·m）")
    axes[1].set_ylim(0.0, 90.0)
    axes[1].set_yticks((0.0, 30.0, 60.0, 90.0))
    axes[1].set_ylabel("相位（°）")
    axes[1].set_xlabel("周期（秒）")


def plot_forward_curves(frequencies: np.ndarray, stations: np.ndarray,
                        baseline: dict, anomaly: dict, output: Path) -> None:
    """按参考图样式比较中心测点的均匀背景和单低阻体响应。"""

    period = 1.0 / frequencies
    order = np.argsort(period)
    period = period[order]
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 9.6), sharex=True, layout="constrained")
    location = 0.0
    index = int(np.flatnonzero(stations == location)[0])
    axes[0].plot(period, baseline["apparent_resistivity"][order, index], "o--",
                 color="#1f77b4", markerfacecolor="white", markeredgecolor="#202020",
                 linewidth=1.8, markersize=5.0, label="均匀背景")
    axes[0].plot(period, anomaly["apparent_resistivity"][order, index], "-",
                 color="#d62728", linewidth=2.2, label="含低阻体")
    axes[1].plot(period, anomaly["phase_display_deg"][order, index], "-",
                 color="#d62728", linewidth=2.2)
    axes[1].plot(period, baseline["phase_display_deg"][order, index], "s--",
                 color="#1f77b4", markerfacecolor="white", markeredgecolor="#202020",
                 linewidth=1.8, markersize=5.0)
    axes[0].set_title("TM｜0 千米测点", fontsize=16)
    _curve_axes(axes)
    axes[0].legend(fontsize=11, loc="upper left")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_inversion_models(mesh, recovered: np.ndarray, config: dict, output: Path) -> None:
    """在相同色标下并列展示真模型与反演恢复模型。"""

    true_model, _ = build_true_model(mesh, config)
    panels = ((true_model, "低阻体真模型"), (recovered, "反演恢复模型"))
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), layout="constrained")
    artist = None
    for axis, (model, title) in zip(axes, panels):
        artist = axis.pcolormesh(
            mesh.nodes_x / 1000.0, mesh.nodes_y / 1000.0,
            _grid(mesh, np.exp(-model)), cmap="jet_r", norm=LogNorm(vmin=10.0, vmax=100.0), shading="flat",
        )
        _outline(axis, config)
        axis.set_xlim(-5.0, 5.0)
        axis.set_ylim(-5.0, 0.15)
        axis.set_title(title)
        axis.set_xlabel("测线位置（千米）")
        axis.set_ylabel("深度（千米）")
    figure.colorbar(artist, ax=axes, label="电阻率（Ω·m）", shrink=0.88)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_inversion_fit(frequencies: np.ndarray, stations: np.ndarray,
                       observed: np.ndarray, predicted: np.ndarray, output: Path) -> None:
    """按相同坐标比较中心测点的观测曲线和反演预测曲线。"""

    n_frequency = frequencies.size
    n_station = stations.size
    observed_blocks = observed.reshape(n_frequency, 2, n_station)
    predicted_blocks = predicted.reshape(n_frequency, 2, n_station)
    period = 1.0 / frequencies
    order = np.argsort(period)
    period = period[order]
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 9.6), sharex=True, layout="constrained")
    location = 0.0
    index = int(np.flatnonzero(stations == location)[0])
    axes[0].plot(period, observed_blocks[order, 0, index], "o", color="#202020",
                 markerfacecolor="white", markersize=5.0, label="正演观测")
    axes[0].plot(period, predicted_blocks[order, 0, index], "-", color="#d62728",
                 linewidth=2.2, label="反演预测")
    axes[1].plot(period, np.mod(observed_blocks[order, 1, index], 180.0), "s",
                 color="#202020", markerfacecolor="white", markersize=5.0)
    axes[1].plot(period, np.mod(predicted_blocks[order, 1, index], 180.0), "-",
                 color="#d62728", linewidth=2.2)
    axes[0].set_title("TM｜0 千米测点：反演拟合", fontsize=16)
    _curve_axes(axes)
    axes[0].legend(fontsize=11, loc="upper left")
    figure.savefig(output, dpi=180)
    plt.close(figure)
