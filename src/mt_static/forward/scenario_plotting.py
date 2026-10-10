"""为任意浅部体场景绘制模型、拟断面和测点曲线。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .plotting import plot_pseudosections


def _station_label(position_m: float) -> str:
    """生成中文测点位置标签。"""

    if position_m == 0.0:
        return "零米"
    direction = "负" if position_m < 0 else "正" if position_m > 0 else "零"
    return f"{direction}{abs(position_m):g}米"


def plot_model_layout(base: dict, scenario: dict, output_dir: Path) -> Path:
    """同时展示深部结构、浅部体与测点位置。"""

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    stations = np.arange(
        base["stations"]["start_m"],
        base["stations"]["stop_m"] + base["stations"]["step_m"] / 2.0,
        base["stations"]["step_m"],
    )
    axes[0].set_facecolor("#eeeeee")
    axes[1].set_facecolor("#eeeeee")
    for body in base["deep_bodies"]:
        color = "#3984b9" if body["resistivity_ohm_m"] < base["background_resistivity_ohm_m"] else "#c77555"
        axes[0].add_patch(Rectangle(
            (body["x_min_m"] / 1000.0, body["z_min_m"] / 1000.0),
            (body["x_max_m"] - body["x_min_m"]) / 1000.0,
            (body["z_max_m"] - body["z_min_m"]) / 1000.0,
            color=color, alpha=0.85,
        ))
        center = (body["x_min_m"] + body["x_max_m"]) / 2000.0
        axes[0].text(center, (body["z_min_m"] + body["z_max_m"]) / 2000.0,
                     f"{body['resistivity_ohm_m']:g} Ω·m", ha="center", va="center")
    for number, body in enumerate(scenario["shallow_bodies"], start=1):
        left = body["x_min_m"] / 1000.0
        width = (body["x_max_m"] - body["x_min_m"]) / 1000.0
        axes[1].add_patch(Rectangle(
            (left, body["z_min_m"]), width, body["z_max_m"] - body["z_min_m"],
            color="#2b8cbe", alpha=0.9,
        ))
        axes[1].text(left + width / 2.0, body["z_min_m"] - 2.0,
                     f"浅部体{number}\n{body['resistivity_ohm_m']:g} Ω·m",
                     ha="center", va="top", fontsize=8)
        axes[0].axvline(left + width / 2.0, color="#2b8cbe", lw=0.8, ls=":")
    axes[0].scatter(stations / 1000.0, np.zeros_like(stations), marker="v", s=10, color="black")
    axes[1].scatter(stations / 1000.0, np.zeros_like(stations), marker="v", s=10, color="black")
    axes[0].set_ylim(-10.0, 0.5)
    axes[1].set_ylim(-25.0, 3.0)
    axes[0].set_xlim(float(stations.min() / 1000.0), float(stations.max() / 1000.0))
    axes[0].set_ylabel("深度（千米）")
    axes[1].set_ylabel("深度（米）")
    axes[1].set_xlabel("测点位置（千米）")
    axes[0].set_title("深部地电模型与测点")
    axes[1].set_title("地表浅部异常体局部放大")
    for ax in axes:
        ax.grid(alpha=0.12)
    path = output_dir / "模型与测点位置.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_station_curve(
    mode: str,
    responses: dict,
    frequencies_hz: np.ndarray,
    station_x_m: np.ndarray,
    position_m: float,
    output_dir: Path,
) -> Path:
    """按当前论文坐标范围比较指定测点的基线与场景曲线。"""

    station_index = int(np.flatnonzero(np.isclose(station_x_m, position_m, atol=1e-9))[0])
    period_s = 1.0 / frequencies_hz
    order = np.argsort(period_s)
    period_s = period_s[order]
    baseline = responses["without_shallow"]
    affected = responses["with_shallow"]
    baseline_rho = baseline["apparent_resistivity"][order, station_index]
    affected_rho = affected["apparent_resistivity"][order, station_index]
    baseline_phase = baseline["phase_deg"][order, station_index]
    affected_phase = affected["phase_deg"][order, station_index]
    lower = 10.0 ** np.floor(np.log10(min(baseline_rho.min(), affected_rho.min())))
    upper = max(1.0e4, 10.0 ** np.ceil(np.log10(max(baseline_rho.max(), affected_rho.max()))))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.4, 7.2), constrained_layout=True)
    for ax, clean, changed, marker in (
        (axes[0], baseline_rho, affected_rho, "o"),
        (axes[1], baseline_phase, affected_phase, "s"),
    ):
        ax.plot(period_s, clean, color="#1f77b4", ls="--", lw=1.4)
        ax.scatter(period_s, clean, marker=marker, s=18, facecolors="white",
                   edgecolors="#222222", linewidths=0.8, zorder=3, label="无浅部异常体")
        ax.plot(period_s, changed, color="#d62728", lw=1.7, label="有浅部异常体")
        ax.set_xscale("log")
        ax.set_xlim(1.0e-3, 1.0e3)
        ax.grid(which="major", alpha=0.5, lw=0.6)
        ax.grid(which="minor", alpha=0.3, lw=0.4)
    axes[0].set_yscale("log")
    axes[0].set_ylim(lower, upper)
    axes[0].set_ylabel("视电阻率（Ω·m）")
    axes[0].set_title(f"{mode}｜{position_m / 1000.0:g}千米测点")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].set_ylim(0.0, 90.0)
    axes[1].set_yticks([0, 30, 60, 90])
    axes[1].set_ylabel("相位（°）")
    axes[1].set_xlabel("周期（秒）")
    path = output_dir / f"{mode}_{_station_label(position_m)}测点曲线.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_scenario_results(
    base: dict,
    scenario: dict,
    responses: dict,
    frequencies_hz: np.ndarray,
    station_x_m: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    """为每个场景完整输出模型图、TE/TM 拟断面及指定测点曲线。"""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [plot_model_layout(base, scenario, output_dir)]
    for mode, data in responses.items():
        paths.append(plot_pseudosections(mode, data, frequencies_hz, station_x_m, output_dir))
        for position in scenario["curve_station_x_m"]:
            paths.append(plot_station_curve(mode, data, frequencies_hz, station_x_m, position, output_dir))
    return paths
