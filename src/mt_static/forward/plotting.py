"""绘制论文 2.1 的同色标拟断面与测点曲线。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

MODE_LABELS = {"TE": "TE", "TM": "TM"}
MODEL_LABELS = {"with_shallow": "有浅部异常体", "without_shallow": "无浅部异常体"}
STATION_LABELS = {-5000.0: "负五千米", 5000.0: "正五千米"}


def plot_pseudosections(
    mode: str,
    mode_responses: dict,
    frequencies_hz: np.ndarray,
    station_x_m: np.ndarray,
    output_dir: Path,
) -> Path:
    """按示例的红低蓝高色带绘制两种模型的拟断面对比。"""
    all_resistivity = np.concatenate(
        [item["apparent_resistivity"].ravel() for item in mode_responses.values()]
    )
    all_phase = np.concatenate([item["phase_deg"].ravel() for item in mode_responses.values()])
    resistivity_levels = np.geomspace(float(all_resistivity.min()), float(all_resistivity.max()), 64)
    phase_levels = np.linspace(float(all_phase.min()), float(all_phase.max()), 64)
    log_frequency = np.log10(frequencies_hz)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    for row, model_name in enumerate(("with_shallow", "without_shallow")):
        data = mode_responses[model_name]
        for column, field in enumerate(("apparent_resistivity", "phase_deg")):
            ax = axes[row, column]
            levels = resistivity_levels if column == 0 else phase_levels
            normalization = (
                LogNorm(vmin=resistivity_levels[0], vmax=resistivity_levels[-1])
                if column == 0 else None
            )
            contour = ax.contourf(
                station_x_m / 1000.0,
                log_frequency,
                data[field],
                levels=levels,
                cmap="jet_r",
                norm=normalization,
                extend="both",
            )
            ax.set_xlim(-10.0, 10.0)
            ax.set_ylim(float(log_frequency.min()), float(log_frequency.max()))
            ax.set_xlabel("测点位置（千米）")
            ax.set_ylabel("频率的常用对数")
            field_label = "视电阻率" if column == 0 else "相位"
            ax.set_title(f"{MODE_LABELS[mode]}｜{MODEL_LABELS[model_name]}｜{field_label}")
            fig.colorbar(contour, ax=ax, label="视电阻率（Ω·m）" if column == 0 else "相位（°）")

    output_path = output_dir / f"{MODE_LABELS[mode]}_视电阻率与相位拟断面.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_station_curves(
    mode: str,
    mode_responses: dict,
    frequencies_hz: np.ndarray,
    station_x_m: np.ndarray,
    position_m: float,
    output_dir: Path,
) -> Path:
    """按示例绘制对数周期、对数视电阻率和固定范围相位曲线。"""
    station_index = int(np.flatnonzero(station_x_m == position_m)[0])
    period_s = 1.0 / frequencies_hz
    order = np.argsort(period_s)
    period_s = period_s[order]
    baseline = mode_responses["without_shallow"]
    affected = mode_responses["with_shallow"]
    baseline_rho = baseline["apparent_resistivity"][order, station_index]
    affected_rho = affected["apparent_resistivity"][order, station_index]
    baseline_phase = baseline["phase_deg"][order, station_index]
    affected_phase = affected["phase_deg"][order, station_index]
    rho_lower = 10.0 ** np.floor(np.log10(min(baseline_rho.min(), affected_rho.min())))
    rho_upper = max(1.0e4, 10.0 ** np.ceil(np.log10(max(baseline_rho.max(), affected_rho.max()))))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.4, 7.2), constrained_layout=True)
    for ax, clean_values, shallow_values, marker in (
        (axes[0], baseline_rho, affected_rho, "o"),
        (axes[1], baseline_phase, affected_phase, "s"),
    ):
        ax.plot(period_s, clean_values, color="#1f77b4", linestyle="--", linewidth=1.4)
        ax.scatter(period_s, clean_values, marker=marker, s=18, facecolors="white",
                   edgecolors="#222222", linewidths=0.8, zorder=3, label="无浅部异常体")
        ax.plot(period_s, shallow_values, color="#d62728", linewidth=1.7,
                label="有浅部异常体")
        ax.set_xscale("log")
        ax.set_xlim(1.0e-3, 1.0e3)
        ax.grid(which="major", color="#a0a0a0", alpha=0.50, linewidth=0.6)
        ax.grid(which="minor", color="#c8c8c8", alpha=0.36, linewidth=0.4)
    axes[0].set_yscale("log")
    axes[0].set_ylim(rho_lower, rho_upper)
    axes[0].set_ylabel("视电阻率（Ω·m）")
    axes[0].set_title(f"{MODE_LABELS[mode]}｜{STATION_LABELS[position_m]}测点")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].set_ylim(0.0, 90.0)
    axes[1].set_yticks([0, 30, 60, 90])
    axes[1].set_ylabel("相位（°）")
    axes[1].set_xlabel("周期（秒）")

    output_path = output_dir / f"{MODE_LABELS[mode]}_{STATION_LABELS[position_m]}测点曲线.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_results(
    responses: dict,
    output_dir: Path,
    frequencies_hz: np.ndarray,
    station_x_m: np.ndarray,
) -> list[Path]:
    """为两种极化生成拟断面与单测点曲线，返回全部图文件路径。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for mode, mode_responses in responses.items():
        output_paths.append(
            plot_pseudosections(mode, mode_responses, frequencies_hz, station_x_m, output_dir)
        )
        for position_m in (-5000.0, 5000.0):
            output_paths.append(
                plot_station_curves(
                    mode, mode_responses, frequencies_hz, station_x_m, position_m, output_dir
                )
            )
    return output_paths
