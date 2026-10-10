"""将阻抗实虚部转换为视电阻率、相位与差异指标。"""

import numpy as np

from .settings import FREQUENCIES_HZ, STATION_X_M, VACUUM_PERMEABILITY


def convert_response(dpred: np.ndarray) -> dict[str, np.ndarray]:
    """按频率、分量、测点三维顺序解析 SimPEG 数据。"""
    expected_size = 2 * FREQUENCIES_HZ.size * STATION_X_M.size
    if dpred.size != expected_size:
        raise ValueError(f"阻抗数据量应为 {expected_size}，实际为 {dpred.size}")
    blocks = np.asarray(dpred).reshape(FREQUENCIES_HZ.size, 2, STATION_X_M.size)
    raw_impedance = blocks[:, 0, :] + 1j * blocks[:, 1, :]
    # 保存原始阻抗，并沿用原绘图脚本的阻抗取负号展示约定。
    display_impedance = -raw_impedance
    angular_frequency = 2.0 * np.pi * FREQUENCIES_HZ[:, None]
    apparent_resistivity = np.abs(display_impedance) ** 2 / (VACUUM_PERMEABILITY * angular_frequency)
    phase_deg = np.mod(np.angle(display_impedance, deg=True), 180.0)
    if not np.all(np.isfinite(apparent_resistivity)) or np.any(apparent_resistivity <= 0):
        raise ValueError("视电阻率包含非有限值或非正值")
    if not np.all(np.isfinite(phase_deg)):
        raise ValueError("相位包含非有限值")
    return {
        "raw_impedance": raw_impedance,
        "display_impedance": display_impedance,
        "apparent_resistivity": apparent_resistivity,
        "phase_deg": phase_deg,
    }


def summarize_differences(responses: dict[str, dict[str, dict[str, np.ndarray]]]) -> dict:
    """报告有无浅部体的可复核差异，不预设结论阈值。"""
    summary = {}
    for mode, mode_responses in responses.items():
        baseline = mode_responses["without_shallow"]
        affected = mode_responses["with_shallow"]
        log_rho_ratio = np.log10(affected["apparent_resistivity"] / baseline["apparent_resistivity"])
        phase_difference = (affected["phase_deg"] - baseline["phase_deg"] + 90.0) % 180.0 - 90.0
        station_metrics = {}
        for position in (-5000.0, 5000.0):
            station_index = int(np.flatnonzero(STATION_X_M == position)[0])
            station_metrics[str(int(position))] = {
                "mean_log10_rho_ratio": float(np.mean(log_rho_ratio[:, station_index])),
                "std_log10_rho_ratio": float(np.std(log_rho_ratio[:, station_index])),
                "geometric_mean_rho_ratio": float(10.0 ** np.mean(log_rho_ratio[:, station_index])),
                "median_absolute_phase_difference_deg": float(np.median(np.abs(phase_difference[:, station_index]))),
                "max_absolute_phase_difference_deg": float(np.max(np.abs(phase_difference[:, station_index]))),
            }
        summary[mode] = {
            "median_absolute_log10_rho_ratio_all": float(np.median(np.abs(log_rho_ratio))),
            "median_absolute_phase_difference_all_deg": float(np.median(np.abs(phase_difference))),
            "shallow_body_stations": station_metrics,
        }
    return summary
