"""沿测线的视电阻率剖面准备与插值。"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def validate_and_sort_inputs(
    stations: np.ndarray,
    frequencies: np.ndarray,
    rho_a: np.ndarray,
    station_ids: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """校验输入并同步排序测点、视电阻率及编号。"""

    x = np.asarray(stations, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    rho = np.asarray(rho_a, dtype=float)
    if x.ndim != 1 or f.ndim != 1 or rho.shape != (f.size, x.size):
        raise ValueError("rho_a 的形状必须是 (频率数, 测点数)")
    if x.size < 3 or f.size < 1:
        raise ValueError("至少需要三个测点和一个频率")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(f)) or not np.all(np.isfinite(rho)):
        raise ValueError("输入包含非有限数值")
    if np.any(f <= 0) or np.any(rho <= 0):
        raise ValueError("频率和视电阻率必须为正")
    order = np.argsort(x)
    sorted_x = x[order]
    if np.any(np.diff(sorted_x) <= 0):
        raise ValueError("测点位置不得重复")
    if station_ids is None:
        station_ids = [f"S{index + 1:02d}" for index in range(x.size)]
    if len(station_ids) != x.size or len(set(station_ids)) != x.size:
        raise ValueError("测点编号须与位置一一对应且不得重复")
    sorted_ids = [str(station_ids[index]) for index in order]
    return sorted_x, f, rho[:, order], sorted_ids


def spatial_profile(
    stations: np.ndarray,
    rho_at_frequency: np.ndarray,
    n_samples: int,
    signal_transform: str,
    interpolation_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按配置选空间信号并插值，不在测线范围外求值。"""

    if n_samples < stations.size:
        raise ValueError("插值采样点数不得小于测点数")
    if signal_transform == "raw":
        signal = np.asarray(rho_at_frequency, dtype=float)
    elif signal_transform == "log":
        signal = np.log(rho_at_frequency)
    else:
        raise ValueError("signal_transform 只支持 raw 或 log")
    sample_x = np.linspace(stations[0], stations[-1], n_samples)
    if interpolation_kind == "linear":
        profile = np.interp(sample_x, stations, signal)
    elif interpolation_kind == "cubic":
        profile = CubicSpline(stations, signal, extrapolate=False)(sample_x)
    else:
        raise ValueError("interpolation_kind 只支持 linear 或 cubic")
    return signal, sample_x, profile
