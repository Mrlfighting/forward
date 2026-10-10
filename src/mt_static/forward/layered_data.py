"""复杂层状模型的共享噪声与静态数据构造。"""

from __future__ import annotations

import numpy as np

def _join_data(rho: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """按频率、数据类型和测点顺序拼接合成观测。"""

    if rho.shape != phase.shape or rho.ndim != 2:
        raise ValueError("视电阻率和相位应具有相同的二维形状")
    return np.stack((rho, phase), axis=1).reshape(-1)


def split_full_data(vector: np.ndarray, n_frequency: int, n_station: int) -> tuple[np.ndarray, np.ndarray]:
    """按频率、视电阻率/相位和测点拆分旧实验数据顺序。"""

    blocks = np.asarray(vector).reshape(n_frequency, 2, n_station)
    return blocks[:, 0, :], blocks[:, 1, :]


def build_layered_observations(
    clean_prediction: np.ndarray,
    physical_static_prediction: np.ndarray,
    n_frequency: int,
    n_station: int,
    relative_rho_noise: float,
    rho_error_floor: float,
    phase_error_floor_deg: float,
    random_seed: int,
) -> dict[str, np.ndarray | float]:
    """沿用旧图：静态数据取物理正演的 rho_a、无浅部体相位和共享随机噪声。"""

    clean_rho, clean_phase = split_full_data(clean_prediction, n_frequency, n_station)
    static_rho, _ = split_full_data(physical_static_prediction, n_frequency, n_station)
    rng = np.random.default_rng(random_seed)
    rho_normal = rng.normal(size=clean_rho.shape)
    phase_normal = rng.normal(size=clean_phase.shape)
    multiplier = np.clip(1.0 + relative_rho_noise * rho_normal, 0.05, None)
    phase_noise = phase_error_floor_deg * (relative_rho_noise / 0.1) * phase_normal
    clean = _join_data(clean_rho * multiplier, clean_phase + phase_noise)
    static = _join_data(static_rho * multiplier, clean_phase + phase_noise)
    rho_relative_std = float(np.hypot(rho_error_floor, relative_rho_noise))
    phase_std = float(np.hypot(
        phase_error_floor_deg,
        phase_error_floor_deg * relative_rho_noise / 0.1,
    ))
    clean_std = _join_data(rho_relative_std * np.abs(clean_rho * multiplier), np.full_like(clean_phase, phase_std))
    static_std = _join_data(rho_relative_std * np.abs(static_rho * multiplier), np.full_like(clean_phase, phase_std))
    if np.any(clean_rho <= 0) or np.any(static_rho <= 0):
        raise ValueError("视电阻率必须为正")
    if not np.array_equal(
        split_full_data(clean, n_frequency, n_station)[1],
        split_full_data(static, n_frequency, n_station)[1],
    ):
        raise AssertionError("旧图要求静态数据相位与无浅部体数据相同")
    return {
        "clean": clean,
        "static": static,
        "clean_std": clean_std,
        "static_std": static_std,
        "clean_noiseless": clean_prediction,
        "physical_static_noiseless": physical_static_prediction,
        "rho_relative_std": rho_relative_std,
        "phase_std_deg": phase_std,
    }
