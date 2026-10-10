"""构造 TM 反演接收器及与之严格一致的数据向量。"""

from __future__ import annotations

import numpy as np
from simpeg.electromagnetics import natural_source as nsem


def _locations(stations: np.ndarray) -> np.ndarray:
    """将一维测线坐标扩充为地表二维接收器坐标。"""

    positions = np.asarray(stations, dtype=float)
    return np.c_[positions, np.zeros(positions.size)]


def build_full_survey(frequencies: np.ndarray, stations: np.ndarray) -> nsem.survey.Survey:
    """每个频率先排列所有测点的视电阻率，再排列所有相位。"""

    locations = _locations(stations)
    sources = []
    for frequency in frequencies:
        rho = nsem.receivers.Impedance(locations, orientation="xy", component="apparent_resistivity")
        phase = nsem.receivers.Impedance(locations, orientation="xy", component="phase")
        sources.append(nsem.sources.Planewave([rho, phase], float(frequency)))
    return nsem.survey.Survey(sources)


def build_reduced_survey(
    frequencies: np.ndarray, stations: np.ndarray, static_indices: np.ndarray
) -> tuple[nsem.survey.Survey, np.ndarray]:
    """普通点保留视电阻率和相位，静态点仅保留相位。"""

    normal = np.setdiff1d(np.arange(len(stations)), np.asarray(static_indices, dtype=int))
    normal_locations = _locations(np.asarray(stations)[normal])
    all_locations = _locations(stations)
    sources = []
    for frequency in frequencies:
        rho = nsem.receivers.Impedance(normal_locations, orientation="xy", component="apparent_resistivity")
        phase = nsem.receivers.Impedance(all_locations, orientation="xy", component="phase")
        sources.append(nsem.sources.Planewave([rho, phase], float(frequency)))
    return nsem.survey.Survey(sources), normal


def build_shape_survey(
    frequencies: np.ndarray, stations: np.ndarray, static_indices: np.ndarray
) -> nsem.survey.Survey:
    """建立与静态点每条曲线 n_f−1 个形态分量等量的占位接收器。"""

    locations = _locations(np.asarray(stations)[np.asarray(static_indices, dtype=int)])
    sources = []
    for frequency in np.asarray(frequencies)[:-1]:
        rho = nsem.receivers.Impedance(locations, orientation="xy", component="apparent_resistivity")
        sources.append(nsem.sources.Planewave([rho], float(frequency)))
    return nsem.survey.Survey(sources)


def join_data(rho: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """按照频率、数据类型、测点顺序拼接二维观测。"""

    if rho.shape != phase.shape or rho.ndim != 2:
        raise ValueError("视电阻率和相位应具有相同的二维形状")
    return np.stack((rho, phase), axis=1).reshape(-1)


def reduce_data(full: np.ndarray, n_frequency: int, n_station: int, normal: np.ndarray) -> np.ndarray:
    """从完整数据中去掉静态点的绝对视电阻率，保留所有相位。"""

    blocks = np.asarray(full).reshape(n_frequency, 2, n_station)
    return np.concatenate((blocks[:, 0, normal], blocks[:, 1, :]), axis=1).ravel()
