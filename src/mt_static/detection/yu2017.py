"""于生宝等（2017）第二章的逐尺度静态效应候选识别。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classification import map_candidate_to_station
from .lipschitz import adjacent_alpha_pairs
from .maxima import extract_modulus_maxima
from .preprocessing import spatial_profile, validate_and_sort_inputs
from .tracking import track_maxima_lines
from .types import FrequencyDiagnostics
from .wavelet import continuous_wavelet_transform


@dataclass(frozen=True)
class Yu2017Config:
    """显式记录原文未唯一规定的数值选择。"""

    scales: tuple[int, ...]
    interpolation_points: int
    interpolation_kind: str
    wavelet_name: str
    boundary_mode: str
    signal_transform: str = "raw"


@dataclass(frozen=True)
class Yu2017NegativePair:
    """一条完整传播线中一对相邻尺度的负指数证据。"""

    frequency_hz: float
    line_index: int
    smaller_scale: int
    larger_scale: int
    alpha: float
    candidate_x_m: float
    nearest_station_index: int
    nearest_station_x_m: float
    mapping_distance_m: float


@dataclass
class Yu2017Result:
    """保留逐频、逐尺度证据，不生成跨频最终测点掩码。"""

    station_x_m: np.ndarray
    station_ids: list[str]
    frequencies_hz: np.ndarray
    diagnostics: list[FrequencyDiagnostics]
    negative_pairs: list[list[Yu2017NegativePair]]
    config: Yu2017Config


def detect_yu2017(
    stations: np.ndarray,
    frequencies: np.ndarray,
    rho_a: np.ndarray,
    config: Yu2017Config,
    station_ids: list[str] | None = None,
) -> Yu2017Result:
    """依原文逐对计算 α，负值仅作为该尺度对的候选证据。"""

    station_x, freq, sorted_rho, sorted_ids = validate_and_sort_inputs(
        stations, frequencies, rho_a, station_ids
    )
    scales = np.asarray(config.scales, dtype=int)
    if scales.size < 2 or np.any(scales <= 0) or not np.array_equal(scales[1:], 2 * scales[:-1]):
        raise ValueError("尺度须为连续二进尺度")
    if config.signal_transform != "raw":
        raise ValueError("于生宝原文模式使用原始视电阻率；对数输入须另列为扩展实验")
    if config.wavelet_name != "gaussian_derivative":
        raise ValueError("当前明确选定的一阶高斯导数小波尚无其他母小波实现")

    diagnostics: list[FrequencyDiagnostics] = []
    all_negative_pairs: list[list[Yu2017NegativePair]] = []
    for frequency_index, frequency in enumerate(freq):
        signal, sample_x, profile = spatial_profile(
            station_x,
            sorted_rho[frequency_index],
            config.interpolation_points,
            config.signal_transform,
            config.interpolation_kind,
        )
        coefficients = continuous_wavelet_transform(
            profile, scales, config.wavelet_name, config.boundary_mode
        )
        maxima = extract_modulus_maxima(coefficients)
        # 原文识别部分没有给出幅值阈值；保留全部非零局部模极大值。
        lines = track_maxima_lines(coefficients, maxima, sample_x, min_line_length=scales.size)
        events: list[Yu2017NegativePair] = []
        for line_index, line in enumerate(lines):
            if not line.accepted:
                continue
            line.alpha_pairs = adjacent_alpha_pairs(line)
            smallest = min(line.nodes, key=lambda node: node.scale_index)
            nearest, distance = map_candidate_to_station(smallest.x_m, station_x)
            for pair_index, alpha in enumerate(line.alpha_pairs):
                if alpha >= 0.0:
                    continue
                events.append(
                    Yu2017NegativePair(
                        frequency_hz=float(frequency),
                        line_index=line_index,
                        smaller_scale=int(scales[pair_index]),
                        larger_scale=int(scales[pair_index + 1]),
                        alpha=float(alpha),
                        candidate_x_m=smallest.x_m,
                        nearest_station_index=nearest,
                        nearest_station_x_m=float(station_x[nearest]),
                        mapping_distance_m=distance,
                    )
                )
        diagnostics.append(
            FrequencyDiagnostics(
                frequency_hz=float(frequency),
                station_x_m=station_x.copy(),
                station_ids=sorted_ids.copy(),
                signal_at_stations=signal,
                sample_x_m=sample_x,
                signal_profile=profile,
                coefficients=coefficients,
                modulus_maxima=maxima,
                selected_maxima=maxima.copy(),
                maxima_lines=lines,
                candidate_x_m=[],
                candidate_station_indices=[],
            )
        )
        all_negative_pairs.append(events)
    return Yu2017Result(
        station_x_m=station_x,
        station_ids=sorted_ids,
        frequencies_hz=freq,
        diagnostics=diagnostics,
        negative_pairs=all_negative_pairs,
        config=config,
    )
