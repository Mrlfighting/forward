"""基础静态测点识别的逐频计算和诊断汇总。"""

from __future__ import annotations

import numpy as np

from .classification import is_static_candidate, map_candidate_to_station
from .lipschitz import adjacent_alpha_pairs, summarize_alpha
from .maxima import extract_modulus_maxima, select_modulus_maxima
from .preprocessing import spatial_profile, validate_and_sort_inputs
from .tracking import track_maxima_lines
from .types import DetectionConfig, DetectionResult, FrequencyDiagnostics
from .wavelet import continuous_wavelet_transform


def detect_static_sites(
    stations: np.ndarray,
    frequencies: np.ndarray,
    rho_a: np.ndarray,
    config: DetectionConfig,
    station_ids: list[str] | None = None,
) -> DetectionResult:
    """识别空间剖面的负指数传播线，并保存完整逐频诊断。"""

    sorted_x, freq, sorted_rho, sorted_ids = validate_and_sort_inputs(
        stations, frequencies, rho_a, station_ids
    )
    scales = np.asarray(config.scales, dtype=int)
    if scales.size < 2 or np.any(scales <= 0) or not np.array_equal(scales[1:], 2 * scales[:-1]):
        raise ValueError("尺度须为至少两个连续的二进尺度")
    if config.mapping_max_distance_m is not None and config.mapping_max_distance_m < 0:
        raise ValueError("测点映射距离不得为负")
    if config.transform_type != "cwt":
        raise ValueError("当前基础识别仅实现 cwt；DWT 的系数位置规则未被当前实验定义")
    if config.candidate_rule not in {"any_pair_negative", "all_pairs_negative", "summary_negative"}:
        raise ValueError("未知传播线候选规则")
    if config.candidate_rule == "summary_negative" and config.alpha_summary_method == "pairwise":
        raise ValueError("汇总指数判据需要非 pairwise 的 alpha 汇总方法")

    votes = np.zeros((freq.size, sorted_x.size), dtype=bool)
    station_alphas: list[list[float]] = [[] for _ in sorted_x]
    diagnostics: list[FrequencyDiagnostics] = []
    for frequency_index, frequency in enumerate(freq):
        signal_at_stations, sample_x, profile = spatial_profile(
            sorted_x,
            sorted_rho[frequency_index],
            config.interpolation_points,
            config.signal_transform,
            config.interpolation_kind,
        )
        coefficients = continuous_wavelet_transform(
            profile, scales, config.wavelet_name, config.boundary_mode
        )
        maxima = extract_modulus_maxima(coefficients)
        selected = select_modulus_maxima(coefficients, maxima, config.modulus_quantile)
        lines = track_maxima_lines(
            coefficients, selected, sample_x, config.min_line_length
        )
        candidates: list[float] = []
        candidate_stations: list[int] = []
        frequency_alphas: list[list[float]] = [[] for _ in sorted_x]
        for line in lines:
            if not line.accepted:
                continue
            line.alpha_pairs = adjacent_alpha_pairs(line)
            line.alpha = summarize_alpha(line, scales, config.alpha_summary_method)
            if not is_static_candidate(
                line.alpha_pairs, line.alpha, config.candidate_rule, config.alpha_threshold
            ):
                continue
            line.static_candidate = True
            smallest = min(line.nodes, key=lambda node: node.scale_index)
            candidate_x = smallest.x_m
            nearest, distance = map_candidate_to_station(candidate_x, sorted_x)
            line.candidate_x_m = candidate_x
            line.mapping_distance_m = float(distance)
            candidates.append(candidate_x)
            if config.mapping_max_distance_m is not None and distance > config.mapping_max_distance_m:
                line.rejection_reason = "outside_station_mapping_distance"
                continue
            line.mapped_station_index = nearest
            candidate_stations.append(nearest)
            if line.alpha is not None:
                frequency_alphas[nearest].append(line.alpha)
            votes[frequency_index, nearest] = True
        for station_index, values in enumerate(frequency_alphas):
            if values:
                station_alphas[station_index].append(float(np.median(values)))
        diagnostics.append(
            FrequencyDiagnostics(
                frequency_hz=float(frequency),
                station_x_m=sorted_x.copy(),
                station_ids=sorted_ids.copy(),
                signal_at_stations=signal_at_stations,
                sample_x_m=sample_x,
                signal_profile=profile,
                coefficients=coefficients,
                modulus_maxima=maxima,
                selected_maxima=selected,
                maxima_lines=lines,
                candidate_x_m=candidates,
                candidate_station_indices=candidate_stations,
            )
        )
    vote_counts = votes.sum(axis=0)
    median_alpha = np.array(
        [np.median(values) if values else np.nan for values in station_alphas], dtype=float
    )
    return DetectionResult(
        station_x_m=sorted_x,
        station_ids=sorted_ids,
        frequencies_hz=freq,
        vote_counts=vote_counts,
        per_frequency_votes=votes,
        median_alpha=median_alpha,
        frequency_diagnostics=diagnostics,
        config=config,
    )
