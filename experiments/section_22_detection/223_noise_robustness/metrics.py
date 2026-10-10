"""固定频率乘测点真值口径，汇总噪声实验的逐次评价。"""

from __future__ import annotations

import numpy as np


def trial_metrics(result, target_indices: list[int], has_static: bool, tolerance_m: float) -> dict:
    """保留严格点位指标与预先规定的 ±500 m 区域诊断。"""

    votes = np.asarray(result.per_frequency_votes, dtype=bool)
    stations = np.asarray(result.station_x_m, dtype=float)
    truth = np.zeros_like(votes)
    if has_static:
        truth[:, target_indices] = True
    tp = int(np.count_nonzero(votes & truth))
    fp = int(np.count_nonzero(votes & ~truth))
    fn = int(np.count_nonzero(~votes & truth))
    negative_count = int(np.count_nonzero(~truth))
    target_x = stations[target_indices]
    regions = [np.abs(stations - value) <= tolerance_m + 1e-9 for value in target_x]
    union = np.logical_or.reduce(regions)
    region_hits = [int(np.count_nonzero(np.any(votes[:, region], axis=1))) for region in regions]
    outside = int(np.count_nonzero(votes[:, ~union]))
    neighbor = int(np.count_nonzero(votes[:, union] & ~truth[:, union]))
    target_counts = [int(np.count_nonzero(votes[:, index])) for index in target_indices]
    if has_static and tp != sum(target_counts):
        raise AssertionError("真值测点频数与 TP 不一致")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": negative_count - fp,
        "precision": (tp / (tp + fp) if tp + fp else 0.0) if has_static else None,
        "recall": tp / (tp + fn) if has_static and tp + fn else None,
        "f1": (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0) if has_static else None,
        "false_positive_rate": fp / negative_count,
        "target_frequency_counts": target_counts if has_static else None,
        "both_targets_detected_any_frequency": bool(all(count > 0 for count in target_counts)) if has_static else None,
        "region_frequency_counts": region_hits if has_static else None,
        "neighbor_false_candidates": neighbor if has_static else None,
        "outside_region_candidates": outside,
        "total_frequency_station_candidates": int(np.count_nonzero(votes)),
    }


def candidate_alpha_min(result) -> np.ndarray:
    """逐频逐测点记录候选传播线的最小相邻尺度指数。"""

    values = np.full(result.per_frequency_votes.shape, np.nan)
    for frequency_index, diagnostics in enumerate(result.frequency_diagnostics):
        for line in diagnostics.maxima_lines:
            if not line.static_candidate or line.mapped_station_index is None:
                continue
            if line.alpha_pairs is None or len(line.alpha_pairs) != len(result.config.scales) - 1:
                raise AssertionError("候选传播线的相邻尺度 alpha 数量不完整")
            index = line.mapped_station_index
            current = values[frequency_index, index]
            minimum = float(min(line.alpha_pairs))
            values[frequency_index, index] = minimum if np.isnan(current) else min(current, minimum)
    if not np.array_equal(np.isfinite(values), result.per_frequency_votes):
        raise AssertionError("alpha 诊断与候选矩阵不一致")
    return values


def distribution(values: list[float | int]) -> dict:
    """返回重复试验指标的均值和四分位数。"""

    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("指标分布必须包含有限数值")
    return {
        "mean": float(np.mean(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def wilson_interval(successes: int, total: int) -> list[float]:
    """给宽松的双目标至少一次命中比例计算 95% Wilson 区间。"""

    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("二项计数无效")
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half_width = z * np.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return [float(center - half_width), float(center + half_width)]
