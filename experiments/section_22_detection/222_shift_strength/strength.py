"""静态位移强度实验的数据注入与逐频评价，不实现识别算法。"""

from __future__ import annotations

import numpy as np


def apply_static_factor(
    baseline_rho: np.ndarray, target_indices: list[int], factor: float
) -> np.ndarray:
    """仅对指定测点的全部频率施加同一视电阻率倍率。"""

    baseline = np.asarray(baseline_rho, dtype=float)
    if baseline.ndim != 2 or not np.all(np.isfinite(baseline)) or np.any(baseline <= 0):
        raise ValueError("基础视电阻率必须是有限正值的二维数组")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("静态位移倍率必须是有限正数")
    if not target_indices or len(set(target_indices)) != len(target_indices):
        raise ValueError("目标测点索引不得为空或重复")
    if any(index < 0 or index >= baseline.shape[1] for index in target_indices):
        raise ValueError("目标测点索引超出范围")
    shifted = baseline.copy()
    shifted[:, target_indices] *= factor
    if not np.all(np.isfinite(shifted)) or np.any(shifted <= 0):
        raise ValueError("施加倍率后的视电阻率无效")
    untouched = np.ones(baseline.shape[1], dtype=bool)
    untouched[target_indices] = False
    if not np.array_equal(shifted[:, untouched], baseline[:, untouched]):
        raise AssertionError("非目标测点被改写")
    if not np.allclose(shifted[:, target_indices] / baseline[:, target_indices], factor, rtol=1e-14, atol=0):
        raise AssertionError("目标测点的实际倍率不一致")
    return shifted


def frequency_station_metrics(
    predicted: np.ndarray, target_indices: list[int], is_control: bool
) -> dict:
    """以频率乘测点为单位计算指标；阴性对照的召回率和 F1 留空。"""

    votes = np.asarray(predicted, dtype=bool)
    if votes.ndim != 2:
        raise ValueError("逐频候选须为二维布尔矩阵")
    truth = np.zeros_like(votes)
    if not is_control:
        truth[:, target_indices] = True
    tp = int(np.count_nonzero(votes & truth))
    fp = int(np.count_nonzero(votes & ~truth))
    fn = int(np.count_nonzero(~votes & truth))
    if is_control:
        return {"tp": tp, "fp": fp, "fn": fn, "precision": None, "recall": None, "f1": None}
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def candidate_alpha_min(result) -> np.ndarray:
    """逐频逐测点汇总候选传播线的最小相邻尺度指数。"""

    values = np.full(result.per_frequency_votes.shape, np.nan)
    for frequency_index, diagnostics in enumerate(result.frequency_diagnostics):
        for line in diagnostics.maxima_lines:
            if not line.static_candidate or line.mapped_station_index is None:
                continue
            if line.alpha_pairs is None or len(line.alpha_pairs) != len(result.config.scales) - 1:
                raise AssertionError("候选传播线缺少完整相邻尺度 alpha")
            index = line.mapped_station_index
            minimum = float(min(line.alpha_pairs))
            previous = values[frequency_index, index]
            values[frequency_index, index] = minimum if np.isnan(previous) else min(previous, minimum)
    if not np.array_equal(np.isfinite(values), result.per_frequency_votes):
        raise AssertionError("alpha 诊断与逐频候选矩阵不一致")
    return values


def strength_record(factor: float, result, target_indices: list[int]) -> tuple[dict, np.ndarray]:
    """汇总单一倍率的逐频指标与两目标命中次数。"""

    is_control = factor == 1.0
    alpha_min = candidate_alpha_min(result)
    counts = [int(result.vote_counts[index]) for index in target_indices]
    metrics = frequency_station_metrics(result.per_frequency_votes, target_indices, is_control)
    if not is_control and metrics["tp"] != sum(counts):
        raise AssertionError("目标命中次数与 TP 不一致")
    alpha_at_targets = []
    for index in target_indices:
        finite = alpha_min[:, index][np.isfinite(alpha_min[:, index])]
        alpha_at_targets.append(float(np.min(finite)) if finite.size else None)
    return {
        "cs": float(factor),
        "delta_s": float(abs(np.log10(factor))),
        "is_negative_control": is_control,
        "target_frequency_counts": counts,
        "target_detected_any_frequency": [count > 0 for count in counts] if not is_control else [False] * len(counts),
        "all_targets_detected_any_frequency": bool(all(count > 0 for count in counts)) if not is_control else None,
        "alpha_min_at_targets": alpha_at_targets,
        "metrics": metrics,
        "total_frequency_station_candidates": int(np.count_nonzero(result.per_frequency_votes)),
    }, alpha_min


def detection_boundary(records: list[dict], target_index: int | None) -> dict:
    """仅在同方向离散测试点呈单调检出时给出检测边界区间。"""

    ordered = sorted(records, key=lambda row: row["delta_s"])
    detected = [
        row["all_targets_detected_any_frequency"] if target_index is None
        else row["target_detected_any_frequency"][target_index]
        for row in ordered
    ]
    if any(earlier and not later for earlier, later in zip(detected, detected[1:])):
        status = "non_monotonic"
        bracket = None
    elif all(detected):
        status = "detected_at_all_tested_strengths"
        bracket = None
    elif not any(detected):
        status = "not_detected_at_tested_strengths"
        bracket = None
    else:
        status = "bounded_by_tested_strengths"
        bracket = {
            "largest_undetected_delta_s": max(row["delta_s"] for row, hit in zip(ordered, detected) if not hit),
            "smallest_detected_delta_s": min(row["delta_s"] for row, hit in zip(ordered, detected) if hit),
        }
    return {
        "status": status,
        "delta_s_interval": bracket,
        "tested": [{"cs": row["cs"], "delta_s": row["delta_s"], "detected": hit}
                   for row, hit in zip(ordered, detected)],
    }
