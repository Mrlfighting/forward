"""负指数传播线候选判别与原测点回映射。"""

from __future__ import annotations

import numpy as np


def is_static_candidate(
    alpha_pairs: list[float],
    alpha_summary: float | None,
    rule: str,
    threshold: float,
) -> bool:
    """按显式配置判断负指数候选，不宣称候选等于静态真值。"""

    if rule == "any_pair_negative":
        return any(value < threshold for value in alpha_pairs)
    if rule == "all_pairs_negative":
        return bool(alpha_pairs) and all(value < threshold for value in alpha_pairs)
    if rule == "summary_negative":
        return alpha_summary is not None and alpha_summary < threshold
    raise ValueError("未知传播线候选规则")


def map_candidate_to_station(candidate_x_m: float, stations_m: np.ndarray) -> tuple[int, float]:
    """映射到最近的原始测点，同时返回映射距离供审计。"""

    index = int(np.argmin(np.abs(stations_m - candidate_x_m)))
    return index, float(abs(stations_m[index] - candidate_x_m))
