"""依据相邻尺度的空间位置和符号追踪模极大值。"""

from __future__ import annotations

import numpy as np

from .types import MaximaLine, MaximaNode


def _node(row: int, index: int, sample_x: np.ndarray, coefficients: np.ndarray) -> MaximaNode:
    return MaximaNode(row, index, float(sample_x[index]), float(coefficients[row, index]))


def track_maxima_lines(
    coefficients: np.ndarray,
    selected_maxima: np.ndarray,
    sample_x: np.ndarray,
    min_line_length: int | None = None,
) -> list[MaximaLine]:
    """按原文搜索区间、同位置优先和式（9）模值差追踪。"""

    n_scales, n_samples = coefficients.shape
    required_length = n_scales if min_line_length is None else min_line_length
    if not 2 <= required_length <= n_scales:
        raise ValueError("最短传播线长度须介于 2 和总尺度数之间")
    coarse_indices = np.flatnonzero(selected_maxima[-1])
    lines = [MaximaLine(nodes=[_node(n_scales - 1, int(i), sample_x, coefficients)]) for i in coarse_indices]
    for row in range(n_scales - 2, -1, -1):
        active = [line for line in lines if line.rejection_reason is None]
        active.sort(key=lambda line: line.nodes[-1].sample_index)
        coarse_positions = [line.nodes[-1].sample_index for line in active]
        mapped_indices: dict[int, int] = {}
        for position, line in enumerate(active):
            current = line.nodes[-1]
            left_coarse = coarse_positions[position - 1] if position > 0 else 0
            right_coarse = coarse_positions[position + 1] if position + 1 < len(active) else n_samples - 1
            left_mapped = mapped_indices.get(position - 1)
            lower = max(left_coarse, left_mapped if left_mapped is not None else 0)
            upper = right_coarse
            if lower > upper:
                line.rejection_reason = "empty_search_interval"
                continue
            candidates = np.flatnonzero(selected_maxima[row, lower : upper + 1]) + lower
            same_sign = candidates[
                np.sign(coefficients[row, candidates]) == np.sign(current.coefficient)
            ]
            if same_sign.size == 0:
                line.rejection_reason = "no_same_sign_maximum"
                continue
            if current.sample_index in same_sign:
                chosen = current.sample_index
            else:
                modulus_difference = np.abs(
                    np.abs(coefficients[row, same_sign]) - abs(current.coefficient)
                )
                chosen = int(same_sign[np.argmin(modulus_difference)])
            line.nodes.append(_node(row, chosen, sample_x, coefficients))
            mapped_indices[position] = chosen
    for line in lines:
        if len(line.nodes) >= required_length:
            line.accepted = True
        elif line.rejection_reason is None:
            line.rejection_reason = "line_too_short"
    return lines
