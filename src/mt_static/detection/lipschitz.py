"""对有效传播线拟合 Lipschitz 指数。"""

from __future__ import annotations

import numpy as np

from .types import MaximaLine


def estimate_alpha(line: MaximaLine, scales: np.ndarray) -> float:
    """最小二乘拟合 ln|W| 对 ln(尺度) 的直线斜率。"""

    if not line.accepted:
        raise ValueError("只能对完整有效传播线估计 alpha")
    ordered = sorted(line.nodes, key=lambda node: node.scale_index)
    x = np.log(np.asarray([scales[node.scale_index] for node in ordered], dtype=float))
    amplitude = np.asarray([abs(node.coefficient) for node in ordered], dtype=float)
    if np.any(amplitude <= 0):
        raise ValueError("传播线的模值必须为正")
    y = np.log(amplitude)
    centered_x = x - np.mean(x)
    return float(np.dot(centered_x, y - np.mean(y)) / np.dot(centered_x, centered_x))


def adjacent_alpha_pairs(line: MaximaLine) -> list[float]:
    """按小尺度到大尺度次序保存每对相邻二进尺度的局部指数。"""

    if not line.accepted:
        raise ValueError("只能对有效传播线计算 alpha")
    ordered = sorted(line.nodes, key=lambda node: node.scale_index)
    pairs = []
    for smaller, larger in zip(ordered[:-1], ordered[1:]):
        if smaller.modulus <= 0 or larger.modulus <= 0:
            raise ValueError("传播节点模值必须为正")
        pairs.append(float(np.log2(larger.modulus / smaller.modulus)))
    return pairs


def summarize_alpha(line: MaximaLine, scales: np.ndarray, method: str) -> float | None:
    """提供显式选择的扩展汇总；原文逐对模式不压缩为单个值。"""

    if line.alpha_pairs is None:
        raise ValueError("须先计算相邻尺度 alpha")
    if method == "pairwise":
        return None
    if method == "mean":
        return float(np.mean(line.alpha_pairs))
    if method == "median":
        return float(np.median(line.alpha_pairs))
    if method == "regression":
        return estimate_alpha(line, scales)
    raise ValueError("未知 alpha 汇总方法")
