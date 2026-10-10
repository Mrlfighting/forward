"""逐尺度提取和筛选小波系数的模极大值。"""

from __future__ import annotations

import numpy as np


def extract_modulus_maxima(coefficients: np.ndarray) -> np.ndarray:
    """沿用旧代码的双侧大于等于规则，端点不参与判定。"""

    amplitude = np.abs(np.asarray(coefficients, dtype=float))
    maxima = np.zeros_like(amplitude, dtype=bool)
    maxima[:, 1:-1] = (
        (amplitude[:, 1:-1] >= amplitude[:, :-2])
        & (amplitude[:, 1:-1] >= amplitude[:, 2:])
        & (amplitude[:, 1:-1] > 0.0)
    )
    return maxima


def select_modulus_maxima(
    coefficients: np.ndarray, maxima: np.ndarray, quantile: float | None
) -> np.ndarray:
    """按实验配置筛选极大值；空阈值表示全部非零极大值。"""

    if quantile is None:
        return maxima.copy()
    if not 0.0 <= quantile < 1.0:
        raise ValueError("模值分位数须在 [0, 1) 内")
    amplitude = np.abs(coefficients)
    thresholds = np.quantile(amplitude, quantile, axis=1)
    return maxima & (amplitude >= thresholds[:, None])
