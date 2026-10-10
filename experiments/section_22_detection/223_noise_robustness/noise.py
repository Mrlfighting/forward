"""对正值视电阻率施加可复现的逐频逐测点相对高斯噪声。"""

from __future__ import annotations

import numpy as np


def relative_gaussian_factor(
    standard_normal: np.ndarray, sigma: float, repair_rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """返回正值噪声倍率；仅重抽使倍率非正的单个样本。"""

    z = np.asarray(standard_normal, dtype=float).copy()
    if z.ndim != 2 or not np.all(np.isfinite(z)):
        raise ValueError("标准正态噪声必须是有限二维数组")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("相对噪声标准差必须非负且有限")
    if sigma == 0:
        return np.ones_like(z), 0
    redraw_count = 0
    invalid = 1.0 + sigma * z <= 0
    while np.any(invalid):
        count = int(np.count_nonzero(invalid))
        redraw_count += count
        z[invalid] = repair_rng.standard_normal(count)
        invalid = 1.0 + sigma * z <= 0
    factors = 1.0 + sigma * z
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
        raise AssertionError("噪声倍率未保持有限正值")
    return factors, redraw_count


def apply_noise(clean_rho: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """保持输入原值，仅对视电阻率逐项乘以噪声倍率。"""

    clean = np.asarray(clean_rho, dtype=float)
    multiplier = np.asarray(factors, dtype=float)
    if clean.shape != multiplier.shape or clean.ndim != 2:
        raise ValueError("视电阻率与噪声倍率应具有相同二维形状")
    if np.any(clean <= 0) or not np.all(np.isfinite(clean)):
        raise ValueError("原始视电阻率必须是有限正值")
    noisy = clean * multiplier
    if np.any(noisy <= 0) or not np.all(np.isfinite(noisy)):
        raise ValueError("含噪视电阻率必须是有限正值")
    return noisy
