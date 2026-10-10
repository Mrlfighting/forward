"""一阶高斯导数小波的离散连续小波变换。"""

from __future__ import annotations

import numpy as np


def gaussian_derivative_kernel(scale: int) -> np.ndarray:
    """按论文式中的尺度平方根归一化构造反对称小波。"""

    if scale <= 0:
        raise ValueError("小波尺度必须为正")
    radius = max(int(np.ceil(5.0 * scale)), 3)
    sample = np.arange(-radius, radius + 1, dtype=float)
    normalized = sample / scale
    return -normalized * np.exp(-0.5 * normalized**2) / np.sqrt(scale)


def continuous_wavelet_transform(
    profile: np.ndarray,
    scales: np.ndarray,
    wavelet_name: str,
    boundary_mode: str,
) -> np.ndarray:
    """按显式指定的小波与边界方式沿空间剖面卷积。"""

    values = np.asarray(profile, dtype=float)
    if wavelet_name != "gaussian_derivative":
        raise ValueError("当前 CWT 实现只支持 gaussian_derivative 小波")
    if boundary_mode not in {"reflect", "symmetric", "constant"}:
        raise ValueError("boundary_mode 只支持 reflect、symmetric 或 constant")
    rows = []
    for scale in scales:
        kernel = gaussian_derivative_kernel(int(scale))
        padding = kernel.size // 2
        padded = np.pad(values, pad_width=padding, mode=boundary_mode)
        rows.append(np.convolve(padded, kernel[::-1], mode="valid"))
    return np.vstack(rows)
