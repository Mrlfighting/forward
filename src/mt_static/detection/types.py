"""静态测点识别的参数与诊断数据结构。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    """保存识别约定；实验数值由外部配置提供。"""

    scales: tuple[int, ...]
    interpolation_points: int
    signal_transform: str = "raw"
    interpolation_kind: str = "linear"
    wavelet_name: str = "gaussian_derivative"
    transform_type: str = "cwt"
    boundary_mode: str = "reflect"
    alpha_summary_method: str = "pairwise"
    candidate_rule: str = "any_pair_negative"
    min_line_length: int | None = None
    modulus_quantile: float | None = None
    mapping_max_distance_m: float | None = None
    alpha_threshold: float = 0.0


@dataclass(frozen=True)
class MaximaNode:
    """单个尺度上的模极大值节点。"""

    scale_index: int
    sample_index: int
    x_m: float
    coefficient: float

    @property
    def modulus(self) -> float:
        """返回小波系数的模值。"""

        return abs(self.coefficient)

    @property
    def sign(self) -> int:
        """返回小波系数的符号。"""

        return int(np.sign(self.coefficient))


@dataclass
class MaximaLine:
    """从最大尺度向最小尺度追踪的一条传播线。"""

    nodes: list[MaximaNode]
    accepted: bool = False
    rejection_reason: str | None = None
    alpha: float | None = None
    alpha_pairs: list[float] | None = None
    static_candidate: bool = False
    candidate_x_m: float | None = None
    mapped_station_index: int | None = None
    mapping_distance_m: float | None = None


@dataclass
class FrequencyDiagnostics:
    """单一频率下保留的全部空间与小波中间量。"""

    frequency_hz: float
    station_x_m: np.ndarray
    station_ids: list[str]
    signal_at_stations: np.ndarray
    sample_x_m: np.ndarray
    signal_profile: np.ndarray
    coefficients: np.ndarray
    modulus_maxima: np.ndarray
    selected_maxima: np.ndarray
    maxima_lines: list[MaximaLine]
    candidate_x_m: list[float]
    candidate_station_indices: list[int]


@dataclass
class DetectionResult:
    """逐频候选及其跨频出现次数；不擅自生成最终静态点分类。"""

    station_x_m: np.ndarray
    station_ids: list[str]
    frequencies_hz: np.ndarray
    vote_counts: np.ndarray
    per_frequency_votes: np.ndarray
    median_alpha: np.ndarray
    frequency_diagnostics: list[FrequencyDiagnostics]
    config: DetectionConfig
