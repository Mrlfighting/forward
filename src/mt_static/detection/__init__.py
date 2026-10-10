"""论文 2.2.1 的基础静态测点识别计算。"""

from .pipeline import detect_static_sites
from .types import DetectionConfig, DetectionResult, FrequencyDiagnostics, MaximaLine, MaximaNode

__all__ = [
    "DetectionConfig",
    "DetectionResult",
    "FrequencyDiagnostics",
    "MaximaLine",
    "MaximaNode",
    "detect_static_sites",
]
