"""检验于生宝原文模式的逐对 α 与异常边界定位。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mt_static.detection.yu2017 import Yu2017Config, detect_yu2017


class Yu2017DetectionTests(unittest.TestCase):
    """检验输出仍为逐频逐尺度证据，不变成最终测点分类。"""

    def setUp(self) -> None:
        """准备固定空间剖面和论文模式配置。"""

        self.stations = np.arange(-10000.0, 10001.0, 500.0)
        self.frequencies = np.array([1.0])
        self.config = Yu2017Config(
            scales=(2, 4, 8, 16, 32),
            interpolation_points=512,
            interpolation_kind="linear",
            wavelet_name="gaussian_derivative",
            boundary_mode="reflect",
        )

    def test_constant_profile_has_no_negative_pair(self) -> None:
        """均匀剖面不能凭数值边界生成静态候选。"""

        rho = np.full((1, self.stations.size), 100.0)
        result = detect_yu2017(self.stations, self.frequencies, rho, self.config)
        self.assertEqual(result.negative_pairs, [[]])
        self.assertLess(np.max(np.abs(result.diagnostics[0].coefficients)), 1e-10)
        self.assertFalse(any(line.accepted for line in result.diagnostics[0].maxima_lines))

    def test_local_shift_produces_two_flank_positions(self) -> None:
        """窄幅局部异常以两侧传播位置出现，逐对 α 保持独立。"""

        rho = np.full((1, self.stations.size), 100.0)
        rho[0, self.stations == 0.0] *= 2.0
        result = detect_yu2017(self.stations, self.frequencies, rho, self.config)
        events = result.negative_pairs[0]
        self.assertGreater(len(events), 0)
        self.assertTrue(all(event.alpha < 0.0 for event in events))
        self.assertEqual(
            {np.sign(event.candidate_x_m) for event in events},
            {-1.0, 1.0},
        )
        self.assertTrue(all(abs(event.candidate_x_m) < 500.0 for event in events))
        self.assertTrue(all(not line.static_candidate for line in result.diagnostics[0].maxima_lines))

    def test_log_input_is_not_silently_used_in_paper_mode(self) -> None:
        """原文模式显式拒绝以对数视电阻率冒充原始视电阻率。"""

        config = Yu2017Config(
            scales=self.config.scales,
            interpolation_points=512,
            interpolation_kind="linear",
            wavelet_name="gaussian_derivative",
            boundary_mode="reflect",
            signal_transform="log",
        )
        rho = np.full((1, self.stations.size), 100.0)
        with self.assertRaises(ValueError):
            detect_yu2017(self.stations, self.frequencies, rho, config)


if __name__ == "__main__":
    unittest.main()
