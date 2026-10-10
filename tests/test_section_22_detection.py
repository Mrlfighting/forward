"""基础识别数学规则的独立单元检查。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mt_static.detection.lipschitz import adjacent_alpha_pairs, summarize_alpha
from mt_static.detection.classification import map_candidate_to_station
from mt_static.detection.maxima import extract_modulus_maxima
from mt_static.detection.preprocessing import spatial_profile, validate_and_sort_inputs
from mt_static.detection.tracking import track_maxima_lines
from mt_static.detection.types import MaximaLine, MaximaNode
from mt_static.detection.wavelet import gaussian_derivative_kernel


class PreprocessingTests(unittest.TestCase):
    """检查测点、编号和剖面预处理。"""

    def test_sort_and_align(self) -> None:
        x, f, rho, ids = validate_and_sort_inputs(
            np.array([200.0, 0.0, 100.0]),
            np.array([1.0]),
            np.array([[30.0, 10.0, 20.0]]),
            ["C", "A", "B"],
        )
        np.testing.assert_array_equal(x, [0.0, 100.0, 200.0])
        np.testing.assert_array_equal(f, [1.0])
        np.testing.assert_array_equal(rho, [[10.0, 20.0, 30.0]])
        self.assertEqual(ids, ["A", "B", "C"])

    def test_cubic_interpolation_has_no_extrapolation(self) -> None:
        stations = np.array([0.0, 1.0, 3.0, 4.0])
        values, sample_x, profile = spatial_profile(
            stations, np.array([1.0, 2.0, 4.0, 5.0]), 16, "raw", "cubic"
        )
        self.assertEqual(sample_x[0], stations[0])
        self.assertEqual(sample_x[-1], stations[-1])
        self.assertEqual(profile[0], values[0])
        self.assertEqual(profile[-1], values[-1])


class MaximaTrackingTests(unittest.TestCase):
    """独立检查论文传播规则，避免把空间最近误作模值最近。"""

    def test_maxima_keeps_coefficient_sign(self) -> None:
        coefficients = np.array([[0.0, -3.0, 0.0, 2.0, 0.0]])
        maxima = extract_modulus_maxima(coefficients)
        self.assertTrue(maxima[0, 1])
        self.assertTrue(maxima[0, 3])
        self.assertEqual(np.sign(coefficients[0, 1]), -1)
        self.assertEqual(np.sign(coefficients[0, 3]), 1)

    def test_same_position_takes_priority(self) -> None:
        coefficients = np.zeros((2, 11))
        coefficients[1, 5] = 10.0
        coefficients[0, 4] = 9.0
        coefficients[0, 5] = 1.0
        maxima = coefficients != 0
        line = track_maxima_lines(coefficients, maxima, np.arange(11.0))[0]
        self.assertTrue(line.accepted)
        self.assertEqual(line.nodes[-1].sample_index, 5)

    def test_fallback_minimizes_modulus_difference(self) -> None:
        coefficients = np.zeros((2, 11))
        coefficients[1, 5] = 10.0
        coefficients[0, 4] = 1.0
        coefficients[0, 8] = 9.0
        maxima = coefficients != 0
        line = track_maxima_lines(coefficients, maxima, np.arange(11.0))[0]
        self.assertTrue(line.accepted)
        self.assertEqual(line.nodes[-1].sample_index, 8)

    def test_no_same_sign_terminates_line(self) -> None:
        coefficients = np.zeros((2, 11))
        coefficients[1, 5] = 10.0
        coefficients[0, 5] = -9.0
        maxima = coefficients != 0
        line = track_maxima_lines(coefficients, maxima, np.arange(11.0))[0]
        self.assertFalse(line.accepted)
        self.assertEqual(line.rejection_reason, "no_same_sign_maximum")

    def test_search_interval_excludes_outside_candidate(self) -> None:
        coefficients = np.zeros((2, 13))
        coefficients[1, 4] = 10.0
        coefficients[1, 8] = 8.0
        coefficients[0, 3] = 9.0
        coefficients[0, 2] = 8.0
        maxima = coefficients != 0
        lines = track_maxima_lines(coefficients, maxima, np.arange(13.0))
        self.assertEqual(lines[0].nodes[-1].sample_index, 3)
        self.assertFalse(lines[1].accepted)

    def test_gaussian_derivative_is_antisymmetric(self) -> None:
        kernel = gaussian_derivative_kernel(4)
        np.testing.assert_allclose(kernel, -kernel[::-1])
        self.assertEqual(kernel[kernel.size // 2], 0.0)


class LipschitzTests(unittest.TestCase):
    """检查相邻尺度指数的方向以及逐对输出。"""

    def test_decay_is_negative(self) -> None:
        line = MaximaLine(
            [MaximaNode(2, 2, 2.0, 2.0), MaximaNode(1, 2, 2.0, 4.0), MaximaNode(0, 2, 2.0, 8.0)],
            accepted=True,
        )
        line.alpha_pairs = adjacent_alpha_pairs(line)
        np.testing.assert_allclose(line.alpha_pairs, [-1.0, -1.0])
        self.assertIsNone(summarize_alpha(line, np.array([2, 4, 8]), "pairwise"))

    def test_growth_is_positive(self) -> None:
        line = MaximaLine(
            [MaximaNode(1, 2, 2.0, 8.0), MaximaNode(0, 2, 2.0, 4.0)],
            accepted=True,
        )
        self.assertAlmostEqual(adjacent_alpha_pairs(line)[0], 1.0)

    def test_candidate_maps_to_original_station(self) -> None:
        index, distance = map_candidate_to_station(240.0, np.array([0.0, 100.0, 300.0]))
        self.assertEqual(index, 2)
        self.assertEqual(distance, 60.0)


if __name__ == "__main__":
    unittest.main()
