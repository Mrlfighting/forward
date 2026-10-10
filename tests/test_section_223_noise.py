"""核对噪声注入的正值约束、配对倍率和逐频指标。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "section_22_detection" / "223_noise_robustness"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from metrics import trial_metrics, wilson_interval
from noise import apply_noise, relative_gaussian_factor


class NoiseExperimentTests(unittest.TestCase):
    """阻止噪声处理改变固定正演或评价粒度。"""

    def test_relative_gaussian_factor_and_paired_inputs(self) -> None:
        z = np.array([[0.0, 1.0], [-1.0, 0.5]])
        factors, redraws = relative_gaussian_factor(z, 0.1, np.random.default_rng(7))
        self.assertEqual(redraws, 0)
        np.testing.assert_allclose(factors, 1.0 + 0.1 * z)
        clean_a = np.full((2, 2), 100.0)
        clean_b = np.full((2, 2), 20.0)
        np.testing.assert_allclose(apply_noise(clean_a, factors) / clean_a,
                                   apply_noise(clean_b, factors) / clean_b)
        np.testing.assert_array_equal(clean_a, np.full((2, 2), 100.0))

    def test_nonpositive_factor_redraws_only_that_sample(self) -> None:
        z = np.array([[-20.0, 0.0]])
        factors, redraws = relative_gaussian_factor(z, 0.15, np.random.default_rng(11))
        self.assertGreaterEqual(redraws, 1)
        self.assertGreater(factors[0, 0], 0)
        self.assertEqual(factors[0, 1], 1.0)

    def test_zero_noise_keeps_input_exactly(self) -> None:
        z = np.array([[2.0, -3.0]])
        factors, redraws = relative_gaussian_factor(z, 0.0, np.random.default_rng(3))
        np.testing.assert_array_equal(factors, np.ones_like(z))
        self.assertEqual(redraws, 0)

    def test_strict_and_region_metrics_are_separate(self) -> None:
        votes = np.array([[False, True, False, True, False],
                          [False, False, True, False, False]])
        result = SimpleNamespace(per_frequency_votes=votes, station_x_m=np.array([-1000, -500, 0, 500, 1000]))
        metrics = trial_metrics(result, [0, 4], True, 500.0)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 3, 4))
        self.assertEqual(metrics["region_frequency_counts"], [1, 1])
        self.assertEqual(metrics["outside_region_candidates"], 1)

    def test_wilson_interval_is_bounded(self) -> None:
        low, high = wilson_interval(15, 30)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
