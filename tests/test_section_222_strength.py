"""检验静态位移只修改目标视电阻率及逐频评价口径。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "section_22_detection" / "222_shift_strength"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from strength import apply_static_factor, detection_boundary, frequency_station_metrics


class StrengthExperimentTests(unittest.TestCase):
    """阻止注入、真值矩阵和边界表述发生隐性变化。"""

    def test_injection_changes_only_two_target_columns(self) -> None:
        baseline = np.arange(1.0, 13.0).reshape(3, 4)
        shifted = apply_static_factor(baseline, [1, 3], 0.7)
        np.testing.assert_array_equal(shifted[:, [0, 2]], baseline[:, [0, 2]])
        np.testing.assert_allclose(shifted[:, [1, 3]], baseline[:, [1, 3]] * 0.7)
        np.testing.assert_array_equal(baseline, np.arange(1.0, 13.0).reshape(3, 4))

    def test_metrics_use_frequency_station_unit(self) -> None:
        votes = np.array([[False, True, False, False], [True, False, False, True]])
        result = frequency_station_metrics(votes, [1, 3], False)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 2))
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 4 / 7)

    def test_negative_control_has_undefined_recall_and_f1(self) -> None:
        result = frequency_station_metrics(np.zeros((2, 4), dtype=bool), [1, 3], True)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 0, 0))
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["f1"])

    def test_nonmonotonic_detection_has_no_single_boundary(self) -> None:
        rows = [
            {"cs": factor, "delta_s": delta, "target_detected_any_frequency": [hit, hit],
             "all_targets_detected_any_frequency": hit}
            for factor, delta, hit in ((0.8, 0.1, True), (0.7, 0.15, False), (0.5, 0.3, True))
        ]
        boundary = detection_boundary(rows, None)
        self.assertEqual(boundary["status"], "non_monotonic")
        self.assertIsNone(boundary["delta_s_interval"])


if __name__ == "__main__":
    unittest.main()
