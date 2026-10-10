"""核对 2.3.1 试验中会直接改变科学结论的数据定义。"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "simpeg"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
RUN_PATH = PROJECT_ROOT / "experiments" / "section_23_inversion" / "231_phase_noise" / "run.py"
SPEC = importlib.util.spec_from_file_location("section_231_run", RUN_PATH)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)

from mt_static.inversion import GLSShapeOnlySimulation, build_full_survey, build_shape_survey, build_inversion_mesh, build_simulation, join_data
from mt_static.inversion.solver import InversionSettings, active_cells


class PhaseNoiseExperimentTests(unittest.TestCase):
    """测试新旧频率及测点变化后最易被混用的输入和变换。"""

    @classmethod
    def setUpClass(cls):
        cls.setup = RUN.read_setup()

    def test_noise_and_error_masks(self):
        """噪声与额外误差严格限定在八测点的同六个频率。"""

        setup = self.setup
        mask = setup["noise_mask"]
        self.assertEqual(mask.shape, (20, 8))
        self.assertEqual(int(mask.sum()), 48)
        np.testing.assert_array_equal(setup["frequency_indices"], [0, 11, 13, 15, 17, 18])
        delta = setup["corrupted_phase"] - setup["static_phase"]
        np.testing.assert_array_equal(delta[~mask], 0.0)
        np.testing.assert_allclose(delta[mask], setup["noise"].ravel(), atol=1e-13, rtol=0)
        np.testing.assert_allclose(setup["static_full"].reshape(20, 2, 8)[:, 0], setup["static_rho"])
        phase_std = setup["static_std"].reshape(20, 2, 8)[:, 1]
        np.testing.assert_array_equal(phase_std[~mask], 0.57)
        np.testing.assert_array_equal(phase_std[mask], np.hypot(0.57, 5.0))
        self.assertAlmostEqual(setup["phase_rms"], 4.726968976533248, places=12)

    def test_phase_preserving_still_uses_ordinary_rho(self):
        """普通点视电阻率始终在观测中，静态点的绝对值不会混入。"""

        setup = self.setup
        reduced = setup["reduced"].reshape(20, 14)
        np.testing.assert_array_equal(reduced[:, :6], setup["static_rho"][:, setup["normal_indices"]])
        np.testing.assert_array_equal(reduced[:, 6:], setup["corrupted_phase"])
        self.assertEqual(setup["reduced"].size, 280)

    def test_each_static_site_has_independent_amplitude_invariance(self):
        """分别改变两点幅值后，白化 shape 均完全消除常数对数幅值。"""

        setup = self.setup
        mesh = build_inversion_mesh()
        settings = InversionSettings(**setup["config"]["inversion"])
        active = active_cells(mesh, settings)
        full_survey = build_full_survey(setup["frequencies"], setup["stations"])
        shape_survey = build_shape_survey(setup["frequencies"], setup["stations"], setup["static_indices"])
        full_simulation = build_simulation(mesh, full_survey, active, settings)
        shape = GLSShapeOnlySimulation(
            full_simulation, shape_survey, setup["frequencies"], 8, setup["static_indices"],
            setup["static_full"], setup["static_std"],
        )
        altered = setup["static_full"].reshape(20, 2, 8).copy()
        altered[:, 0, setup["static_indices"][0]] *= 2.3
        altered[:, 0, setup["static_indices"][1]] *= 0.4
        np.testing.assert_allclose(
            shape.transform_observed(altered.ravel()),
            shape.transform_observed(setup["static_full"]), atol=1e-10, rtol=0,
        )
        self.assertEqual(shape.n_shape, 38)

    def test_saved_reference_exceeds_homogeneous_start(self):
        """正式结果必须来自通过目标区门槛的验证运行。"""

        path = PROJECT_ROOT / "results" / "section_23" / "231_phase_noise" / "summary.json"
        if not path.is_file():
            self.skipTest("尚未生成 2.3.1 正式结果摘要")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "validated")
        self.assertTrue(summary["clean_reference_better_than_initial"])
        self.assertLess(
            summary["results"]["clean_full"]["roi_log10_rmse"],
            summary["initial_model_metrics"]["roi_log10_rmse"],
        )

    def test_shape_jacobian_and_adjoint_on_independent_linear_forward(self):
        """用独立线性正演检验 shape 导数的有限差分及转置关系。"""

        frequencies = np.array([1.0, 3.0, 9.0])
        stations = np.array([-1000.0, 1000.0])
        static_indices = np.array([1])
        rho = np.array([[100.0, 150.0], [90.0, 170.0], [80.0, 180.0]])
        phase = np.full_like(rho, -135.0)
        full = join_data(rho, phase)
        std = join_data(0.02 * rho, np.full_like(rho, 0.57))
        jacobian = np.random.default_rng(11).normal(0, 0.2, size=(full.size, 3))

        class LinearForward:
            def __init__(self):
                self.model = None

            def fields(self, m=None):
                return None

            def dpred(self, m=None, f=None):
                return full + jacobian @ m

            def Jvec(self, m, v, f=None):
                return jacobian @ v

            def Jtvec(self, m, v, f=None):
                return jacobian.T @ v

        base = LinearForward()
        survey = build_shape_survey(frequencies, stations, static_indices)
        shape = GLSShapeOnlySimulation(base, survey, frequencies, 2, static_indices, full, std)
        model = np.array([0.2, -0.1, 0.05])
        direction = np.array([0.4, 0.3, -0.2])
        epsilon = 1e-5
        finite_difference = (
            shape.dpred(model + epsilon * direction) - shape.dpred(model - epsilon * direction)
        ) / (2 * epsilon)
        jvec = shape.Jvec(model, direction)
        np.testing.assert_allclose(jvec, finite_difference, atol=1e-8, rtol=1e-6)
        data_direction = np.array([0.6, -0.3])
        self.assertAlmostEqual(
            float(np.dot(jvec, data_direction)),
            float(np.dot(direction, shape.Jtvec(model, data_direction))), places=9,
        )


if __name__ == "__main__":
    unittest.main()
