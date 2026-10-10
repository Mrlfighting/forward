"""核查多场景模型与已验证的 2.1 正演模型完全一致。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
SIMPEG_ROOT = PROJECT_ROOT / "external" / "simpeg"
if str(SIMPEG_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPEG_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mt_static.forward.mesh import build_mesh
from mt_static.forward.scenario_config import load_base_config, load_scenario_config
from mt_static.forward.scenario_model import build_scenario_models


class ForwardScenarioTests(unittest.TestCase):
    """重点防止场景配置在网格上悄悄改变物理模型。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base = load_base_config(PROJECT_ROOT / "configs/section_21/base_v1.json")
        cls.mesh = build_mesh()

    def _load_model(self, scenario_id: str):
        scenario = load_scenario_config(
            PROJECT_ROOT / f"configs/section_21/scenarios/{scenario_id}.json",
            self.base["base_id"],
        )
        return build_scenario_models(self.mesh, self.base, scenario)

    def test_two_body_model_matches_validated_forward(self) -> None:
        baseline, model, geometry = self._load_model("two_low_200x5_v1")
        path = PROJECT_ROOT / "results/section_21_forward/structured/section_21_forward.npz"
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(baseline, saved["without_shallow_log_conductivity"])
            np.testing.assert_array_equal(model, saved["with_shallow_log_conductivity"])
            np.testing.assert_array_equal(self.mesh.nodes_x, saved["mesh_x_nodes_m"])
            np.testing.assert_array_equal(self.mesh.nodes_y, saved["mesh_z_nodes_m"])
        self.assertEqual(geometry["different_cell_count"], 4)

    def test_four_bodies_have_exact_geometry(self) -> None:
        baseline, model, geometry = self._load_model("four_low_200x5_v1")
        self.assertEqual(geometry["different_cell_count"], 8)
        centers = []
        for body in geometry["shallow_bodies"]:
            np.testing.assert_allclose(
                body["specified_bounds_m"], body["effective_bounds_m"], atol=1e-7, rtol=0.0
            )
            np.testing.assert_allclose(body["effective_size_m"], [200.0, 5.0], atol=1e-7, rtol=0.0)
            self.assertEqual(body["cell_count"], 2)
            self.assertEqual(body["resistivity_ohm_m"], 10.0)
            bounds = body["effective_bounds_m"]
            centers.append((bounds[0][0] + bounds[1][0]) / 2.0)
        np.testing.assert_allclose(centers, [-5000.0, -1500.0, 1500.0, 5000.0], atol=1e-7, rtol=0.0)
        self.assertEqual(int(np.count_nonzero(baseline != model)), 8)

    def test_off_grid_body_is_rejected(self) -> None:
        scenario = load_scenario_config(
            PROJECT_ROOT / "configs/section_21/scenarios/four_low_200x5_v1.json",
            self.base["base_id"],
        )
        scenario["shallow_bodies"][0]["x_max_m"] += 1.0
        with self.assertRaisesRegex(ValueError, "不对齐"):
            build_scenario_models(self.mesh, self.base, scenario)


if __name__ == "__main__":
    unittest.main()
