"""构建仅在浅部异常体上不同的两套地电模型。"""

import warnings

import numpy as np
from discretize import TensorMesh
from simpeg.utils import BreakingChangeWarning, model_builder

from .settings import BACKGROUND_RESISTIVITY_OHM_M


ANOMALY_REGIONS = {
    "left_deep_conductor": ([-8000.0, -8000.0], [-2000.0, -2000.0], 10.0),
    "right_deep_resistor": ([2000.0, -8000.0], [8000.0, -2000.0], 1000.0),
    "left_shallow_conductor": ([-5100.0, -5.0], [-4900.0, 0.0], 10.0),
    "right_shallow_conductor": ([4900.0, -5.0], [5100.0, 0.0], 10.0),
}


def describe_anomaly_regions(mesh: TensorMesh) -> dict[str, dict]:
    """记录每个异常体的坐标范围及网格实际表示的尺寸。"""
    descriptions = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BreakingChangeWarning)
        for region_name, (lower_corner, upper_corner, resistivity) in ANOMALY_REGIONS.items():
            cell_indices = np.asarray(
                model_builder.get_indices_block(
                    np.asarray(lower_corner), np.asarray(upper_corner), mesh.cell_centers
                ),
                dtype=int,
            )
            if cell_indices.size == 0:
                raise ValueError(f"异常体 {region_name} 未被网格表示")
            centers = mesh.cell_centers[cell_indices]
            x_indices = np.searchsorted(mesh.nodes_x, centers[:, 0]) - 1
            z_indices = np.searchsorted(mesh.nodes_y, centers[:, 1]) - 1
            effective_lower = [
                float(np.min(mesh.nodes_x[x_indices])),
                float(np.min(mesh.nodes_y[z_indices])),
            ]
            effective_upper = [
                float(np.max(mesh.nodes_x[x_indices + 1])),
                float(np.max(mesh.nodes_y[z_indices + 1])),
            ]
            descriptions[region_name] = {
                "resistivity_ohm_m": float(resistivity),
                "specified_bounds_m": [list(lower_corner), list(upper_corner)],
                "specified_size_m": [
                    float(upper_corner[0] - lower_corner[0]),
                    float(upper_corner[1] - lower_corner[1]),
                ],
                "effective_bounds_m": [effective_lower, effective_upper],
                "effective_size_m": [
                    effective_upper[0] - effective_lower[0],
                    effective_upper[1] - effective_lower[1],
                ],
                "cell_count": int(cell_indices.size),
            }
    return descriptions


def build_models(mesh: TensorMesh) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """返回无浅部体和有浅部体的对数电导率及异常单元数。"""
    without_shallow = np.full(mesh.n_cells, np.log(1.0 / BACKGROUND_RESISTIVITY_OHM_M))
    cell_counts: dict[str, int] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BreakingChangeWarning)
        for region_name in ("left_deep_conductor", "right_deep_resistor"):
            lower_corner, upper_corner, resistivity = ANOMALY_REGIONS[region_name]
            cell_indices = model_builder.get_indices_block(
                np.asarray(lower_corner), np.asarray(upper_corner), mesh.cell_centers
            )
            cell_counts[region_name] = int(np.asarray(cell_indices).size)
            without_shallow[cell_indices] = np.log(1.0 / resistivity)
        with_shallow = without_shallow.copy()
        for region_name in ("left_shallow_conductor", "right_shallow_conductor"):
            lower_corner, upper_corner, resistivity = ANOMALY_REGIONS[region_name]
            cell_indices = model_builder.get_indices_block(
                np.asarray(lower_corner), np.asarray(upper_corner), mesh.cell_centers
            )
            cell_counts[region_name] = int(np.asarray(cell_indices).size)
            with_shallow[cell_indices] = np.log(1.0 / resistivity)
    changed_cells = int(np.count_nonzero(without_shallow != with_shallow))
    if changed_cells != cell_counts["left_shallow_conductor"] + cell_counts["right_shallow_conductor"]:
        raise AssertionError("两套模型的差异并非仅来自浅部异常体")
    return without_shallow, with_shallow, cell_counts
