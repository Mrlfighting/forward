"""按公共模型与任意浅部体列表构造二维电导率模型。"""

from __future__ import annotations

import warnings

import numpy as np
from discretize import TensorMesh
from simpeg.utils import BreakingChangeWarning, model_builder


def _body_cells(mesh: TensorMesh, body: dict) -> np.ndarray:
    """按现有 2.1 模型的单元中心选择规则获取异常单元。"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BreakingChangeWarning)
        indices = model_builder.get_indices_block(
            np.array([body["x_min_m"], body["z_min_m"]]),
            np.array([body["x_max_m"], body["z_max_m"]]),
            mesh.cell_centers,
        )
    return np.asarray(indices, dtype=int)


def _describe_cells(mesh: TensorMesh, body: dict, indices: np.ndarray) -> dict:
    """报告输入几何和网格实际表示几何。"""

    centers = mesh.cell_centers[indices]
    x_index = np.searchsorted(mesh.nodes_x, centers[:, 0]) - 1
    z_index = np.searchsorted(mesh.nodes_y, centers[:, 1]) - 1
    effective = [
        [float(np.min(mesh.nodes_x[x_index])), float(np.min(mesh.nodes_y[z_index]))],
        [float(np.max(mesh.nodes_x[x_index + 1])), float(np.max(mesh.nodes_y[z_index + 1]))],
    ]
    specified = [
        [body["x_min_m"], body["z_min_m"]],
        [body["x_max_m"], body["z_max_m"]],
    ]
    return {
        "body_id": body["body_id"],
        "resistivity_ohm_m": body["resistivity_ohm_m"],
        "specified_bounds_m": specified,
        "effective_bounds_m": effective,
        "specified_size_m": [body["x_max_m"] - body["x_min_m"], body["z_max_m"] - body["z_min_m"]],
        "effective_size_m": [effective[1][0] - effective[0][0], effective[1][1] - effective[0][1]],
        "cell_count": int(indices.size),
    }


def build_scenario_models(mesh: TensorMesh, base: dict, scenario: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """生成公共基线与场景模型，并拒绝无法准确表示的浅部体。"""

    baseline = np.full(mesh.n_cells, np.log(1.0 / base["background_resistivity_ohm_m"]))
    descriptions = {"deep_bodies": [], "shallow_bodies": []}
    for body in base["deep_bodies"]:
        indices = _body_cells(mesh, body)
        if indices.size == 0:
            raise ValueError(f"深部体 {body['body_id']} 未被网格表示")
        baseline[indices] = np.log(1.0 / body["resistivity_ohm_m"])
        descriptions["deep_bodies"].append(_describe_cells(mesh, body, indices))
    model = baseline.copy()
    occupied = np.zeros(mesh.n_cells, dtype=bool)
    for body in scenario["shallow_bodies"]:
        indices = _body_cells(mesh, body)
        if indices.size == 0:
            raise ValueError(f"浅部体 {body['body_id']} 未被网格表示")
        if np.any(occupied[indices]):
            raise ValueError(f"浅部体 {body['body_id']} 与其他浅部体占用同一网格单元")
        description = _describe_cells(mesh, body, indices)
        specified = np.asarray(description["specified_bounds_m"], dtype=float)
        effective = np.asarray(description["effective_bounds_m"], dtype=float)
        if not np.allclose(specified, effective, atol=1e-7, rtol=0.0):
            raise ValueError(f"浅部体 {body['body_id']} 与当前网格不对齐：实际边界 {effective.tolist()}")
        occupied[indices] = True
        model[indices] = np.log(1.0 / body["resistivity_ohm_m"])
        descriptions["shallow_bodies"].append(description)
    descriptions["different_cell_count"] = int(np.count_nonzero(baseline != model))
    return baseline, model, descriptions
