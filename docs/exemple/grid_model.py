"""教学案例的二维网格、单低阻体模型和活动单元。"""

from __future__ import annotations

import numpy as np
from discretize import TensorMesh


def vertical_nodes(step_m: float) -> np.ndarray:
    """为宽频段在浅部加密，并向深部延伸以容纳最低频率。"""

    if step_m in (25.0, 50.0):
        shallow = np.arange(0.0, 3000.0 + step_m / 2.0, step_m)
    else:
        raise ValueError("示例仅使用已设计的 25 m 或 50 m 浅部纵向间距")
    middle = np.arange(3500.0, 8000.0 + 1.0, 500.0)
    deep = np.array([10000.0, 14000.0, 22000.0, 38000.0, 70000.0,
                     140000.0, 280000.0, 560000.0, 1120000.0])
    return -np.unique(np.r_[shallow, middle, deep])[::-1]


def build_mesh(config: dict, kind: str) -> TensorMesh:
    """构造独立的正演网格或较粗的反演网格。"""

    if kind not in ("forward", "inversion"):
        raise ValueError("网格类型只能是 forward 或 inversion")
    parameters = config["mesh"]
    cell_width = float(parameters[f"{kind}_core_cell_width_m"])
    cell_count = int(parameters[f"{kind}_core_cell_count"])
    pad_count = int(parameters[f"{kind}_padding_cell_count_each_side"])
    factor = float(parameters["padding_factor"])
    if cell_width <= 0 or cell_count <= 0 or cell_count % 2 or pad_count < 1 or factor <= 1:
        raise ValueError("网格尺寸或外扩参数无效")
    horizontal = [(cell_width, pad_count, -factor), (cell_width, cell_count), (cell_width, pad_count, factor)]
    centered = TensorMesh([horizontal, [1.0]], x0="CN")
    z_nodes = vertical_nodes(25.0 if kind == "forward" else 50.0)
    return TensorMesh([horizontal, np.diff(z_nodes)], x0=[centered.x0[0], z_nodes[0]])


def anomaly_mask(mesh: TensorMesh, anomaly: dict) -> np.ndarray:
    """以网格单元中心判断哪些单元属于低阻体。"""

    centers = mesh.cell_centers
    return (
        (centers[:, 0] > anomaly["x_min_m"])
        & (centers[:, 0] < anomaly["x_max_m"])
        & (centers[:, 1] > anomaly["z_min_m"])
        & (centers[:, 1] < anomaly["z_max_m"])
    )


def build_true_model(mesh: TensorMesh, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """用对数电导率表示均匀背景及单个矩形低阻体。"""

    background = float(config["background_resistivity_ohm_m"])
    anomaly = config["anomaly"]
    anomaly_rho = float(anomaly["resistivity_ohm_m"])
    if background <= 0 or anomaly_rho <= 0 or anomaly_rho >= background:
        raise ValueError("示例要求正电阻率且异常体比背景更低阻")
    mask = anomaly_mask(mesh, anomaly)
    if not np.any(mask):
        raise ValueError("网格没有解析低阻异常体")
    log_sigma = np.full(mesh.n_cells, np.log(1.0 / background))
    log_sigma[mask] = np.log(1.0 / anomaly_rho)
    return log_sigma, mask


def active_cells(mesh: TensorMesh, config: dict) -> np.ndarray:
    """只把测线附近、地表以下 8 km 的单元列为反演未知量。"""

    centers = mesh.cell_centers
    parameters = config["mesh"]
    return (
        (np.abs(centers[:, 0]) <= parameters["active_half_width_m"])
        & (centers[:, 1] >= -parameters["active_depth_m"])
        & (centers[:, 1] <= 0.0)
    )


def check_geometry(mesh: TensorMesh, mask: np.ndarray, config: dict) -> dict:
    """核对网格实际表达的矩形边界，避免尺寸被悄悄取整。"""

    centers = mesh.cell_centers[mask]
    x_indices = np.searchsorted(mesh.nodes_x, centers[:, 0]) - 1
    z_indices = np.searchsorted(mesh.nodes_y, centers[:, 1]) - 1
    actual = {
        "x_min_m": float(mesh.nodes_x[x_indices].min()),
        "x_max_m": float(mesh.nodes_x[x_indices + 1].max()),
        "z_min_m": float(mesh.nodes_y[z_indices].min()),
        "z_max_m": float(mesh.nodes_y[z_indices + 1].max()),
    }
    for key, value in actual.items():
        if not np.isclose(value, config["anomaly"][key], atol=1e-8, rtol=0):
            raise AssertionError(f"网格实际 {key}={value}，与指定异常体不同")
    actual["cell_count"] = int(mask.sum())
    return actual
