"""复杂层状组合模型的独立网格、真值模型及评价区域。"""

from __future__ import annotations

import numpy as np
from discretize import TensorMesh


def build_layered_forward_mesh(static_positions_m: np.ndarray) -> TensorMesh:
    """复现旧实验精确解析 4 m 浅部体的 156×105 正演网格。"""

    nodes = np.arange(-10200.0, 10200.0 + 1.0, 200.0)
    offsets = np.array([
        -200.0, -100.0, -60.0, -30.0, -14.0, -6.0, -2.0,
        2.0, 6.0, 14.0, 30.0, 60.0, 100.0, 200.0,
    ])
    for center in np.asarray(static_positions_m, dtype=float):
        nodes = nodes[(nodes <= center - 200.0) | (nodes >= center + 200.0)]
        nodes = np.r_[nodes, center + offsets]
    nodes = np.unique(np.round(nodes, 8))
    padding = 200.0 * 1.3 ** np.arange(1, 17)
    widths_x = np.r_[padding[::-1], np.diff(nodes), padding]
    shallow = np.r_[0.0, 4.0, 10.0, np.arange(20.0, 81.0, 10.0)]
    middle = np.r_[
        np.arange(120.0, 681.0, 40.0),
        np.arange(780.0, 3681.0, 100.0),
        np.arange(3880.0, 12001.0, 200.0),
    ]
    deep = 12000.0 + np.cumsum(2000.0 * 2.0 ** np.arange(10))
    nodes_z = -np.unique(np.r_[shallow, middle, deep])[::-1]
    return TensorMesh(
        [widths_x, np.diff(nodes_z)],
        x0=np.r_[nodes[0] - padding.sum(), nodes_z[0]],
    )


def target_masks(mesh: TensorMesh, targets: list[dict]) -> list[np.ndarray]:
    """按两个椭圆的中心和半轴取得单元中心掩码。"""

    centers = mesh.cell_centers
    return [
        ((centers[:, 0] - item["center_x_m"]) / item["radius_x_m"]) ** 2
        + ((centers[:, 1] - item["center_z_m"]) / item["radius_z_m"]) ** 2 <= 1.0
        for item in targets
    ]


def layered_background(mesh: TensorMesh, layers: list[dict], background_resistivity: float) -> np.ndarray:
    """仅按深度赋三层背景，深于 12 km 的边界仍采用固定背景值。"""

    model = np.full(mesh.n_cells, np.log(1.0 / background_resistivity))
    depth = mesh.cell_centers[:, 1]
    for item in layers:
        mask = (depth <= -item["top_depth_m"]) & (depth >= -item["bottom_depth_m"])
        model[mask] = np.log(1.0 / item["resistivity_ohm_m"])
    return model


def build_layered_models(
    mesh: TensorMesh,
    layers: list[dict],
    targets: list[dict],
    surface_bodies: list[dict],
    background_resistivity: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """生成无浅部体和有浅部体的对数电导率真值模型。"""

    clean = layered_background(mesh, layers, background_resistivity)
    for mask, item in zip(target_masks(mesh, targets), targets):
        if not np.any(mask):
            raise ValueError("椭圆目标没有落入正演网格")
        clean[mask] = np.log(1.0 / item["resistivity_ohm_m"])
    static = clean.copy()
    centers = mesh.cell_centers
    counts = []
    for item in surface_bodies:
        mask = (
            (np.abs(centers[:, 0] - item["center_x_m"]) <= item["width_m"] / 2.0 + 1e-8)
            & (np.abs(centers[:, 1] - item["center_z_m"]) <= item["height_m"] / 2.0 + 1e-8)
        )
        count = int(mask.sum())
        if count != 1:
            raise ValueError(f"每个 4 m 浅部体必须恰好占一个单元，实际为 {count}")
        counts.append(count)
        static[mask] = np.log(1.0 / item["resistivity_ohm_m"])
    return clean, static, counts
