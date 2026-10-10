"""严格沿用旧 L4 实测反演网格及活动域。"""

from __future__ import annotations

import numpy as np
from discretize import TensorMesh
from pymatsolver import SolverLU
from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem


def build_mesh(stations: np.ndarray) -> TensorMesh:
    station_spacing = float(np.median(np.diff(np.sort(stations))))
    core_width = max(0.5 * station_spacing, 1.0)
    core_margin = 5.0 * core_width
    core_left = np.floor((stations.min() - core_margin) / core_width) * core_width
    core_right = np.ceil((stations.max() + core_margin) / core_width) * core_width
    core_nodes = np.arange(core_left, core_right + core_width, core_width)
    core_h = np.diff(core_nodes)
    padding = core_width * 1.40 ** np.arange(1, 23)
    hx = np.r_[padding[::-1], core_h, padding]
    x0 = core_nodes[0] - padding.sum()
    depth_nodes = np.array([
        0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0,
        150.0, 200.0, 300.0, 500.0, 750.0, 1000.0, 1500.0,
        2000.0, 3000.0, 5000.0, 7500.0, 10000.0, 15000.0,
        20000.0, 30000.0, 50000.0, 75000.0, 100000.0, 150000.0,
        200000.0, 300000.0,
    ])
    deep_nodes = 300000.0 + np.cumsum(100000.0 * 1.6 ** np.arange(6))
    z_nodes = -np.unique(np.r_[depth_nodes, deep_nodes])[::-1]
    return TensorMesh([hx, np.diff(z_nodes)], x0=np.r_[x0, z_nodes[0]])


def build_active_cells(mesh: TensorMesh, stations: np.ndarray,
                       active_depth: float, side_padding: float) -> np.ndarray:
    centers = mesh.cell_centers
    return (
        (centers[:, 1] >= -active_depth)
        & (centers[:, 1] <= 0.0)
        & (centers[:, 0] >= stations.min() - side_padding)
        & (centers[:, 0] <= stations.max() + side_padding)
    )


def build_simulation(mesh: TensorMesh, survey: nsem.survey.Survey,
                     active_cells: np.ndarray, background_rho: float):
    inject = maps.InjectActiveCells(
        mesh, active_cells=active_cells,
        value_inactive=np.log(1.0 / background_rho),
    )
    sigma_map = maps.ExpMap(mesh) * inject
    return nsem.simulation.Simulation2DElectricField(
        mesh, survey=survey, sigmaMap=sigma_map, solver=SolverLU,
        storeInnerProduct=True,
    )
