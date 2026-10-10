"""用普通测点数据求共同一维初始模型，供层状二维反演使用。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from discretize import TensorMesh
from pymatsolver import SolverLU
from simpeg import directives, inverse_problem, inversion, maps, optimization, regularization
from simpeg.data import Data
from simpeg.data_misfit import L2DataMisfit
from simpeg.electromagnetics import natural_source as nsem


@dataclass
class OneDBackgroundResult:
    """保留一维剖面、投影到二维的初值和实际拟合信息。"""

    profile: np.ndarray
    depth_m: np.ndarray
    active_start: np.ndarray
    full_log_conductivity: np.ndarray
    phi_d: float
    elapsed_seconds: float


def build_normal_station_survey(
    frequencies: np.ndarray, stations: np.ndarray, normal_indices: np.ndarray
) -> nsem.survey.Survey:
    """普通测点的 rho_a 与相位均进入共同一维背景预反演。"""

    locations = np.c_[stations[normal_indices], np.zeros(normal_indices.size)]
    sources = []
    for frequency in frequencies:
        receivers = [
            nsem.receivers.Impedance(locations, orientation="xy", component="apparent_resistivity"),
            nsem.receivers.Impedance(locations, orientation="xy", component="phase"),
        ]
        sources.append(nsem.sources.Planewave(receivers, float(frequency)))
    return nsem.survey.Survey(sources)


def _vertical_projection(
    mesh: TensorMesh, maximum_depth_m: float, background_resistivity: float
) -> tuple[maps.LinearMap, TensorMesh, np.ndarray]:
    """将一维深度参数复制到二维网格各水平单元。"""

    nx, nz = mesh.shape_cells
    depth_centers = mesh.cell_centers_y
    selected_rows = np.flatnonzero((depth_centers >= -maximum_depth_m) & (depth_centers <= 0.0))
    row_to_parameter = {int(row): index for index, row in enumerate(selected_rows)}
    cell_rows = np.tile(np.arange(nz), (nx, 1)).ravel(order="F")
    rows, columns = [], []
    for cell, row in enumerate(cell_rows):
        parameter = row_to_parameter.get(int(row))
        if parameter is not None:
            rows.append(cell)
            columns.append(parameter)
    projection = sp.csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(mesh.n_cells, selected_rows.size),
    )
    offset = np.full(mesh.n_cells, np.log(1.0 / background_resistivity))
    offset[np.asarray(rows, dtype=int)] = 0.0
    profile_mesh = TensorMesh(
        [mesh.h[1][selected_rows]],
        x0=[mesh.nodes_y[selected_rows[0]]],
    )
    return maps.LinearMap(projection, b=offset), profile_mesh, selected_rows


def invert_common_one_d_background(
    mesh: TensorMesh,
    active: np.ndarray,
    frequencies: np.ndarray,
    stations: np.ndarray,
    normal_indices: np.ndarray,
    full_observed: np.ndarray,
    full_standard_deviation: np.ndarray,
    background_resistivity: float,
    maximum_depth_m: float = 12000.0,
    max_iter: int = 15,
) -> OneDBackgroundResult:
    """保持旧实验的一维投影、正则化、β 估计及停止规则。"""

    n_frequency, n_station = frequencies.size, stations.size
    survey = build_normal_station_survey(frequencies, stations, normal_indices)
    observed = np.asarray(full_observed).reshape(n_frequency, 2, n_station)[:, :, normal_indices].reshape(-1)
    standard_deviation = (
        np.asarray(full_standard_deviation)
        .reshape(n_frequency, 2, n_station)[:, :, normal_indices]
        .reshape(-1)
    )
    vertical_map, profile_mesh, selected_rows = _vertical_projection(
        mesh, maximum_depth_m, background_resistivity
    )
    simulation = nsem.simulation.Simulation2DElectricField(
        mesh,
        survey=survey,
        sigmaMap=maps.ExpMap(mesh) * vertical_map,
        solver=SolverLU,
        storeInnerProduct=True,
    )
    dmisfit = L2DataMisfit(
        simulation=simulation,
        data=Data(survey, dobs=observed, standard_deviation=standard_deviation),
    )
    start = np.full(profile_mesh.n_cells, np.log(1.0 / background_resistivity))
    reg = regularization.WeightedLeastSquares(
        profile_mesh,
        mapping=maps.IdentityMap(nP=profile_mesh.n_cells),
        alpha_s=1.0,
        length_scale_x=800.0,
    )
    reg.reference_model = start.copy()
    opt = optimization.ProjectedGNCG(
        maxIter=max_iter,
        maxIterLS=20,
        cg_maxiter=30,
        cg_rtol=1e-3,
        lower=np.log(1.0 / 5000.0),
        upper=np.log(1.0 / 1.0),
    )
    problem = inverse_problem.BaseInvProblem(dmisfit, reg, opt)
    inv = inversion.BaseInversion(
        problem,
        directiveList=[
            directives.BetaEstimate_ByEig(beta0_ratio=1.0),
            directives.BetaSchedule(coolingFactor=2.0, coolingRate=2),
            directives.TargetMisfit(chifact=1.0),
            directives.UpdatePreconditioner(update_every_iteration=True),
        ],
    )
    print(f"共同一维初始模型：{observed.size} 项数据，{profile_mesh.n_cells} 个深度参数", flush=True)
    started = time.perf_counter()
    profile = np.asarray(inv.run(start), dtype=float)
    elapsed = time.perf_counter() - started
    full_model = np.asarray(vertical_map * profile, dtype=float)
    return OneDBackgroundResult(
        profile=profile,
        depth_m=mesh.cell_centers_y[selected_rows],
        active_start=full_model[active],
        full_log_conductivity=full_model,
        phi_d=float(dmisfit(profile)),
        elapsed_seconds=elapsed,
    )
