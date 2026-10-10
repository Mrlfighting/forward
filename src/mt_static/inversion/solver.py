"""沿用旧八测点实验的反演网格、正则化及固定迭代求解器。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from discretize import TensorMesh
from pymatsolver import SolverLU
from simpeg import directives, inverse_problem, inversion, maps, optimization, regularization
from simpeg.electromagnetics import natural_source as nsem


@dataclass(frozen=True)
class InversionSettings:
    """集中记录四种方法必须共用的反演参数。"""

    background_resistivity: float = 100.0
    max_iter: int = 5
    cg_maxiter: int = 15
    initial_beta: float = 4.47e-7
    cooling_factor: float = 2.0
    cooling_rate: int = 2
    active_depth_m: float = 12000.0
    active_x_limit_m: float = 16000.0
    depth_weight_start_m: float = 8000.0
    depth_weight_max: float = 25.0
    lateral_weight_start_m: float = 10000.0
    lateral_weight_max: float = 50.0


@dataclass
class InversionResult:
    """保留全网格恢复模型、预测、目标函数及实际耗时。"""

    recovered_log_conductivity: np.ndarray
    predicted: np.ndarray
    phi_d: float
    elapsed_seconds: float


def build_inversion_mesh() -> TensorMesh:
    """构造旧实验的 80×64 粗反演网格，不解析 5 m 浅部体。"""

    core_nodes = np.arange(-10400.0, 10400.0 + 1.0, 400.0)
    pad = 400.0 * 1.3 ** np.arange(1, 15)
    hx = np.r_[pad[::-1], np.diff(core_nodes), pad]
    x0 = core_nodes[0] - pad.sum()
    depth = np.unique(np.r_[
        0.0,
        np.arange(20.0, 201.0, 20.0),
        np.arange(300.0, 1001.0, 100.0),
        np.arange(1200.0, 4001.0, 200.0),
        np.arange(4400.0, 10001.0, 400.0),
        np.arange(11000.0, 16001.0, 1000.0),
    ])
    deep = 16000.0 + np.cumsum(2000.0 * 2.0 ** np.arange(10))
    z_nodes = -np.unique(np.r_[depth, deep])[::-1]
    return TensorMesh([hx, np.diff(z_nodes)], x0=np.r_[x0, z_nodes[0]])


def active_cells(mesh: TensorMesh, settings: InversionSettings) -> np.ndarray:
    """仅反演测线中心 ±16 km、地表以下 12 km 的活动单元。"""

    centers = mesh.cell_centers
    return (
        (np.abs(centers[:, 0]) <= settings.active_x_limit_m)
        & (centers[:, 1] >= -settings.active_depth_m)
        & (centers[:, 1] <= 0.0)
    )


def build_simulation(mesh: TensorMesh, survey, active: np.ndarray, settings: InversionSettings):
    """使用活动域对数电导率参数与旧 TM 二维电场模拟器。"""

    inject = maps.InjectActiveCells(
        mesh,
        active_cells=active,
        value_inactive=np.log(1.0 / settings.background_resistivity),
    )
    sigma_map = maps.ExpMap(mesh) * inject
    return nsem.simulation.Simulation2DElectricField(
        mesh,
        survey=survey,
        sigmaMap=sigma_map,
        solver=SolverLU,
        storeInnerProduct=True,
    )


def initial_model(active: np.ndarray, settings: InversionSettings) -> np.ndarray:
    """所有方法从 100 Ω·m 均匀模型开始。"""

    return np.full(int(active.sum()), np.log(1.0 / settings.background_resistivity))


def model_regularization(
    mesh: TensorMesh, active: np.ndarray, start: np.ndarray, settings: InversionSettings
):
    """重建旧实验的水平、垂向及边缘加权最小二乘正则化。"""

    reg = regularization.WeightedLeastSquares(
        mesh,
        active_cells=active,
        mapping=maps.IdentityMap(nP=int(active.sum())),
        alpha_s=1.0,
        length_scale_x=2000.0,
        length_scale_y=1200.0,
    )
    reg.reference_model = start.copy()
    centers = mesh.cell_centers[active]
    depth_fraction = np.clip(
        (np.maximum(-centers[:, 1], 0.0) - settings.depth_weight_start_m)
        / (settings.active_depth_m - settings.depth_weight_start_m),
        0.0, 1.0,
    )
    lateral_fraction = np.clip(
        (np.abs(centers[:, 0]) - settings.lateral_weight_start_m)
        / (settings.active_x_limit_m - settings.lateral_weight_start_m),
        0.0, 1.0,
    )
    reg.set_weights(
        deep_background=1.0 + (settings.depth_weight_max - 1.0) * depth_fraction**2,
        lateral_background=1.0 + (settings.lateral_weight_max - 1.0) * lateral_fraction**2,
    )
    return reg


def invert_fixed(
    label: str,
    dmisfit,
    prediction_simulation,
    mesh: TensorMesh,
    active: np.ndarray,
    start: np.ndarray,
    settings: InversionSettings,
) -> InversionResult:
    """以同一正则化、初始 β 与五次 GNCG 迭代反演一组方法。"""

    reg = model_regularization(mesh, active, start, settings)
    opt = optimization.ProjectedGNCG(
        maxIter=settings.max_iter,
        maxIterLS=20,
        cg_maxiter=settings.cg_maxiter,
        cg_rtol=1e-3,
        tolF=0.0,
        tolX=0.0,
        tolG=0.0,
        lower=np.log(1.0 / 5000.0),
        upper=np.log(1.0 / 1.0),
    )
    problem = inverse_problem.BaseInvProblem(dmisfit, reg, opt)
    problem.beta = settings.initial_beta
    inv = inversion.BaseInversion(
        problem,
        directiveList=[
            directives.UpdateSensitivityWeights(
                every_iteration=False,
                threshold_method="global",
                normalization_method="maximum",
            ),
            directives.BetaSchedule(
                coolingFactor=settings.cooling_factor,
                coolingRate=settings.cooling_rate,
            ),
            directives.UpdatePreconditioner(update_every_iteration=True),
        ],
    )
    print(f"开始{label}：{settings.max_iter} 次固定迭代", flush=True)
    started = time.perf_counter()
    recovered_active = np.asarray(inv.run(start.copy()), dtype=float)
    elapsed = time.perf_counter() - started
    recovered = np.full(mesh.n_cells, np.log(1.0 / settings.background_resistivity))
    recovered[active] = recovered_active
    predicted = np.asarray(prediction_simulation.dpred(recovered_active), dtype=float)
    return InversionResult(recovered, predicted, float(dmisfit(recovered_active)), elapsed)
