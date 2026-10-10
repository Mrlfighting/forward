"""沿用旧 L4 正则化、初始 β 估计与 GNCG 反演。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from discretize import TensorMesh
from simpeg import directives, inverse_problem, inversion, maps, optimization, regularization
from simpeg.data import Data
from simpeg import data_misfit


@dataclass(frozen=True)
class InversionSettings:
    rho_relative_error: float = 0.10
    phase_error_deg: float = 5.0
    shape_weight: float = 0.5
    starting_rho: float = 1000.0
    max_iter: int = 20
    cg_maxiter: int = 40
    beta0_ratio: float = 1.0
    cooling_factor: float = 2.0
    cooling_rate: int = 2
    target_chifact: float = 1.0
    length_scale_x: float = 2500.0
    length_scale_z: float = 3000.0
    active_depth: float = 300000.0
    active_side_padding: float = 50000.0
    seed: int = 42


@dataclass
class InversionResult:
    label: str
    recovered_active: np.ndarray
    recovered_full: np.ndarray
    predicted: np.ndarray
    phi_d: float
    elapsed_seconds: float
    iterations: int


def run_inversion(label: str, observed: np.ndarray, standard_deviation: np.ndarray,
                  simulation, mesh: TensorMesh, active_cells: np.ndarray,
                  starting_model: np.ndarray, settings: InversionSettings) -> InversionResult:
    """三方法共用旧 L4 求解流程，不改变边界或正则化权重。"""

    data = Data(simulation.survey, dobs=observed,
                standard_deviation=standard_deviation)
    dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data)
    n_active = int(active_cells.sum())
    reg = regularization.WeightedLeastSquares(
        mesh,
        active_cells=active_cells,
        mapping=maps.IdentityMap(nP=n_active),
        alpha_s=1.0,
        length_scale_x=settings.length_scale_x,
        length_scale_y=settings.length_scale_z,
    )
    reg.reference_model = starting_model.copy()
    depth = np.maximum(-mesh.cell_centers[active_cells, 1], 0.0)
    depth_fraction = np.clip(
        (depth - 0.65 * settings.active_depth) / (0.35 * settings.active_depth),
        0.0, 1.0,
    )
    centre = 0.5 * (
        mesh.cell_centers[active_cells, 0].min()
        + mesh.cell_centers[active_cells, 0].max()
    )
    lateral = np.abs(mesh.cell_centers[active_cells, 0] - centre)
    lateral_fraction = np.clip(
        (lateral - 0.65 * settings.active_side_padding)
        / (0.35 * settings.active_side_padding), 0.0, 1.0,
    )
    reg.set_weights(
        deep_background=1.0 + 24.0 * depth_fraction**2,
        lateral_background=1.0 + 49.0 * lateral_fraction**2,
    )
    opt = optimization.ProjectedGNCG(
        maxIter=settings.max_iter,
        maxIterLS=20,
        cg_maxiter=settings.cg_maxiter,
        cg_rtol=1.0e-3,
        lower=np.log(1.0 / 10000.0),
        upper=np.log(1.0 / 1.0),
    )
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    inv = inversion.BaseInversion(
        inv_prob,
        directiveList=[
            directives.UpdateSensitivityWeights(
                every_iteration=False, threshold_method="global",
                normalization_method="maximum",
            ),
            directives.BetaEstimate_ByEig(
                beta0_ratio=settings.beta0_ratio, random_seed=settings.seed,
            ),
            directives.BetaSchedule(
                coolingFactor=settings.cooling_factor,
                coolingRate=settings.cooling_rate,
            ),
            directives.TargetMisfit(chifact=settings.target_chifact),
            directives.UpdatePreconditioner(update_every_iteration=True),
        ],
    )
    print(f"开始{label}：{observed.size} 项数据、{n_active} 个活动单元", flush=True)
    started = time.perf_counter()
    recovered_active = np.asarray(inv.run(starting_model.copy()), dtype=float)
    elapsed = time.perf_counter() - started
    predicted = np.asarray(simulation.dpred(recovered_active), dtype=float)
    recovered_full = np.full(mesh.n_cells, np.log(1.0 / settings.starting_rho))
    recovered_full[active_cells] = recovered_active
    iterations = int(getattr(opt, "iter", -1))
    result = InversionResult(
        label, recovered_active, recovered_full, predicted,
        float(dmis(recovered_active)), float(elapsed), iterations,
    )
    print(f"{label} 完成：phi_d={result.phi_d:.3f}，迭代={iterations}，耗时={elapsed:.1f}s", flush=True)
    return result
