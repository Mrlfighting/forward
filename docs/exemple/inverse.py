"""教学案例独立实现的单方法视电阻率与相位联合反演。"""

from __future__ import annotations

import time

import numpy as np
from pymatsolver import SolverLU
from simpeg import data_misfit, directives, inverse_problem, inversion, maps, optimization, regularization
from simpeg.data import Data
from simpeg.electromagnetics import natural_source as nsem

from forward import build_rho_phase_survey
from grid_model import active_cells, anomaly_mask


def build_simulation(mesh, survey, active: np.ndarray, config: dict):
    """将活动单元对数电导率映射到全网格，再创建 TM 模拟器。"""

    background = float(config["background_resistivity_ohm_m"])
    inject = maps.InjectActiveCells(
        mesh,
        active_cells=active,
        value_inactive=np.log(1.0 / background),
    )
    sigma_map = maps.ExpMap(mesh) * inject
    return nsem.simulation.Simulation2DElectricField(
        mesh,
        survey=survey,
        sigmaMap=sigma_map,
        solver=SolverLU,
        storeInnerProduct=True,
    )


def build_regularization(mesh, active: np.ndarray, start: np.ndarray, config: dict):
    """设置参考模型、方向长度尺度和边缘背景权重。"""

    parameters = config["inversion"]
    grid = config["mesh"]
    result = regularization.WeightedLeastSquares(
        mesh,
        active_cells=active,
        mapping=maps.IdentityMap(nP=int(active.sum())),
        alpha_s=float(parameters["regularization_smallness_weight"]),
        length_scale_x=float(parameters["regularization_length_scale_x_m"]),
        length_scale_y=float(parameters["regularization_length_scale_y_m"]),
    )
    result.reference_model = start.copy()
    centers = mesh.cell_centers[active]
    active_depth = float(grid["active_depth_m"])
    active_half_width = float(grid["active_half_width_m"])
    depth_start = float(parameters["depth_weight_start_m"])
    lateral_start = float(parameters["lateral_weight_start_m"])
    depth_fraction = np.clip(
        (np.maximum(-centers[:, 1], 0.0) - depth_start) / (active_depth - depth_start),
        0.0, 1.0,
    )
    lateral_fraction = np.clip(
        (np.abs(centers[:, 0]) - lateral_start) / (active_half_width - lateral_start),
        0.0, 1.0,
    )
    result.set_weights(
        deep_background=1.0 + (float(parameters["depth_weight_max"]) - 1.0) * depth_fraction**2,
        lateral_background=1.0 + (float(parameters["lateral_weight_max"]) - 1.0) * lateral_fraction**2,
    )
    return result


def solve_inverse(mesh, active: np.ndarray, start: np.ndarray, simulation, misfit, config: dict) -> dict:
    """用固定初始 β、GNCG 和 β 冷却恢复全网格模型。"""

    parameters = config["inversion"]
    reg = build_regularization(mesh, active, start, config)
    optimizer = optimization.ProjectedGNCG(
        maxIter=int(parameters["max_iter"]),
        maxIterLS=int(parameters["max_linesearch_iter"]),
        cg_maxiter=int(parameters["cg_maxiter"]),
        cg_rtol=float(parameters["cg_relative_tolerance"]),
        tolF=0.0,
        tolX=0.0,
        tolG=0.0,
        lower=np.log(1.0 / float(parameters["maximum_resistivity_ohm_m"])),
        upper=np.log(1.0 / float(parameters["minimum_resistivity_ohm_m"])),
    )
    problem = inverse_problem.BaseInvProblem(misfit, reg, optimizer)
    problem.beta = float(parameters["initial_beta"])
    runner = inversion.BaseInversion(
        problem,
        directiveList=[
            directives.UpdateSensitivityWeights(
                every_iteration=False,
                threshold_method="global",
                normalization_method="maximum",
            ),
            directives.BetaSchedule(
                coolingFactor=float(parameters["cooling_factor"]),
                coolingRate=int(parameters["cooling_rate"]),
            ),
            directives.UpdatePreconditioner(update_every_iteration=True),
        ],
    )
    print(f"开始教学案例单低阻体：{parameters['max_iter']} 次固定迭代", flush=True)
    started = time.perf_counter()
    recovered_active = np.asarray(runner.run(start.copy()), dtype=float)
    elapsed = time.perf_counter() - started
    background_log_sigma = np.log(1.0 / float(config["background_resistivity_ohm_m"]))
    recovered = np.full(mesh.n_cells, background_log_sigma)
    recovered[active] = recovered_active
    return {
        "recovered_log_conductivity": recovered,
        "predicted": np.asarray(simulation.dpred(recovered_active), dtype=float),
        "phi_d": float(misfit(recovered_active)),
        "elapsed_seconds": elapsed,
    }


def model_metrics(mesh, log_conductivity: np.ndarray, config: dict) -> dict:
    """在同一评价区域比较均匀初始模型与恢复模型。"""

    centers = mesh.cell_centers
    roi = (np.abs(centers[:, 0]) <= 4000.0) & (centers[:, 1] >= -5000.0) & (centers[:, 1] <= 0.0)
    true_rho = np.full(mesh.n_cells, config["background_resistivity_ohm_m"])
    block = anomaly_mask(mesh, config["anomaly"])
    true_rho[block] = config["anomaly"]["resistivity_ohm_m"]
    recovered_rho = np.exp(-log_conductivity)
    return {
        "roi_log10_rmse": float(np.sqrt(np.mean((np.log10(recovered_rho[roi]) - np.log10(true_rho[roi])) ** 2))),
        "anomaly_geometric_mean_resistivity_ohm_m": float(np.exp(np.mean(np.log(recovered_rho[block])))),
    }


def invert(mesh, config: dict, frequencies_hz: np.ndarray,
           stations_m: np.ndarray, observed: np.ndarray) -> dict:
    """设置无噪声合成观测的误差权重，调用本文件内的反演流程。"""

    active = active_cells(mesh, config)
    survey = build_rho_phase_survey(frequencies_hz, stations_m)
    if survey.nD != observed.size:
        raise ValueError("反演观测与 survey 数据数量不一致")
    blocks = np.asarray(observed).reshape(frequencies_hz.size, 2, stations_m.size)
    standard_deviation = np.empty_like(blocks)
    standard_deviation[:, 0, :] = config["inversion"]["rho_relative_error"] * blocks[:, 0, :]
    standard_deviation[:, 1, :] = config["inversion"]["phase_error_deg"]
    if np.any(standard_deviation <= 0):
        raise ValueError("反演数据误差必须为正")
    standard_deviation = standard_deviation.ravel()
    simulation = build_simulation(mesh, survey, active, config)
    misfit = data_misfit.L2DataMisfit(
        simulation=simulation,
        data=Data(survey, dobs=observed, standard_deviation=standard_deviation),
    )
    background_log_sigma = np.log(1.0 / float(config["background_resistivity_ohm_m"]))
    start = np.full(int(active.sum()), background_log_sigma)
    result = solve_inverse(mesh, active, start, simulation, misfit, config)
    initial_full = np.full(mesh.n_cells, background_log_sigma)
    return {
        "active_cells": active,
        "observed": observed,
        "standard_deviation": standard_deviation,
        "recovered_log_conductivity": result["recovered_log_conductivity"],
        "predicted": result["predicted"],
        "phi_d": result["phi_d"],
        "elapsed_seconds": result["elapsed_seconds"],
        "initial_metrics": model_metrics(mesh, initial_full, config),
        "recovered_metrics": model_metrics(mesh, result["recovered_log_conductivity"], config),
    }
