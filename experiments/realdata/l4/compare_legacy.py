"""只读比较新旧 L4 模块在相同模型上的观测和预测。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from mt_static.realdata.l4.mesh import build_active_cells, build_mesh, build_simulation
from mt_static.realdata.l4.shape import GLSAmplitudeInvariantShapeSimulation
from mt_static.realdata.l4.solver import InversionSettings
from mt_static.realdata.l4.survey import (
    full_standard_deviation, full_survey, phase_corrected_survey, transformed_survey,
)


def load_legacy(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("l4_legacy_readonly", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("旧 L4 脚本无法读取")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def difference(name: str, new: np.ndarray, old: np.ndarray, tolerance: float) -> float:
    if new.shape != old.shape:
        raise AssertionError(f"{name} 形状不同：{new.shape} vs {old.shape}")
    value = float(np.max(np.abs(new - old)))
    if value > tolerance:
        raise AssertionError(f"{name} 新旧差异 {value:.6g} 超过 {tolerance:g}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="L4 迁移前后的同模型预测比较")
    parser.add_argument("--legacy-script", type=Path, required=True)
    args = parser.parse_args()
    legacy = load_legacy(args.legacy_script.resolve())
    prepared = np.load(ROOT / "data/realdata/l4/prepared.npz")
    setup = json.loads((ROOT / "results/realdata/l4/setup.json").read_text(encoding="utf-8"))
    settings = InversionSettings(**setup["inversion_parameters"])
    stations = prepared["stations_m"]
    frequencies = prepared["frequencies_hz"]
    rho = prepared["rho_xy_ohm_m"]
    phase = prepared["phase_simpeg_deg"]
    indices = np.asarray(setup["selected_static_indices"], dtype=int)
    old_mesh = legacy.build_mesh(stations)
    mesh = build_mesh(stations)
    checks = {
        "mesh_hx": difference("mesh_hx", mesh.h[0], old_mesh.h[0], 0.0),
        "mesh_hz": difference("mesh_hz", mesh.h[1], old_mesh.h[1], 0.0),
        "mesh_x0": difference("mesh_x0", np.asarray(mesh.x0), np.asarray(old_mesh.x0), 0.0),
    }
    active = build_active_cells(mesh, stations, settings.active_depth, settings.active_side_padding)
    old_active = legacy.build_active_cells(old_mesh, stations, settings.active_depth, settings.active_side_padding)
    if not np.array_equal(active, old_active):
        raise AssertionError("活动单元掩码不一致")
    full_obs = legacy.join_full(rho, phase)
    full_std = full_standard_deviation(
        rho, phase, settings.rho_relative_error, settings.phase_error_deg,
    )
    checks["full_observed"] = difference("full_observed", prepared["full_observed"], full_obs, 0.0)
    checks["full_std"] = difference(
        "full_std", full_std,
        legacy.full_standard_deviation(rho, phase, settings.rho_relative_error, settings.phase_error_deg), 0.0,
    )
    survey_full = full_survey(frequencies, stations)
    old_survey_full = legacy.full_survey(frequencies, stations)
    simulation = build_simulation(mesh, survey_full, active, settings.starting_rho)
    old_simulation = legacy.build_simulation(old_mesh, old_survey_full, old_active, settings.starting_rho)
    model = np.full(int(active.sum()), np.log(1.0 / settings.starting_rho))
    predicted = np.asarray(simulation.dpred(model), dtype=float)
    old_predicted = np.asarray(old_simulation.dpred(model), dtype=float)
    checks["full_prediction"] = difference("full_prediction", predicted, old_predicted, 1e-8)
    normal = np.setdiff1d(np.arange(stations.size), indices)
    phase_obs = legacy.select_regular(rho, phase, normal)
    checks["phase_observed"] = difference(
        "phase_observed", phase_obs, np.concatenate((rho[:, normal], phase), axis=1).ravel(), 0.0,
    )
    survey_phase = phase_corrected_survey(frequencies, stations, indices)
    old_survey_phase = legacy.phase_corrected_survey(frequencies, stations, indices)
    phase_simulation = build_simulation(mesh, survey_phase, active, settings.starting_rho)
    old_phase_simulation = legacy.build_simulation(old_mesh, old_survey_phase, old_active, settings.starting_rho)
    checks["phase_prediction"] = difference(
        "phase_prediction", np.asarray(phase_simulation.dpred(model)),
        np.asarray(old_phase_simulation.dpred(model)), 1e-8,
    )
    shape_survey = transformed_survey(frequencies, stations, indices)
    old_shape_survey = legacy.transformed_survey(frequencies, stations, indices)
    new_shape = GLSAmplitudeInvariantShapeSimulation(
        simulation, shape_survey, frequencies, stations.size, indices, full_obs, full_std,
    )
    old_shape = legacy.GLSAmplitudeInvariantShapeSimulation(
        old_simulation, old_shape_survey, frequencies, stations.size, indices, full_obs, full_std,
    )
    checks["shape_operator"] = difference(
        "shape_operator", new_shape.shape_operators, old_shape.shape_operators, 1e-12,
    )
    checks["shape_observed"] = difference(
        "shape_observed", new_shape.transform_observed(full_obs),
        old_shape.transform_observed(full_obs), 1e-10,
    )
    checks["shape_std"] = difference(
        "shape_std",
        new_shape.transform_standard_deviation(full_obs, full_std, settings.shape_weight),
        old_shape.transform_standard_deviation(full_obs, full_std, settings.shape_weight), 1e-12,
    )
    checks["shape_prediction"] = difference(
        "shape_prediction", np.asarray(new_shape.dpred(model)),
        np.asarray(old_shape.dpred(model)), 1e-8,
    )
    destination = ROOT / "results/realdata/l4/legacy_comparison.json"
    destination.write_text(json.dumps({"status": "pass", "max_abs_difference": checks},
                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("旧 L4 与新 L4 的网格、观测、shape 算子和同模型预测比较通过", flush=True)


if __name__ == "__main__":
    main()
