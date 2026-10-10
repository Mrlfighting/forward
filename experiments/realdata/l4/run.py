"""L4 实测数据准备、正式逐频识别与三方法反演入口。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from mt_static.realdata.l4.data import file_sha256, prepare_l4, save_prepared
from mt_static.realdata.l4.mesh import build_active_cells, build_mesh, build_simulation
from mt_static.realdata.l4.selection import identify_sites, save_detection, selected_static_indices
from mt_static.realdata.l4.shape import GLSAmplitudeInvariantShapeSimulation
from mt_static.realdata.l4.solver import InversionSettings, run_inversion
from mt_static.realdata.l4.survey import (
    full_standard_deviation, full_survey, join_full, phase_corrected_survey,
    select_regular, transformed_survey,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_legacy_prepared(prepared, legacy_dir: Path) -> dict[str, object]:
    """逐项核对旧 L4 数据准备结果；只读取旧目录。"""

    mapping = {
        "station_codes": (prepared.station_codes, "MT_station_codes.npy"),
        "stations_m": (prepared.stations_m, "MT_stations_m.npy"),
        "frequencies_hz": (prepared.frequencies_hz, "MT_frequencies_hz.npy"),
        "rho_xy_ohm_m": (prepared.rho_xy_ohm_m, "MT_rho_xy_ohm_m.npy"),
        "phase_raw_deg": (prepared.phase_raw_deg, "MT_phase_raw_deg.npy"),
        "phase_normal_deg": (prepared.phase_normal_deg, "MT_phase_normal_deg.npy"),
        "phase_simpeg_deg": (prepared.phase_simpeg_deg, "MT_phase_simpeg_xy_deg.npy"),
        "full_observed": (prepared.full_observed, "MT_data_full.npy"),
    }
    checks = {}
    for name, (new, filename) in mapping.items():
        old = np.load(legacy_dir / filename)
        checks[name] = {
            "equal": bool(np.array_equal(new, old)),
            "max_abs_difference": float(np.max(np.abs(new - old))),
        }
        if not checks[name]["equal"]:
            raise AssertionError(f"L4 准备数据与旧结果不一致：{name}")
    return checks


def validate_inputs(prepared, result, indices: np.ndarray, expected_votes: list[int]) -> None:
    if prepared.stations_m.size != 15 or prepared.frequencies_hz.size != 73:
        raise AssertionError("L4 必须为 15 测点×73 频率")
    if prepared.frequencies_hz[0] != 0.00114 or prepared.frequencies_hz[-1] != 320.0001:
        raise AssertionError("L4 频率端点与旧主流程不一致")
    if not np.array_equal(result.vote_counts, expected_votes):
        raise AssertionError(f"新识别器逐点频数发生变化：{result.vote_counts.tolist()}")
    if not np.array_equal(indices, [0, 4, 13, 14]):
        raise AssertionError("用户确认的四点掩码发生变化")


def prepare_observations(prepared, indices: np.ndarray, settings: InversionSettings,
                         mesh, active_cells) -> dict[str, object]:
    """按照旧 L4 分量排列构造三方法观测与误差。"""

    frequencies = prepared.frequencies_hz
    stations = prepared.stations_m
    rho = prepared.rho_xy_ohm_m
    phase = prepared.phase_simpeg_deg
    normal = np.setdiff1d(np.arange(stations.size), indices)
    full_obs = join_full(rho, phase)
    full_std = full_standard_deviation(
        rho, phase, settings.rho_relative_error, settings.phase_error_deg,
    )
    phase_obs = select_regular(rho, phase, normal)
    phase_std = select_regular(
        settings.rho_relative_error * np.abs(rho),
        np.full_like(phase, settings.phase_error_deg), normal,
    )
    survey_full = full_survey(frequencies, stations)
    survey_phase = phase_corrected_survey(frequencies, stations, indices)
    survey_shape = transformed_survey(frequencies, stations, indices)
    simulation_full = build_simulation(mesh, survey_full, active_cells, settings.starting_rho)
    shape_simulation = GLSAmplitudeInvariantShapeSimulation(
        simulation_full, survey_shape, frequencies, stations.size, indices,
        full_obs, full_std,
    )
    shape_obs = shape_simulation.transform_observed(full_obs)
    shape_std = shape_simulation.transform_standard_deviation(
        full_obs, full_std, shape_weight=settings.shape_weight,
    )
    expected = {
        "full": frequencies.size * 2 * stations.size,
        "phase": frequencies.size * (normal.size + stations.size),
        "shape": frequencies.size * (normal.size + stations.size)
                 + (frequencies.size - 1) * indices.size,
    }
    actual = {"full": full_obs.size, "phase": phase_obs.size, "shape": shape_obs.size}
    if actual != expected or expected != {"full": 2190, "phase": 1898, "shape": 2186}:
        raise AssertionError(f"三方法数据量不一致：{actual}")
    if survey_full.nD != actual["full"] or survey_phase.nD != actual["phase"] or survey_shape.nD != actual["shape"]:
        raise AssertionError("接收器数量与观测向量不一致")
    if not np.array_equal(phase_obs, shape_obs[:phase_obs.size]):
        raise AssertionError("形态法没有保留相位基线数据")
    operators = shape_simulation.shape_operators
    invariance = float(np.max(np.abs(operators @ np.ones(frequencies.size))))
    if invariance > 1e-10:
        raise AssertionError("形态算子重新引入了常数对数幅值")
    return {
        "normal_indices": normal,
        "counts": actual,
        "shape_invariance_error": invariance,
        "observed": {"full": full_obs, "phase": phase_obs, "shape": shape_obs},
        "standard_deviation": {"full": full_std, "phase": phase_std, "shape": shape_std},
        "survey": {"full": survey_full, "phase": survey_phase, "shape": survey_shape},
        "full_simulation": simulation_full,
        "shape_simulation": shape_simulation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="L4 实测识别与三方法反演")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/realdata/l4.json")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--methods", default="full,phase,shape")
    parser.add_argument("--legacy-prepared-dir", type=Path)
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods or len(methods) != len(set(methods)) or not set(methods) <= {"full", "phase", "shape"}:
        parser.error("methods 仅可包含 full、phase、shape，且不可重复")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = InversionSettings(**config["inversion"])
    raw_dir = ROOT / config["raw_directory"]
    prepared = prepare_l4(raw_dir, config["minimum_frequency_hz"], config["station_spacing_m"])
    data_root = ROOT / "data/realdata/l4"
    result_root = ROOT / "results/realdata/l4"
    result_root.mkdir(parents=True, exist_ok=True)
    save_prepared(data_root / "prepared.npz", prepared)
    manifest = {
        "config_sha256": file_sha256(config_path),
        "raw_sha256": prepared.source_sha256,
        "station_count": int(prepared.stations_m.size),
        "frequency_count": int(prepared.frequencies_hz.size),
        "frequency_range_hz": [float(prepared.frequencies_hz[0]), float(prepared.frequencies_hz[-1])],
        "component": "xy",
        "data_order": "frequency, component[rho_a, phase], station",
        "frequency_interpolation": False,
    }
    if args.legacy_prepared_dir is not None:
        manifest["legacy_prepared_checks"] = check_legacy_prepared(
            prepared, args.legacy_prepared_dir.resolve(),
        )
    write_json(data_root / "manifest.json", manifest)
    detected = identify_sites(prepared, config["detection"])
    static_indices = selected_static_indices(
        detected, config["selected_static_station_ids"],
    )
    validate_inputs(prepared, detected, static_indices, config["expected_vote_counts"])
    save_detection(result_root / "identification", detected, static_indices)
    mesh = build_mesh(prepared.stations_m)
    active = build_active_cells(
        mesh, prepared.stations_m, settings.active_depth, settings.active_side_padding,
    )
    if tuple(mesh.shape_cells) != (82, 34) or int(active.sum()) != 1736:
        raise AssertionError("L4 旧网格或活动域尺寸发生变化")
    observations = prepare_observations(prepared, static_indices, settings, mesh, active)
    inversion_dir = result_root / "inversion"
    inversion_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        inversion_dir / "mesh.npz", hx=mesh.h[0], hz=mesh.h[1],
        x0=np.asarray(mesh.x0), active_cells=active,
    )
    np.savez_compressed(
        inversion_dir / "observations.npz",
        full_observed=observations["observed"]["full"],
        full_standard_deviation=observations["standard_deviation"]["full"],
        phase_observed=observations["observed"]["phase"],
        phase_standard_deviation=observations["standard_deviation"]["phase"],
        shape_observed=observations["observed"]["shape"],
        shape_standard_deviation=observations["standard_deviation"]["shape"],
    )
    setup = {
        "status": "setup_passed",
        "data_manifest_sha256": file_sha256(data_root / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "stations_m": prepared.stations_m.tolist(),
        "station_codes": prepared.station_codes.tolist(),
        "frequencies_hz": prepared.frequencies_hz.tolist(),
        "vote_counts": detected.vote_counts.astype(int).tolist(),
        "selected_static_station_ids": config["selected_static_station_ids"],
        "selected_static_indices": static_indices.tolist(),
        "selected_static_station_provenance": config["selected_static_station_provenance"],
        "mesh_shape_cells": list(mesh.shape_cells),
        "mesh_active_cells": int(active.sum()),
        "data_counts": observations["counts"],
        "shape_invariance_error": observations["shape_invariance_error"],
        "inversion_parameters": asdict(settings),
        "detection_config": config["detection"],
    }
    write_json(result_root / "setup.json", setup)
    print("L4 输入、逐频识别、四点掩码、网格和 shape 关卡通过", flush=True)
    if args.check_only:
        return
    start = np.full(int(active.sum()), np.log(1.0 / settings.starting_rho))
    summary_path = result_root / "summary.json"
    summary = {
        "status": "running", "setup_sha256": file_sha256(result_root / "setup.json"),
        "methods": {},
        "comparison_note": "四点掩码不同于旧 L4 五点结果；不可直接比较旧图或单独按 WRMS 排序",
    }
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("setup_sha256") == summary["setup_sha256"]:
            summary["methods"] = previous.get("methods", {})
    for method in methods:
        output = inversion_dir / f"{method}.npz"
        if method in summary["methods"] and output.exists():
            print(f"{method} 已完成，跳过重复反演", flush=True)
            continue
        if method == "full":
            simulation = observations["full_simulation"]
        elif method == "phase":
            simulation = build_simulation(
                mesh, observations["survey"]["phase"], active, settings.starting_rho,
            )
        else:
            simulation = observations["shape_simulation"]
        result = run_inversion(
            method, observations["observed"][method],
            observations["standard_deviation"][method], simulation,
            mesh, active, start, settings,
        )
        full_prediction = result.predicted if method == "full" else np.asarray(
            observations["full_simulation"].dpred(result.recovered_active), dtype=float,
        )
        residual = (result.predicted - observations["observed"][method]) / observations["standard_deviation"][method]
        np.savez_compressed(
            output,
            recovered_active=result.recovered_active,
            recovered_full=result.recovered_full,
            predicted=result.predicted,
            predicted_full=full_prediction,
            normalized_residual=residual,
        )
        summary["methods"][method] = {
            "n_data": int(result.predicted.size),
            "phi_d": result.phi_d,
            "phi_d_per_datum": float(result.phi_d / result.predicted.size),
            "wrms": float(np.sqrt(np.mean(residual**2))),
            "iterations": result.iterations,
            "elapsed_seconds": result.elapsed_seconds,
        }
        predicted_blocks = full_prediction.reshape(prepared.frequencies_hz.size, 2, prepared.stations_m.size)
        ordinary = observations["normal_indices"]
        rho_residual = (
            predicted_blocks[:, 0, :] - prepared.rho_xy_ohm_m
        ) / (settings.rho_relative_error * prepared.rho_xy_ohm_m)
        phase_residual = (
            predicted_blocks[:, 1, :] - prepared.phase_simpeg_deg
        ) / settings.phase_error_deg
        summary["methods"][method]["common_data_fit"] = {
            "all_site_phase_wrms": float(np.sqrt(np.mean(phase_residual**2))),
            "ordinary_site_rho_wrms": float(np.sqrt(np.mean(rho_residual[:, ordinary]**2))),
            "candidate_site_rho_diagnostic_wrms": float(np.sqrt(np.mean(rho_residual[:, static_indices]**2))),
            "candidate_rho_is_fitted": method == "full",
        }
        if method == "shape":
            shape_residual = residual[observations["shape_simulation"].n_regular:]
            summary["methods"][method]["shape_residual_rms"] = float(np.sqrt(np.mean(shape_residual**2)))
            summary["methods"][method]["shape_residual_max_abs"] = float(np.max(np.abs(shape_residual)))
        write_json(summary_path, summary)
    if set(summary["methods"]) == {"full", "phase", "shape"}:
        summary["status"] = "completed_needs_interpretation"
    write_json(summary_path, summary)
    print(f"当前结果状态：{summary['status']}", flush=True)


if __name__ == "__main__":
    main()
