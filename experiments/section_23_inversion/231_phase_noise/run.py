"""运行论文 2.3.1 双体模型随机频点相位噪声反演对比。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "simpeg"))

from simpeg import data_misfit
from simpeg.data import Data

from mt_static.inversion import (
    GLSShapeOnlySimulation,
    build_full_survey,
    build_inversion_mesh,
    build_reduced_survey,
    build_shape_survey,
    build_simulation,
    join_data,
    reduce_data,
    active_cells,
    invert_fixed,
)
from mt_static.inversion.solver import InversionSettings, initial_model


CONFIG_PATH = PROJECT_ROOT / "configs" / "section_23" / "phase_noise.json"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "section_23" / "231_phase_noise"


def sha256(path: Path) -> str:
    """核对只读来源文件的内容哈希。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_setup() -> dict:
    """从新双浅部体正演提取旧八测点并构造唯一的噪声样本。"""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source_path = PROJECT_ROOT / config["source_forward"]
    if sha256(source_path).lower() != config["source_sha256"].lower():
        raise ValueError("新双体正演数据哈希与固定配置不一致")
    if config["mode"] != "TM" or config["station_x_m"] != [
        -10000.0, -8000.0, -5000.0, -2000.0, 2000.0, 5000.0, 8000.0, 10000.0
    ] or config["static_station_x_m"] != [-5000.0, 5000.0]:
        raise ValueError("场景模式、八测点或静态点位置发生未经确认的改变")
    if config["interference_count"] != 6 or config["phase_noise_sigma_deg"] != 5.0:
        raise ValueError("随机频点数或噪声标准差发生未经确认的改变")
    with np.load(source_path, allow_pickle=False) as source:
        frequencies = np.asarray(source["frequencies_hz"], dtype=float).copy()
        all_stations = np.asarray(source["station_x_m"], dtype=float).copy()
        positions = np.asarray(config["station_x_m"], dtype=float)
        station_indices = np.searchsorted(all_stations, positions)
        if station_indices.size != 8 or not np.array_equal(all_stations[station_indices], positions):
            raise ValueError("八测点未在新正演场景中精确找到")
        clean_rho = np.asarray(source["tm_without_shallow_apparent_resistivity"][:, station_indices], dtype=float)
        static_rho = np.asarray(source["tm_with_shallow_apparent_resistivity"][:, station_indices], dtype=float)
        clean_display_phase = np.asarray(source["tm_without_shallow_phase_deg"][:, station_indices], dtype=float)
        static_display_phase = np.asarray(source["tm_with_shallow_phase_deg"][:, station_indices], dtype=float)
        clean_raw = np.angle(source["tm_without_shallow_raw_impedance"][:, station_indices], deg=True)
        static_raw = np.angle(source["tm_with_shallow_raw_impedance"][:, station_indices], deg=True)
    if frequencies.size != 20 or np.any(clean_rho <= 0) or np.any(static_rho <= 0):
        raise ValueError("新场景频率数或视电阻率无效")
    if not np.allclose(clean_display_phase - 180.0, clean_raw, atol=1e-10, rtol=0):
        raise ValueError("无浅部体相位未与 SimPEG 原始角度对齐")
    if not np.allclose(static_display_phase - 180.0, static_raw, atol=1e-10, rtol=0):
        raise ValueError("双浅部体相位未与 SimPEG 原始角度对齐")
    static_indices = np.searchsorted(positions, np.asarray(config["static_station_x_m"], dtype=float))
    if not np.array_equal(positions[static_indices], config["static_station_x_m"]):
        raise ValueError("静态点索引不对应两处浅部体")
    rng_frequency = np.random.default_rng(config["frequency_seed"])
    frequency_indices = np.sort(rng_frequency.choice(
        np.arange(frequencies.size), size=config["interference_count"], replace=False,
    ))
    if not np.array_equal(frequency_indices, [0, 11, 13, 15, 17, 18]):
        raise ValueError("随机频点未复现先前确认的六个索引")
    noise_rng = np.random.default_rng(config["frequency_seed"] + config["noise_seed_offset"])
    noise = noise_rng.normal(0.0, config["phase_noise_sigma_deg"], size=(6, 8))
    clean_phase = clean_display_phase - 180.0
    static_phase = static_display_phase - 180.0
    corrupted_phase = static_phase.copy()
    corrupted_phase[frequency_indices, :] += noise
    noise_mask = np.zeros((frequencies.size, positions.size), dtype=bool)
    noise_mask[frequency_indices, :] = True
    phase_difference = corrupted_phase - static_phase
    if not np.array_equal(phase_difference[~noise_mask], np.zeros(np.count_nonzero(~noise_mask))):
        raise AssertionError("未选频点的相位被改写")
    if not np.allclose(phase_difference[noise_mask], noise.ravel(), atol=1e-13, rtol=0):
        raise AssertionError("相位噪声排列与六频八测点掩码不一致")
    rho_error = float(config["rho_relative_error"])
    phase_floor = float(config["phase_error_floor_deg"])
    clean_rho_std = rho_error * clean_rho
    static_rho_std = rho_error * static_rho
    clean_phase_std = np.full_like(clean_phase, phase_floor)
    static_phase_std = np.full_like(static_phase, phase_floor)
    static_phase_std[noise_mask] = np.hypot(phase_floor, config["phase_noise_sigma_deg"])
    clean_full = join_data(clean_rho, clean_phase)
    static_full = join_data(static_rho, corrupted_phase)
    clean_std = join_data(clean_rho_std, clean_phase_std)
    static_std = join_data(static_rho_std, static_phase_std)
    normal_indices = np.setdiff1d(np.arange(positions.size), static_indices)
    reduced = reduce_data(static_full, frequencies.size, positions.size, normal_indices)
    reduced_std = reduce_data(static_std, frequencies.size, positions.size, normal_indices)
    if clean_full.size != 320 or static_full.size != 320 or reduced.size != 280:
        raise AssertionError("完整或相位保留数据数量与八测点二十频设定不一致")
    return {
        "config": config,
        "source_path": source_path,
        "frequencies": frequencies,
        "stations": positions,
        "station_indices": station_indices,
        "static_indices": static_indices,
        "normal_indices": normal_indices,
        "clean_rho": clean_rho,
        "static_rho": static_rho,
        "clean_display_phase": clean_display_phase,
        "static_display_phase": static_display_phase,
        "clean_phase": clean_phase,
        "static_phase": static_phase,
        "corrupted_phase": corrupted_phase,
        "noise": noise,
        "noise_mask": noise_mask,
        "frequency_indices": frequency_indices,
        "clean_full": clean_full,
        "static_full": static_full,
        "clean_std": clean_std,
        "static_std": static_std,
        "reduced": reduced,
        "reduced_std": reduced_std,
        "phase_rms": float(np.sqrt(np.mean(noise**2))),
    }


def target_metrics(mesh, recovered: np.ndarray) -> dict:
    """用新双深部体真值评价目标与背景区域的对数电阻率误差。"""

    centers = mesh.cell_centers
    rho = np.exp(-recovered)
    true_rho = np.full(mesh.n_cells, 100.0)
    left = (
        (centers[:, 0] >= -8000) & (centers[:, 0] <= -2000)
        & (centers[:, 1] >= -8000) & (centers[:, 1] <= -2000)
    )
    right = (
        (centers[:, 0] >= 2000) & (centers[:, 0] <= 8000)
        & (centers[:, 1] >= -8000) & (centers[:, 1] <= -2000)
    )
    roi = (np.abs(centers[:, 0]) <= 10000) & (centers[:, 1] >= -12000) & (centers[:, 1] <= 0)
    true_rho[left] = 10.0
    true_rho[right] = 1000.0
    background = roi & ~(left | right)
    if not all(np.any(mask) for mask in (left, right, roi, background)):
        raise AssertionError("新双深部体评价区域没有活动网格单元")
    return {
        "roi_log10_rmse": float(np.sqrt(np.mean((np.log10(rho[roi]) - np.log10(true_rho[roi])) ** 2))),
        "background_log10_rmse": float(np.sqrt(np.mean((np.log10(rho[background]) - np.log10(true_rho[background])) ** 2))),
        "left_geometric_mean_rho": float(np.exp(np.mean(np.log(rho[left])))),
        "right_geometric_mean_rho": float(np.exp(np.mean(np.log(rho[right])))),
    }


def weighted_misfit(predicted: np.ndarray, observed: np.ndarray, standard_deviation: np.ndarray) -> dict:
    """仅比较同一数据子集的标准化残差，不跨方法单独以目标函数排序。"""

    residual = (np.asarray(predicted) - np.asarray(observed)) / np.asarray(standard_deviation)
    return {
        "n_data": int(residual.size),
        "rms_standardized": float(np.sqrt(np.mean(residual**2))),
        "sum_squared_standardized": float(np.dot(residual, residual)),
    }


def save_json(path: Path, payload: dict) -> None:
    """以严格 JSON 记录来源和数值，不允许非有限值。"""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    """先验证数据与 shape 算子，再做四种固定参数反演。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-only", action="store_true")
    arguments = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup = read_setup()
    frequencies = setup["frequencies"]
    stations = setup["stations"]
    static_indices = setup["static_indices"]
    full_survey = build_full_survey(frequencies, stations)
    reduced_survey, normal_indices = build_reduced_survey(frequencies, stations, static_indices)
    shape_survey = build_shape_survey(frequencies, stations, static_indices)
    if not np.array_equal(normal_indices, setup["normal_indices"]):
        raise AssertionError("survey 普通点与观测数据普通点不一致")
    if (full_survey.nD, reduced_survey.nD, shape_survey.nD) != (320, 280, 38):
        raise AssertionError("survey 数据量与预定 320/280/38 不一致")
    mesh = build_inversion_mesh()
    settings = InversionSettings(**setup["config"]["inversion"])
    active = active_cells(mesh, settings)
    if tuple(mesh.shape_cells) != (80, 64) or int(active.sum()) != 3100:
        raise AssertionError("反演网格或活动单元数量与旧程序不同")
    start = initial_model(active, settings)
    full_simulation = build_simulation(mesh, full_survey, active, settings)
    shape_simulation = GLSShapeOnlySimulation(
        full_simulation, shape_survey, frequencies, stations.size, static_indices,
        setup["static_full"], setup["static_std"],
    )
    shape_observed = shape_simulation.transform_observed(setup["static_full"])
    adjusted = setup["static_full"].reshape(frequencies.size, 2, stations.size).copy()
    adjusted[:, 0, static_indices] *= 3.7
    if not np.allclose(shape_simulation.transform_observed(adjusted.ravel()), shape_observed, atol=1e-11, rtol=0):
        raise AssertionError("跨频常数视电阻率倍率没有被 shape 算子消除")
    report = {
        "source_forward_sha256": sha256(setup["source_path"]),
        "config_sha256": sha256(CONFIG_PATH),
        "station_x_m": stations.tolist(),
        "static_station_x_m": stations[static_indices].tolist(),
        "noise_frequency_indices": setup["frequency_indices"].tolist(),
        "noise_frequencies_hz": frequencies[setup["frequency_indices"]].tolist(),
        "phase_noise_sigma_deg": setup["config"]["phase_noise_sigma_deg"],
        "actual_phase_rms_deg": setup["phase_rms"],
        "noise_count": int(setup["noise_mask"].sum()),
        "phase_std_noisy_deg": float(np.hypot(0.57, 5.0)),
        "rho_relative_error": setup["config"]["rho_relative_error"],
        "shape_weight": setup["config"]["shape_weight"],
        "data_counts": {"full": 320, "phase_preserving": 280, "shape": 38, "combined": 318},
        "inversion_mesh_shape": list(mesh.shape_cells),
        "active_cell_count": int(active.sum()),
        "inversion_settings": asdict(settings),
        "phase_convention": "saved display phase - 180 degrees = SimPEG raw phase",
        "shape_constant_amplitude_invariance_max_abs": float(
            np.max(np.abs(shape_simulation.transform_observed(adjusted.ravel()) - shape_observed))
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if arguments.setup_only:
        return
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError("验证结果目录已有内容，拒绝覆盖")
    OUTPUT_ROOT.mkdir(parents=True)
    save_json(OUTPUT_ROOT / "setup.json", report)
    np.savez_compressed(
        OUTPUT_ROOT / "input.npz",
        frequencies_hz=frequencies,
        station_x_m=stations,
        station_indices_in_forward=setup["station_indices"],
        static_station_indices=static_indices,
        clean_rho=setup["clean_rho"],
        static_rho=setup["static_rho"],
        clean_phase_display_deg=setup["clean_display_phase"],
        static_phase_display_deg=setup["static_display_phase"],
        phase_noise_deg=setup["noise"],
        phase_noise_mask=setup["noise_mask"],
        corrupted_phase_deg=setup["corrupted_phase"],
        clean_full=setup["clean_full"],
        static_full=setup["static_full"],
        clean_std=setup["clean_std"],
        static_std=setup["static_std"],
        phase_preserving=setup["reduced"],
        phase_preserving_std=setup["reduced_std"],
        shape_observed=shape_observed,
        shape_std=shape_simulation.transform_standard_deviation(setup["config"]["shape_weight"]),
        mesh_x_nodes_m=mesh.nodes_x,
        mesh_z_nodes_m=mesh.nodes_y,
        active_cells=active,
    )
    clean_dmis = data_misfit.L2DataMisfit(
        simulation=full_simulation,
        data=Data(full_survey, dobs=setup["clean_full"], standard_deviation=setup["clean_std"]),
    )
    static_dmis = data_misfit.L2DataMisfit(
        simulation=full_simulation,
        data=Data(full_survey, dobs=setup["static_full"], standard_deviation=setup["static_std"]),
    )
    reduced_simulation = build_simulation(mesh, reduced_survey, active, settings)
    reduced_dmis = data_misfit.L2DataMisfit(
        simulation=reduced_simulation,
        data=Data(reduced_survey, dobs=setup["reduced"], standard_deviation=setup["reduced_std"]),
    )
    shape_std = shape_simulation.transform_standard_deviation(setup["config"]["shape_weight"])
    shape_dmis = data_misfit.L2DataMisfit(
        simulation=shape_simulation,
        data=Data(shape_survey, dobs=shape_observed, standard_deviation=shape_std),
    )
    combined_dmis = reduced_dmis + shape_dmis
    results = {}
    method_definitions = (
        ("clean_full", clean_dmis, full_simulation),
        ("static_full", static_dmis, full_simulation),
        ("phase_preserving", reduced_dmis, reduced_simulation),
        ("phase_plus_shape", combined_dmis, full_simulation),
    )
    for name, dmisfit, prediction_simulation in method_definitions:
        result = invert_fixed(name, dmisfit, prediction_simulation, mesh, active, start, settings)
        np.savez_compressed(
            OUTPUT_ROOT / f"{name}.npz",
            recovered_log_conductivity=result.recovered_log_conductivity,
            predicted=result.predicted,
        )
        evaluation = target_metrics(mesh, result.recovered_log_conductivity)
        evaluation.update({"phi_d": result.phi_d, "elapsed_seconds": result.elapsed_seconds})
        if name == "clean_full":
            evaluation["data_misfit"] = weighted_misfit(result.predicted, setup["clean_full"], setup["clean_std"])
        elif name == "static_full":
            evaluation["data_misfit"] = weighted_misfit(result.predicted, setup["static_full"], setup["static_std"])
        elif name == "phase_preserving":
            evaluation["data_misfit"] = weighted_misfit(result.predicted, setup["reduced"], setup["reduced_std"])
        else:
            recovered_active = result.recovered_log_conductivity[active]
            reduced_prediction = reduced_simulation.dpred(recovered_active)
            shape_prediction = shape_simulation.dpred(recovered_active)
            evaluation["phase_preserving_misfit"] = weighted_misfit(
                reduced_prediction, setup["reduced"], setup["reduced_std"]
            )
            evaluation["shape_misfit"] = weighted_misfit(shape_prediction, shape_observed, shape_std)
            np.savez_compressed(
                OUTPUT_ROOT / "phase_plus_shape_predictions.npz",
                reduced_prediction=reduced_prediction,
                shape_prediction=shape_prediction,
            )
        results[name] = evaluation
        report["results"] = results
        save_json(OUTPUT_ROOT / "summary_partial.json", report)
        print(f"{name} 已完成：ROI RMSE={evaluation['roi_log10_rmse']:.4f}", flush=True)
    report["initial_model_metrics"] = target_metrics(
        mesh, np.full(mesh.n_cells, np.log(1.0 / settings.background_resistivity))
    )
    report["clean_reference_better_than_initial"] = bool(
        results["clean_full"]["roi_log10_rmse"] < report["initial_model_metrics"]["roi_log10_rmse"]
    )
    report["status"] = "validated" if report["clean_reference_better_than_initial"] else "needs_review"
    save_json(OUTPUT_ROOT / "summary.json", report)
    print(f"验证状态：{report['status']}", flush=True)


if __name__ == "__main__":
    main()
