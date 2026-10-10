"""使用旧图对应的四静态点矩形模型复现论文 2.3.1。"""

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
    active_cells,
    build_full_survey,
    build_inversion_mesh,
    build_reduced_survey,
    build_shape_survey,
    build_simulation,
    invert_fixed,
    reduce_data,
)
from mt_static.inversion.solver import InversionSettings, initial_model


CONFIG_PATH = PROJECT_ROOT / "configs" / "section_23" / "phase_noise_four_static_rectangles.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "section_23" / "231_phase_noise" / "four_static_rectangles_v1"


def file_sha256(path: Path) -> str:
    """计算固定输入文件的内容哈希。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    """保存不含非有限浮点数的可审计结果。"""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def model_metrics(mesh, log_conductivity: np.ndarray, rectangles: list[dict]) -> dict:
    """按照图中两个深部矩形的真实范围评价恢复模型。"""

    centers = mesh.cell_centers
    recovered = np.exp(-log_conductivity)
    true_rho = np.full(mesh.n_cells, 100.0)
    masks = []
    for rectangle in rectangles:
        x0 = rectangle["center_x_m"]
        z0 = rectangle["center_z_m"]
        hx = rectangle["width_m"] / 2.0
        hz = rectangle["height_m"] / 2.0
        mask = (
            (centers[:, 0] >= x0 - hx) & (centers[:, 0] <= x0 + hx)
            & (centers[:, 1] >= z0 - hz) & (centers[:, 1] <= z0 + hz)
        )
        if not np.any(mask):
            raise AssertionError("深部目标没有对应的反演网格单元")
        true_rho[mask] = rectangle["resistivity_ohm_m"]
        masks.append(mask)
    roi = (np.abs(centers[:, 0]) <= 10000.0) & (centers[:, 1] >= -12000.0) & (centers[:, 1] <= 0.0)
    background = roi & ~masks[0] & ~masks[1]
    below = (np.abs(centers[:, 0]) <= 8000.0) & (centers[:, 1] >= -12000.0) & (centers[:, 1] < -6600.0)

    def rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean((np.log10(recovered[mask]) - np.log10(true_rho[mask])) ** 2)))

    def geometric_mean(mask: np.ndarray) -> float:
        return float(np.exp(np.mean(np.log(recovered[mask]))))

    return {
        "roi_log10_rmse": rmse(roi),
        "background_log10_rmse": rmse(background),
        "left_geometric_mean_rho": geometric_mean(masks[0]),
        "right_geometric_mean_rho": geometric_mean(masks[1]),
        "below_targets_geometric_mean_rho": geometric_mean(below),
    }


def weighted_misfit(prediction: np.ndarray, observation: np.ndarray, std: np.ndarray) -> dict:
    """仅报告当前方法所使用数据的标准化残差。"""

    residual = (prediction - observation) / std
    return {
        "n_data": int(residual.size),
        "sum_squared_standardized": float(residual @ residual),
        "rms_standardized": float(np.sqrt(np.mean(residual**2))),
    }


def read_setup() -> dict:
    """读取经哈希锁定的旧正演数组，并严格复现旧图加噪方式。"""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    directory = PROJECT_ROOT / config["source_directory"]
    for filename, expected in config["source_sha256"].items():
        if file_sha256(directory / filename) != expected:
            raise ValueError(f"旧图正演数据哈希不一致：{filename}")
    frequencies = np.load(directory / "frequencies_hz.npy", allow_pickle=False)
    stations = np.load(directory / "station_x_m.npy", allow_pickle=False)
    clean = np.load(directory / "clean_full.npy", allow_pickle=False)
    static = np.load(directory / "static_full.npy", allow_pickle=False)
    true_clean = np.load(directory / "true_clean_log_conductivity.npy", allow_pickle=False)
    true_static = np.load(directory / "true_static_log_conductivity.npy", allow_pickle=False)
    if frequencies.size != 24 or stations.size != 8 or clean.size != 384 or static.size != 384:
        raise AssertionError("旧图必须为 24 频、8 测点、384 项全数据")
    if not np.array_equal(stations, config["station_x_m"]):
        raise AssertionError("旧图八测点位置与配置不一致")
    static_indices = np.searchsorted(stations, config["static_station_x_m"])
    if not np.array_equal(stations[static_indices], config["static_station_x_m"]):
        raise AssertionError("四个静态测点未在八点布局中精确找到")
    if np.any(clean.reshape(24, 2, 8)[:, 0] <= 0) or np.any(static.reshape(24, 2, 8)[:, 0] <= 0):
        raise AssertionError("视电阻率中存在非正值")
    if not np.array_equal(clean.reshape(24, 2, 8)[:, 1], static.reshape(24, 2, 8)[:, 1]):
        raise AssertionError("旧图输入应当使用无浅部体相位作为静态数据的初始相位")
    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(static)):
        raise AssertionError("旧图输入存在非有限值")

    frequency_rng = np.random.default_rng(config["frequency_seed"])
    frequency_indices = np.sort(frequency_rng.choice(np.arange(24), size=config["interference_count"], replace=False))
    if not np.array_equal(frequency_indices, [0, 14, 16, 19, 21, 22]):
        raise AssertionError("旧图随机频点索引未复现")
    noise_rng = np.random.default_rng(config["frequency_seed"] + config["noise_seed_offset"])
    noise = noise_rng.normal(0.0, config["phase_noise_sigma_deg"], size=(6, 4))
    corrupted = static.reshape(24, 2, 8).copy()
    frequency_mask = np.zeros(24, dtype=bool)
    frequency_mask[frequency_indices] = True
    corrupted[np.ix_(frequency_mask, [1], static_indices)] += noise[:, None, :]
    corrupted = corrupted.ravel()
    noise_mask = np.zeros((24, 8), dtype=bool)
    noise_mask[np.ix_(frequency_mask, static_indices)] = True
    differences = (corrupted - static).reshape(24, 2, 8)
    if np.any(differences[:, 0, :] != 0) or np.any(differences[:, 1, :][~noise_mask] != 0):
        raise AssertionError("指定四点六频以外的数据被改变")
    if not np.allclose(differences[:, 1, :][noise_mask], noise.ravel(), atol=1e-13, rtol=0):
        raise AssertionError("噪声样本与数据排列不一致")

    relative_rho_error = float(np.hypot(config["rho_error_floor"], config["synthetic_relative_rho_noise"]))
    base_phase_std = float(np.hypot(config["phase_error_floor_deg"], config["phase_error_floor_deg"]))
    clean_std = np.empty((24, 2, 8), dtype=float)
    clean_std[:, 0, :] = relative_rho_error * clean.reshape(24, 2, 8)[:, 0, :]
    clean_std[:, 1, :] = base_phase_std
    static_std = np.empty((24, 2, 8), dtype=float)
    static_std[:, 0, :] = relative_rho_error * corrupted.reshape(24, 2, 8)[:, 0, :]
    static_std[:, 1, :] = base_phase_std
    static_std[:, 1, static_indices] = np.hypot(base_phase_std, config["static_phase_downweight_std_deg"])
    normal_indices = np.setdiff1d(np.arange(stations.size), static_indices)
    reduced = reduce_data(corrupted, 24, 8, normal_indices)
    reduced_std = reduce_data(static_std.ravel(), 24, 8, normal_indices)
    if reduced.size != 288:
        raise AssertionError("四静态点的相位保留数据量应为 288")
    return {
        "config": config,
        "frequencies": frequencies,
        "stations": stations,
        "static_indices": static_indices,
        "normal_indices": normal_indices,
        "clean": clean,
        "static": static,
        "corrupted": corrupted,
        "clean_std": clean_std.ravel(),
        "static_std": static_std.ravel(),
        "reduced": reduced,
        "reduced_std": reduced_std,
        "true_clean": true_clean,
        "true_static": true_static,
        "frequency_indices": frequency_indices,
        "noise": noise,
        "noise_mask": noise_mask,
        "phase_rms": float(np.sqrt(np.mean(noise**2))),
        "relative_rho_error": relative_rho_error,
        "base_phase_std": base_phase_std,
    }


def main() -> None:
    """先验收旧图输入，再用正式模块运行四种固定参数反演。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup = read_setup()
    config = setup["config"]
    frequencies = setup["frequencies"]
    stations = setup["stations"]
    static_indices = setup["static_indices"]
    full_survey = build_full_survey(frequencies, stations)
    reduced_survey, normal_indices = build_reduced_survey(frequencies, stations, static_indices)
    shape_survey = build_shape_survey(frequencies, stations, static_indices)
    if not np.array_equal(normal_indices, setup["normal_indices"]):
        raise AssertionError("普通测点索引与相位保留数据不一致")
    if (full_survey.nD, reduced_survey.nD, shape_survey.nD) != (384, 288, 92):
        raise AssertionError("四静态点数据量与旧图不一致")
    mesh = build_inversion_mesh()
    settings = InversionSettings(**config["inversion"])
    active = active_cells(mesh, settings)
    if tuple(mesh.shape_cells) != (80, 64) or int(active.sum()) != 3100:
        raise AssertionError("反演网格或活动单元数与旧图不一致")
    start = initial_model(active, settings)
    clean_simulation = build_simulation(mesh, full_survey, active, settings)
    static_simulation = build_simulation(mesh, full_survey, active, settings)
    shape_full_simulation = build_simulation(mesh, full_survey, active, settings)
    shape_simulation = GLSShapeOnlySimulation(
        shape_full_simulation, shape_survey, frequencies, stations.size, static_indices,
        setup["corrupted"], setup["static_std"],
    )
    shape_observed = shape_simulation.transform_observed(setup["corrupted"])
    shape_std = shape_simulation.transform_standard_deviation(config["shape_weight"])
    scaled = setup["corrupted"].reshape(24, 2, 8).copy()
    for index, factor in zip(static_indices, [2.0, 3.0, 4.0, 5.0]):
        scaled[:, 0, index] *= factor
    invariance_error = float(np.max(np.abs(shape_simulation.transform_observed(scaled.ravel()) - shape_observed)))
    if invariance_error > 1e-10:
        raise AssertionError("shape 数据未消除静态点跨频常数幅值")
    report = {
        "source_sha256": config["source_sha256"],
        "config_sha256": file_sha256(CONFIG_PATH),
        "station_x_m": stations.tolist(),
        "static_station_x_m": stations[static_indices].tolist(),
        "normal_station_x_m": stations[normal_indices].tolist(),
        "frequency_count": int(frequencies.size),
        "noise_frequency_indices": setup["frequency_indices"].tolist(),
        "noise_frequencies_hz": frequencies[setup["frequency_indices"]].tolist(),
        "phase_noise_sigma_deg": config["phase_noise_sigma_deg"],
        "actual_phase_rms_deg": setup["phase_rms"],
        "noise_data_count": int(setup["noise_mask"].sum()),
        "rho_relative_error": setup["relative_rho_error"],
        "phase_base_std_deg": setup["base_phase_std"],
        "static_all_frequency_phase_std_deg": float(np.hypot(setup["base_phase_std"], config["static_phase_downweight_std_deg"])),
        "shape_weight": config["shape_weight"],
        "shape_amplitude_invariance_max_abs": invariance_error,
        "data_counts": {"full": 384, "phase_preserving": 288, "shape": 92, "combined": 380},
        "deep_rectangles": config["deep_rectangles"],
        "shallow_rectangles": config["shallow_rectangles"],
        "historical_static_phase_policy": config["historical_static_phase_policy"],
        "inversion_mesh_shape": list(mesh.shape_cells),
        "active_cell_count": int(active.sum()),
        "inversion_settings": asdict(settings),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.setup_only:
        return
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"结果目录已有文件，拒绝覆盖：{output}")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "setup.json", report)
    np.savez_compressed(
        output / "input.npz",
        frequencies_hz=frequencies,
        station_x_m=stations,
        static_station_indices=static_indices,
        ordinary_station_indices=normal_indices,
        clean_full=setup["clean"],
        static_full_before_interference=setup["static"],
        static_full=setup["corrupted"],
        clean_std=setup["clean_std"],
        static_std=setup["static_std"],
        phase_preserving=setup["reduced"],
        phase_preserving_std=setup["reduced_std"],
        phase_noise_deg=setup["noise"],
        phase_noise_mask=setup["noise_mask"],
        shape_observed=shape_observed,
        shape_std=shape_std,
        mesh_x_nodes_m=mesh.nodes_x,
        mesh_z_nodes_m=mesh.nodes_y,
        active_cells=active,
        true_clean_log_conductivity=setup["true_clean"],
        true_static_log_conductivity=setup["true_static"],
    )
    clean_dmis = data_misfit.L2DataMisfit(
        simulation=clean_simulation,
        data=Data(full_survey, dobs=setup["clean"], standard_deviation=setup["clean_std"]),
    )
    static_dmis = data_misfit.L2DataMisfit(
        simulation=static_simulation,
        data=Data(full_survey, dobs=setup["corrupted"], standard_deviation=setup["static_std"]),
    )
    reduced_simulation = build_simulation(mesh, reduced_survey, active, settings)
    reduced_dmis = data_misfit.L2DataMisfit(
        simulation=reduced_simulation,
        data=Data(reduced_survey, dobs=setup["reduced"], standard_deviation=setup["reduced_std"]),
    )
    shape_dmis = data_misfit.L2DataMisfit(
        simulation=shape_simulation,
        data=Data(shape_survey, dobs=shape_observed, standard_deviation=shape_std),
    )
    method_definitions = (
        ("clean_full", clean_dmis, clean_simulation),
        ("static_full", static_dmis, static_simulation),
        ("phase_preserving", reduced_dmis, reduced_simulation),
        ("phase_plus_shape", reduced_dmis + shape_dmis, shape_full_simulation),
    )
    results = {}
    for name, dmisfit, simulation in method_definitions:
        result = invert_fixed(name, dmisfit, simulation, mesh, active, start, settings)
        np.savez_compressed(
            output / f"{name}.npz",
            recovered_log_conductivity=result.recovered_log_conductivity,
            predicted=result.predicted,
        )
        evaluation = model_metrics(mesh, result.recovered_log_conductivity, config["deep_rectangles"])
        evaluation.update({"phi_d": result.phi_d, "elapsed_seconds": result.elapsed_seconds})
        if name == "phase_plus_shape":
            recovered_active = result.recovered_log_conductivity[active]
            reduced_prediction = reduced_simulation.dpred(recovered_active)
            shape_prediction = shape_simulation.dpred(recovered_active)
            evaluation["phase_preserving_misfit"] = weighted_misfit(
                reduced_prediction, setup["reduced"], setup["reduced_std"]
            )
            evaluation["shape_misfit"] = weighted_misfit(shape_prediction, shape_observed, shape_std)
            np.savez_compressed(
                output / "phase_plus_shape_predictions.npz",
                reduced_prediction=reduced_prediction,
                shape_prediction=shape_prediction,
            )
        else:
            observed, std = {
                "clean_full": (setup["clean"], setup["clean_std"]),
                "static_full": (setup["corrupted"], setup["static_std"]),
                "phase_preserving": (setup["reduced"], setup["reduced_std"]),
            }[name]
            evaluation["data_misfit"] = weighted_misfit(result.predicted, observed, std)
        results[name] = evaluation
        report["results"] = results
        save_json(output / "summary_partial.json", report)
        print(f"{name} 已完成：目标区 RMSE={evaluation['roi_log10_rmse']:.6f}", flush=True)
    report["initial_model_metrics"] = model_metrics(
        mesh, np.full(mesh.n_cells, np.log(1.0 / settings.background_resistivity)), config["deep_rectangles"]
    )
    report["clean_reference_better_than_initial"] = bool(
        results["clean_full"]["roi_log10_rmse"] < report["initial_model_metrics"]["roi_log10_rmse"]
    )
    report["status"] = "validated" if report["clean_reference_better_than_initial"] else "needs_review"
    save_json(output / "summary.json", report)
    print(f"四静态点结果状态：{report['status']}", flush=True)


if __name__ == "__main__":
    main()
