"""复现复杂层状组合模型的四组固定参数二维反演。"""

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
from mt_static.inversion.layered_metrics import layered_model_metrics
from mt_static.inversion.solver import InversionSettings


CONFIG_PATH = PROJECT_ROOT / "configs" / "section_23" / "layered_model.json"
INPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "section_23" / "232_layered_model"
OUTPUT_DIR = PROJECT_ROOT / "results" / "section_23" / "232_layered_model"
CACHE_DIR = PROJECT_ROOT / "cache" / "section_23" / "232_layered_model"


def sha256(path: Path) -> str:
    """计算正式输入和配置的内容哈希。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, content: dict) -> None:
    """以中文可读形式保存有限数值摘要。"""

    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def weighted_misfit(predicted: np.ndarray, observed: np.ndarray, std: np.ndarray) -> dict:
    """只在同一数据定义内报告标准化残差。"""

    residual = (predicted - observed) / std
    return {
        "n_data": int(residual.size),
        "sum_squared_standardized": float(residual @ residual),
        "rms_standardized": float(np.sqrt(np.mean(residual**2))),
    }


def main() -> None:
    """输入核验后运行共同一维初值下的四种数据选择方法。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    input_path = INPUT_DIR / "inputs.npz"
    manifest = json.loads((INPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if sha256(CONFIG_PATH) != manifest["config_sha256"] or sha256(input_path) != manifest["input_sha256"]:
        raise ValueError("配置或正演输入哈希与清单不一致")
    with np.load(input_path, allow_pickle=False) as saved:
        inputs = {key: saved[key] for key in saved.files}
    frequencies = inputs["frequencies_hz"]
    stations = inputs["station_x_m"]
    static_indices = inputs["static_station_indices"]
    normal_indices = inputs["normal_station_indices"]
    if not np.array_equal(stations[static_indices], config["static_station_x_m"]):
        raise AssertionError("静态点位置与配置不一致")
    if tuple(inputs["forward_mesh_x_nodes_m"].shape) != (157,):
        raise AssertionError("层状正演网格水平节点数不一致")
    full_survey = build_full_survey(frequencies, stations)
    reduced_survey, normal_from_survey = build_reduced_survey(frequencies, stations, static_indices)
    shape_survey = build_shape_survey(frequencies, stations, static_indices)
    if not np.array_equal(normal_indices, normal_from_survey):
        raise AssertionError("普通测点索引不一致")
    if (full_survey.nD, reduced_survey.nD, shape_survey.nD) != (384, 336, 46):
        raise AssertionError("完整、相位保留或形态数据量不一致")
    mesh = build_inversion_mesh()
    settings = InversionSettings(**config["inversion"])
    active = active_cells(mesh, settings)
    if tuple(mesh.shape_cells) != (80, 64) or int(active.sum()) != 3100:
        raise AssertionError("反演网格或活动域与旧图不一致")
    start = inputs["one_d_active_start"]
    if start.shape != (int(active.sum()),) or not np.all(np.isfinite(start)):
        raise AssertionError("一维共同初始模型尺寸或数值无效")
    clean = inputs["clean_full"]
    static = inputs["static_full"]
    clean_std = inputs["clean_std"]
    static_std = inputs["static_std"]
    reduced = reduce_data(static, frequencies.size, stations.size, normal_indices)
    reduced_std = reduce_data(static_std, frequencies.size, stations.size, normal_indices)
    if not np.array_equal(
        clean.reshape(frequencies.size, 2, stations.size)[:, 1, :],
        static.reshape(frequencies.size, 2, stations.size)[:, 1, :],
    ):
        raise AssertionError("旧实验静态点和普通点的相位噪声必须共享")
    clean_simulation = build_simulation(mesh, full_survey, active, settings)
    static_simulation = build_simulation(mesh, full_survey, active, settings)
    reduced_simulation = build_simulation(mesh, reduced_survey, active, settings)
    shape_full_simulation = build_simulation(mesh, full_survey, active, settings)
    shape_simulation = GLSShapeOnlySimulation(
        shape_full_simulation,
        shape_survey,
        frequencies,
        stations.size,
        static_indices,
        static,
        static_std,
    )
    shape_observed = shape_simulation.transform_observed(static)
    shape_std = shape_simulation.transform_standard_deviation(config["shape_weight"])
    scaled = static.reshape(frequencies.size, 2, stations.size).copy()
    for index, factor in zip(static_indices, (2.0, 3.0)):
        scaled[:, 0, index] *= factor
    shape_invariance = float(
        np.max(np.abs(shape_simulation.transform_observed(scaled.ravel()) - shape_observed))
    )
    if shape_invariance > 1e-10:
        raise AssertionError("shape 未消除静态点跨频常数幅值")
    report = {
        "status": "setup_validated",
        "input_sha256": manifest["input_sha256"],
        "config_sha256": manifest["config_sha256"],
        "station_x_m": stations.tolist(),
        "static_station_x_m": stations[static_indices].tolist(),
        "frequency_count": int(frequencies.size),
        "layers": config["layers"],
        "targets": config["targets"],
        "surface_bodies": config["surface_bodies"],
        "surface_body_cell_counts": manifest["surface_body_cell_counts"],
        "static_shift_ratios": manifest["static_shift_ratios"],
        "noise": config["noise"],
        "rho_relative_std": manifest["rho_relative_std"],
        "phase_std_deg": manifest["phase_std_deg"],
        "shape_weight": config["shape_weight"],
        "shape_amplitude_invariance_max_abs": shape_invariance,
        "data_counts": {
            "full": 384,
            "phase_preserving": 336,
            "shape": 46,
            "combined": 382,
        },
        "one_d_start": {
            "source_normal_station_x_m": stations[normal_indices].tolist(),
            "n_data": manifest["one_d_data_count"],
            "n_parameters": manifest["one_d_parameter_count"],
            "phi_d": manifest["one_d_phi_d"],
        },
        "inversion_mesh_shape": list(mesh.shape_cells),
        "active_cell_count": int(active.sum()),
        "inversion_settings": asdict(settings),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.setup_only:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "setup.json", report)
    clean_dmis = data_misfit.L2DataMisfit(
        simulation=clean_simulation,
        data=Data(full_survey, dobs=clean, standard_deviation=clean_std),
    )
    static_dmis = data_misfit.L2DataMisfit(
        simulation=static_simulation,
        data=Data(full_survey, dobs=static, standard_deviation=static_std),
    )
    reduced_dmis = data_misfit.L2DataMisfit(
        simulation=reduced_simulation,
        data=Data(reduced_survey, dobs=reduced, standard_deviation=reduced_std),
    )
    shape_dmis = data_misfit.L2DataMisfit(
        simulation=shape_simulation,
        data=Data(shape_survey, dobs=shape_observed, standard_deviation=shape_std),
    )
    methods = (
        ("clean_full", clean_dmis, clean_simulation),
        ("static_full", static_dmis, static_simulation),
        ("phase_preserving", reduced_dmis, reduced_simulation),
        ("phase_plus_shape", reduced_dmis + shape_dmis, shape_full_simulation),
    )
    results = {}
    for name, dmisfit, simulation in methods:
        outcome = invert_fixed(name, dmisfit, simulation, mesh, active, start, settings)
        np.savez_compressed(
            OUTPUT_DIR / f"{name}.npz",
            recovered_log_conductivity=outcome.recovered_log_conductivity,
            predicted=outcome.predicted,
        )
        evaluation = layered_model_metrics(
            mesh,
            outcome.recovered_log_conductivity,
            config["layers"],
            config["targets"],
            settings.background_resistivity,
        )
        evaluation.update({"phi_d": outcome.phi_d, "elapsed_seconds": outcome.elapsed_seconds})
        if name == "phase_plus_shape":
            recovered_active = outcome.recovered_log_conductivity[active]
            reduced_predicted = reduced_simulation.dpred(recovered_active)
            shape_predicted = shape_simulation.dpred(recovered_active)
            evaluation["phase_preserving_misfit"] = weighted_misfit(
                reduced_predicted, reduced, reduced_std
            )
            evaluation["shape_misfit"] = weighted_misfit(
                shape_predicted, shape_observed, shape_std
            )
            np.savez_compressed(
                OUTPUT_DIR / "phase_plus_shape_predictions.npz",
                reduced_prediction=reduced_predicted,
                shape_prediction=shape_predicted,
            )
        else:
            observed, std = {
                "clean_full": (clean, clean_std),
                "static_full": (static, static_std),
                "phase_preserving": (reduced, reduced_std),
            }[name]
            evaluation["data_misfit"] = weighted_misfit(outcome.predicted, observed, std)
        results[name] = evaluation
        report["results"] = results
        save_json(CACHE_DIR / "summary_partial.json", report)
        print(f"{name} 已完成：目标区 log10 RMSE={evaluation['roi_log10_rmse']:.6f}", flush=True)
    initial_full = np.full(mesh.n_cells, np.log(1.0 / settings.background_resistivity))
    initial_full[active] = start
    report["initial_model_metrics"] = layered_model_metrics(
        mesh, initial_full, config["layers"], config["targets"], settings.background_resistivity
    )
    report["clean_reference_better_than_initial"] = bool(
        results["clean_full"]["roi_log10_rmse"]
        < report["initial_model_metrics"]["roi_log10_rmse"]
    )
    report["status"] = "validated" if report["clean_reference_better_than_initial"] else "needs_review"
    save_json(OUTPUT_DIR / "summary.json", report)
    print("复杂层状组合模型反演完成", flush=True)


if __name__ == "__main__":
    main()
