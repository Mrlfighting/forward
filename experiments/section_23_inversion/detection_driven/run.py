"""三组合成模型先识别静态点，再用识别掩码运行数据选择反演。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external" / "simpeg"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from simpeg import data_misfit
from simpeg.data import Data

from mt_static.detection import DetectionConfig, detect_static_sites
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
from mt_static.inversion.one_d_background import invert_common_one_d_background
from mt_static.inversion.solver import InversionSettings, initial_model


CONFIG_PATH = ROOT / "configs/section_23/detection_driven.json"
OUTPUT_ROOT = ROOT / "results/section_23/detection_driven"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_input(case: dict) -> tuple[dict[str, np.ndarray], Path]:
    path = ROOT / case["input"]
    if sha256(path).lower() != case["input_sha256"].lower():
        raise ValueError(f"固定输入哈希不一致：{path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]).copy() for name in archive.files}
    frequencies = data["frequencies_hz"]
    stations = data["station_x_m"]
    full = data["static_full"]
    std = data["static_std"]
    if stations.ndim != 1 or stations.size != 8 or not np.all(np.diff(stations) > 0):
        raise AssertionError("识别与反演均须使用原实验同一组八个有序测点")
    if frequencies.ndim != 1 or not np.all(np.diff(frequencies) > 0):
        raise AssertionError("输入频率必须严格递增")
    if full.shape != (frequencies.size * 2 * stations.size,) or std.shape != full.shape:
        raise AssertionError("TM 数据须按频率、视电阻率/相位、测点顺序排列")
    if not np.all(np.isfinite(full)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise AssertionError("观测或标准差中有无效值")
    if np.any(full.reshape(frequencies.size, 2, stations.size)[:, 0, :] <= 0):
        raise AssertionError("视电阻率必须为正")
    return data, path


def save_detection(output: Path, result, selected: np.ndarray) -> None:
    """保存逐频剖面、小波诊断、传播线以及固定投票后的掩码。"""

    details = result.frequency_diagnostics
    np.savez_compressed(
        output / "detection.npz",
        frequencies_hz=result.frequencies_hz,
        station_x_m=result.station_x_m,
        per_frequency_votes=result.per_frequency_votes,
        vote_counts=result.vote_counts,
        selected_indices=selected,
        sample_x_m=details[0].sample_x_m,
        signal_at_stations=np.stack([item.signal_at_stations for item in details]),
        signal_profiles=np.stack([item.signal_profile for item in details]),
        wavelet_coefficients=np.stack([item.coefficients for item in details]),
        modulus_maxima=np.stack([item.modulus_maxima for item in details]),
        selected_maxima=np.stack([item.selected_maxima for item in details]),
    )
    lines = []
    for item in details:
        frequency_lines = []
        for line in item.maxima_lines:
            frequency_lines.append({
                "accepted": bool(line.accepted),
                "rejection_reason": line.rejection_reason,
                "static_candidate": bool(line.static_candidate),
                "candidate_x_m": line.candidate_x_m,
                "mapped_station_index": line.mapped_station_index,
                "mapping_distance_m": line.mapping_distance_m,
                "alpha_pairs": line.alpha_pairs,
                "nodes": [
                    {"scale_index": node.scale_index, "sample_index": node.sample_index,
                     "x_m": node.x_m, "coefficient": node.coefficient}
                    for node in line.nodes
                ],
            })
        lines.append({"frequency_hz": item.frequency_hz, "lines": frequency_lines})
    save_json(output / "maxima_lines.json", {"frequencies": lines})


def detection_metrics(detected: np.ndarray, truth: np.ndarray, n_stations: int) -> dict:
    predicted_set = set(detected.tolist())
    truth_set = set(truth.tolist())
    tp = len(predicted_set & truth_set)
    fp = len(predicted_set - truth_set)
    fn = len(truth_set - predicted_set)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "unit": "station",
        "station_count": n_stations,
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "F1": f1,
    }


def common_fit(predicted: np.ndarray, observed: np.ndarray, std: np.ndarray,
               n_frequency: int, n_station: int) -> dict:
    residual = ((predicted - observed) / std).reshape(n_frequency, 2, n_station)
    return {
        "all_site_phase_wrms": float(np.sqrt(np.mean(residual[:, 1, :] ** 2))),
        "all_site_rho_diagnostic_wrms": float(np.sqrt(np.mean(residual[:, 0, :] ** 2))),
    }


def rectangle_roi_error(mesh, recovered: np.ndarray, config: dict) -> float:
    """只在反演完成后用合成真值计算深部目标区误差。"""

    centers = mesh.cell_centers
    roi = ((np.abs(centers[:, 0]) <= 10000.0)
           & (centers[:, 1] >= -12000.0) & (centers[:, 1] <= 0.0))
    truth = np.full(mesh.n_cells, config["background_resistivity_ohm_m"])
    for body in config["deep_rectangles"]:
        half_x = body["width_m"] / 2.0
        half_z = body["height_m"] / 2.0
        inside = ((np.abs(centers[:, 0] - body["center_x_m"]) <= half_x)
                  & (np.abs(centers[:, 1] - body["center_z_m"]) <= half_z))
        truth[inside] = body["resistivity_ohm_m"]
    if not np.any(roi):
        raise AssertionError("目标评价区域为空")
    difference = -recovered[roi] / np.log(10.0) - np.log10(truth[roi])
    return float(np.sqrt(np.mean(difference**2)))


def invert_detected(case_id: str, case: dict, data: dict[str, np.ndarray],
                    selected: np.ndarray, output: Path, summary: dict) -> None:
    """只按检测掩码构造相位保留与 shape；绝不回退到真值掩码。"""

    original_config = json.loads((ROOT / case["original_config"]).read_text(encoding="utf-8"))
    settings = InversionSettings(**original_config["inversion"])
    frequencies = data["frequencies_hz"]
    stations = data["station_x_m"]
    full = data["static_full"]
    full_std = data["static_std"]
    normal = np.setdiff1d(np.arange(stations.size), selected)
    reduced = reduce_data(full, frequencies.size, stations.size, normal)
    reduced_std = reduce_data(full_std, frequencies.size, stations.size, normal)
    full_survey = build_full_survey(frequencies, stations)
    reduced_survey, survey_normal = build_reduced_survey(frequencies, stations, selected)
    shape_survey = build_shape_survey(frequencies, stations, selected)
    if not np.array_equal(normal, survey_normal):
        raise AssertionError("观测与接收器的普通点掩码不一致")
    expected = {
        "full": 2 * frequencies.size * stations.size,
        "phase_preserving": frequencies.size * (normal.size + stations.size),
        "shape": selected.size * (frequencies.size - 1),
    }
    if (full_survey.nD, reduced_survey.nD, shape_survey.nD) != tuple(expected.values()):
        raise AssertionError("检测掩码构造的数据量与接收器不一致")
    mesh = build_inversion_mesh()
    active = active_cells(mesh, settings)
    if tuple(mesh.shape_cells) != (80, 64) or int(active.sum()) != 3100:
        raise AssertionError("反演网格或活动域偏离原实验")
    if case["initial_model"] == "uniform":
        start = initial_model(active, settings)
        start_provenance = "原实验统一均匀初值"
    else:
        one_d = invert_common_one_d_background(
            mesh, active, frequencies, stations, normal, full, full_std,
            settings.background_resistivity,
            max_iter=int(original_config["one_d_preinversion_max_iter"]),
        )
        start = one_d.active_start
        start_provenance = "仅从检测为普通点的测点重算共同一维初值"
        np.savez_compressed(output / "one_d_start.npz", profile=one_d.profile,
                            depth_m=one_d.depth_m, active_start=start)
        summary["one_d_phi_d"] = one_d.phi_d
    full_simulation = build_simulation(mesh, full_survey, active, settings)
    reduced_simulation = build_simulation(mesh, reduced_survey, active, settings)
    shape_simulation = GLSShapeOnlySimulation(
        full_simulation, shape_survey, frequencies, stations.size,
        selected, full, full_std,
    )
    shape_observed = shape_simulation.transform_observed(full)
    shape_std = shape_simulation.transform_standard_deviation(original_config["shape_weight"])
    scaled = full.reshape(frequencies.size, 2, stations.size).copy()
    scaled[:, 0, selected] *= 2.0
    invariance = float(np.max(np.abs(
        shape_simulation.transform_observed(scaled.ravel()) - shape_observed
    )))
    if invariance > 1e-10:
        raise AssertionError("shape 重新引入了静态点的常数幅值")
    np.savez_compressed(
        output / "observations.npz",
        frequencies_hz=frequencies, station_x_m=stations,
        detected_static_indices=selected, ordinary_indices=normal,
        full_observed=full, full_std=full_std,
        phase_observed=reduced, phase_std=reduced_std,
        shape_observed=shape_observed, shape_std=shape_std,
        initial_active_model=start, active_cells=active,
    )
    summary["data_counts"] = {**expected, "phase_plus_shape": expected["phase_preserving"] + expected["shape"]}
    summary["inversion_settings"] = asdict(settings)
    summary["initial_model_provenance"] = start_provenance
    summary["shape_amplitude_invariance_max_abs"] = invariance
    save_json(output / "summary.json", summary)
    reduced_misfit = data_misfit.L2DataMisfit(
        simulation=reduced_simulation,
        data=Data(reduced_survey, dobs=reduced, standard_deviation=reduced_std),
    )
    shape_misfit = data_misfit.L2DataMisfit(
        simulation=shape_simulation,
        data=Data(shape_survey, dobs=shape_observed, standard_deviation=shape_std),
    )
    methods = (
        ("phase_preserving", reduced_misfit, reduced_simulation),
        ("phase_plus_shape", reduced_misfit + shape_misfit, full_simulation),
    )
    summary["status"] = "running_inversions"
    save_json(output / "summary.json", summary)
    for name, misfit, prediction_simulation in methods:
        saved_result = output / f"{name}.npz"
        if name in summary.get("inversions", {}) and saved_result.exists():
            continue
        outcome = invert_fixed(name, misfit, prediction_simulation, mesh, active, start, settings)
        np.savez_compressed(saved_result,
                            recovered_log_conductivity=outcome.recovered_log_conductivity,
                            predicted=outcome.predicted)
        predicted_full = full_simulation.dpred(outcome.recovered_log_conductivity[active])
        summary.setdefault("inversions", {})[name] = {
            "phi_d": outcome.phi_d,
            "elapsed_seconds": outcome.elapsed_seconds,
            "n_data": expected["phase_preserving"] if name == "phase_preserving"
                      else expected["phase_preserving"] + expected["shape"],
            "common_data_fit": common_fit(
                predicted_full, full, full_std, frequencies.size, stations.size,
            ),
        }
        if case_id == "231_four_sites":
            summary["inversions"][name]["roi_log10_rmse"] = rectangle_roi_error(
                mesh, outcome.recovered_log_conductivity, original_config,
            )
        save_json(output / "summary.json", summary)
    summary["status"] = "completed_with_detected_mask"
    save_json(output / "summary.json", summary)


def run_case(case_id: str, case: dict, experiment: dict, detection_config: DetectionConfig,
             detection_hash: str, detect_only: bool) -> None:
    data, source_path = load_input(case)
    frequencies = data["frequencies_hz"]
    stations = data["station_x_m"]
    rho = data["static_full"].reshape(frequencies.size, 2, stations.size)[:, 0, :]
    result = detect_static_sites(stations, frequencies, rho, detection_config)
    if not np.array_equal(result.station_x_m, stations):
        raise AssertionError("检测器改变了八个反演测点的排列")
    minimum_votes = math.ceil(experiment["minimum_frequency_fraction"] * frequencies.size)
    selected = np.flatnonzero(result.vote_counts >= minimum_votes)
    output = OUTPUT_ROOT / case_id
    output.mkdir(parents=True, exist_ok=True)
    save_detection(output, result, selected)
    # 真值只用于检测完成后的评价，不参加掩码形成或反演数据选择。
    truth = np.asarray(data["static_station_indices"], dtype=int)
    baseline = ROOT / case["oracle_result_directory"] / "static_full.npz"
    summary = {
        "status": "detection_completed",
        "case_id": case_id,
        "source_input": str(source_path.relative_to(ROOT)),
        "source_input_sha256": sha256(source_path),
        "original_config_sha256": sha256(ROOT / case["original_config"]),
        "detection_config_sha256": detection_hash,
        "station_policy": experiment["station_policy"],
        "minimum_frequency_fraction": experiment["minimum_frequency_fraction"],
        "minimum_votes": minimum_votes,
        "frequency_count": int(frequencies.size),
        "station_x_m": stations.tolist(),
        "vote_counts": result.vote_counts.astype(int).tolist(),
        "detected_indices": selected.tolist(),
        "detected_x_m": stations[selected].tolist(),
        "truth_indices_evaluation_only": truth.tolist(),
        "truth_x_m_evaluation_only": stations[truth].tolist(),
        "classification_metrics": detection_metrics(selected, truth, stations.size),
        "full_data_reference": str(baseline.relative_to(ROOT)),
        "full_data_reference_sha256": sha256(baseline),
        "phase_error_policy": (
            "沿用原实验逐项标准差；误差位置由真值设定，故本对照并非全盲实验"
            if case_id == "231_four_sites" else "沿用原实验已经保存的逐项标准差"
        ),
    }
    previous_path = output / "summary.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if (previous.get("source_input_sha256") == summary["source_input_sha256"]
                and previous.get("detection_config_sha256") == summary["detection_config_sha256"]
                and previous.get("detected_indices") == summary["detected_indices"]):
            summary["inversions"] = previous.get("inversions", {})
    save_json(previous_path, summary)
    print(f"{case_id}：票数 {summary['vote_counts']}，检测位置 {summary['detected_x_m']}", flush=True)
    if detect_only:
        if (selected.size > 0
                and set(summary.get("inversions", {})) == {"phase_preserving", "phase_plus_shape"}
                and all((output / f"{name}.npz").exists()
                        for name in ("phase_preserving", "phase_plus_shape"))):
            summary["status"] = "completed_with_detected_mask"
            save_json(previous_path, summary)
        return
    if selected.size == 0:
        summary["status"] = "no_detected_sites_inversion_not_applicable"
        summary["reason"] = "按固定20%规则无候选；相位保留会退化为全数据，shape 无可定义的候选点形态，禁止用真值回填"
        save_json(output / "summary.json", summary)
        return
    invert_detected(case_id, case, data, selected, output, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="合成实验按正式识别结果构造反演掩码")
    parser.add_argument("--case", choices=("all", "231_two_sites", "231_four_sites", "232_layered"),
                        default="all")
    parser.add_argument("--detect-only", action="store_true")
    args = parser.parse_args()
    experiment = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if experiment["minimum_frequency_fraction"] != 0.2 or experiment["station_policy"] != "same_as_inversion":
        raise ValueError("本轮固定为反演八测点和至少20%频率票数")
    detection_path = ROOT / experiment["detection_config"]
    detection_config = DetectionConfig(**json.loads(detection_path.read_text(encoding="utf-8"))["detection"])
    case_ids = list(experiment["cases"]) if args.case == "all" else [args.case]
    for case_id in case_ids:
        run_case(case_id, experiment["cases"][case_id], experiment,
                 detection_config, sha256(detection_path), args.detect_only)


if __name__ == "__main__":
    main()
