"""使用已保存的论文 2.1 TM 正演数据检查双浅部体识别。"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mt_static.detection import DetectionConfig, detect_static_sites
from plot_diagnostics import plot_case_diagnostics


FORWARD_DIR = PROJECT_ROOT / "results" / "section_21_forward" / "structured"
FORWARD_NPZ = FORWARD_DIR / "section_21_forward.npz"
FORWARD_SUMMARY = FORWARD_DIR / "section_21_summary.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "section_22" / "site_types.json"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "section_22" / "221_site_types"
CASE_DIRS = {
    "without_shallow": "table2_s0_no_static_negative",
    "with_shallow": "table2_s2_multiple_discrete_static_sites",
}


def file_sha256(path: Path) -> str:
    """记录所用正演文件的内容哈希。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_positions(forward_summary: dict) -> list[float]:
    """从正演摘要读取两处浅部体中心。"""

    regions = forward_summary["models"]["anomaly_regions"]
    positions = []
    for name in ("left_shallow_conductor", "right_shallow_conductor"):
        left, right = regions[name]["specified_bounds_m"]
        positions.append((float(left[0]) + float(right[0])) / 2.0)
    return positions


def save_detection(
    label: str, result, rho_a: np.ndarray, target_indices: list[int], destination: Path
) -> dict:
    """保存逐频候选与完整传播线，不定义跨频最终分类。"""

    destination.mkdir(parents=True, exist_ok=True)
    details = result.frequency_diagnostics
    np.savez_compressed(
        destination / "detection_diagnostics.npz",
        station_x_m=result.station_x_m,
        frequencies_hz=result.frequencies_hz,
        sample_x_m=details[0].sample_x_m,
        signal_profiles=np.stack([item.signal_profile for item in details]),
        wavelet_coefficients=np.stack([item.coefficients for item in details]),
        modulus_maxima=np.stack([item.modulus_maxima for item in details]),
        per_frequency_votes=result.per_frequency_votes,
        occurrence_counts=result.vote_counts,
    )
    line_details = []
    with (destination / "static_candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "frequency_hz", "candidate_x_m", "station_index", "station_x_m",
            "mapping_distance_m", "alpha_pairs",
        ])
        for item in details:
            recorded_lines = []
            for line in item.maxima_lines:
                recorded_lines.append({
                    "nodes": [asdict(node) for node in line.nodes],
                    "accepted": line.accepted,
                    "rejection_reason": line.rejection_reason,
                    "alpha_pairs": line.alpha_pairs,
                    "static_candidate": line.static_candidate,
                    "candidate_x_m": line.candidate_x_m,
                    "mapped_station_index": line.mapped_station_index,
                    "mapping_distance_m": line.mapping_distance_m,
                })
                if line.static_candidate and line.mapped_station_index is not None:
                    index = line.mapped_station_index
                    writer.writerow([
                        item.frequency_hz, line.candidate_x_m, index,
                        result.station_x_m[index], line.mapping_distance_m,
                        json.dumps(line.alpha_pairs),
                    ])
            line_details.append({"frequency_hz": item.frequency_hz, "lines": recorded_lines})
    (destination / "maxima_lines.json").write_text(
        json.dumps(line_details, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_case_diagnostics(result, rho_a, target_indices if label == "with_shallow" else [], destination)
    return {
        "candidate_station_occurrences": [
            {"station_x_m": float(result.station_x_m[index]), "frequency_count": int(result.vote_counts[index])}
            for index in np.flatnonzero(result.vote_counts)
        ],
        "target_frequency_counts": {
            str(int(result.station_x_m[index])): int(result.vote_counts[index])
            for index in target_indices
        },
        "target_frequencies_hz": {
            str(int(result.station_x_m[index])): result.frequencies_hz[result.per_frequency_votes[:, index]].tolist()
            for index in target_indices
        },
        "total_frequency_station_candidates": int(np.sum(result.per_frequency_votes)),
        "diagnostic_dir": str(destination),
    }


def main() -> None:
    """只识别已有 TM 数组，同时计算无浅部体对照。"""

    config_payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = DetectionConfig(**config_payload["detection"])
    forward_summary = json.loads(FORWARD_SUMMARY.read_text(encoding="utf-8"))
    if forward_summary["data_conventions"]["injected_static_shift"] != "none; only shallow conductivity cells differ":
        raise ValueError("正演摘要中的静态效应来源与本实验预期不符")
    with np.load(FORWARD_NPZ) as forward:
        stations = np.asarray(forward["station_x_m"], dtype=float)
        frequencies = np.asarray(forward["frequencies_hz"], dtype=float)
        target_x = target_positions(forward_summary)
        target_indices = [int(np.argmin(np.abs(stations - value))) for value in target_x]
        if any(not np.isclose(stations[index], value, atol=1e-9) for index, value in zip(target_indices, target_x)):
            raise ValueError("浅部体中心与原始测点不重合")
        responses = {
            "with_shallow": np.asarray(forward["tm_with_shallow_apparent_resistivity"], dtype=float),
            "without_shallow": np.asarray(forward["tm_without_shallow_apparent_resistivity"], dtype=float),
        }
    summary = {
        "source_forward_npz": str(FORWARD_NPZ),
        "source_forward_sha256": file_sha256(FORWARD_NPZ),
        "config_path": str(CONFIG_PATH),
        "detection_config": asdict(config),
        "mode": "TM",
        "case_dirs": CASE_DIRS,
        "comparison_scope": "仅比较新 2.1 正演数据的 S0 与 S2；S1/S3 仍是旧表 2 数据，不计算四组合并指标",
        "target_station_x_m": target_x,
        "station_count": int(stations.size),
        "frequency_count": int(frequencies.size),
        "results": {},
        "final_static_sites": None,
        "final_classification_reason": "基础方法没有定义 MT 跨频汇总规则；仅报告逐频候选",
    }
    for label, rho_a in responses.items():
        result = detect_static_sites(stations, frequencies, rho_a, config)
        destination = OUTPUT_ROOT / CASE_DIRS[label]
        record = save_detection(label, result, rho_a, target_indices, destination)
        summary["results"][label] = record
        case_summary = {
            "experiment_id": "S0" if label == "without_shallow" else "S2",
            "source_scenario": label,
            "source_forward_npz": str(FORWARD_NPZ),
            "source_forward_sha256": summary["source_forward_sha256"],
            "mode": "TM",
            "station_count": int(stations.size),
            "frequency_count": int(frequencies.size),
            "known_static_x_m": [] if label == "without_shallow" else target_x,
            "detection_config": asdict(config),
            "result": record,
            "final_static_sites": None,
            "final_classification_reason": summary["final_classification_reason"],
        }
        (destination / "detection_summary.json").write_text(
            json.dumps(case_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved comparison: {OUTPUT_ROOT / 'comparison.json'}")


if __name__ == "__main__":
    main()
