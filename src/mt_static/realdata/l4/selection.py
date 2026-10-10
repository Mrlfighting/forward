"""调用正式逐频识别器，并保留人工确认的反演掩码。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mt_static.detection import DetectionConfig, detect_static_sites

from .data import PreparedL4


def identify_sites(prepared: PreparedL4, settings: dict):
    station_ids = [f"S{index + 1:02d}" for index in range(prepared.stations_m.size)]
    return detect_static_sites(
        prepared.stations_m, prepared.frequencies_hz, prepared.rho_xy_ohm_m,
        DetectionConfig(**settings), station_ids=station_ids,
    )


def selected_static_indices(result, station_ids: list[str]) -> np.ndarray:
    """只使用用户确认的名单，不从逐频候选暗中产生最终分类。"""

    if len(station_ids) != len(set(station_ids)):
        raise ValueError("人工确认的测点编号重复")
    missing = sorted(set(station_ids) - set(result.station_ids))
    if missing:
        raise ValueError(f"测点编号不存在：{missing}")
    indices = np.asarray([result.station_ids.index(name) for name in station_ids], dtype=int)
    if np.any(result.vote_counts[indices] == 0):
        raise ValueError("所选测点必须至少在一个频率出现候选")
    return np.sort(indices)


def save_detection(output_dir: Path, result, static_indices: np.ndarray) -> None:
    """保存逐频诊断及所有传播线，不省略被拒绝的候选。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    details = result.frequency_diagnostics
    np.savez_compressed(
        output_dir / "detection_diagnostics.npz",
        station_x_m=result.station_x_m,
        frequencies_hz=result.frequencies_hz,
        per_frequency_votes=result.per_frequency_votes,
        vote_counts=result.vote_counts,
        median_alpha=result.median_alpha,
        sample_x_m=details[0].sample_x_m,
        signal_at_stations=np.stack([item.signal_at_stations for item in details]),
        signal_profiles=np.stack([item.signal_profile for item in details]),
        wavelet_coefficients=np.stack([item.coefficients for item in details]),
        modulus_maxima=np.stack([item.modulus_maxima for item in details]),
        selected_maxima=np.stack([item.selected_maxima for item in details]),
        selected_static_indices=static_indices,
    )
    line_records = []
    with (output_dir / "static_candidates.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frequency_hz", "candidate_x_m", "station_id", "station_x_m", "alpha_pairs"])
        for item in details:
            lines = []
            for line in item.maxima_lines:
                row = {
                    "accepted": line.accepted,
                    "rejection_reason": line.rejection_reason,
                    "static_candidate": line.static_candidate,
                    "candidate_x_m": line.candidate_x_m,
                    "mapped_station_index": line.mapped_station_index,
                    "mapping_distance_m": line.mapping_distance_m,
                    "alpha_pairs": line.alpha_pairs,
                    "nodes": [
                        {"scale_index": node.scale_index, "sample_index": node.sample_index,
                         "x_m": node.x_m, "coefficient": node.coefficient}
                        for node in line.nodes
                    ],
                }
                lines.append(row)
                if line.static_candidate and line.mapped_station_index is not None:
                    index = line.mapped_station_index
                    writer.writerow([item.frequency_hz, line.candidate_x_m,
                                     result.station_ids[index], result.station_x_m[index],
                                     json.dumps(line.alpha_pairs)])
            line_records.append({"frequency_hz": item.frequency_hz, "lines": lines})
    (output_dir / "maxima_lines.json").write_text(
        json.dumps(line_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
