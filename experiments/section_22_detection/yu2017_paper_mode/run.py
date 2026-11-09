"""只读已保存 TM 正演，验证于生宝第二章的逐尺度识别流程。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mt_static.detection.yu2017 import Yu2017Config, detect_yu2017
from plot import plot_case


CONFIG_PATH = ROOT / "configs/section_22/yu2017_paper_mode.json"
OUTPUT_ROOT = ROOT / "results/section_22/yu2017_paper_mode"


def sha256(path: Path) -> str:
    """核对固定正演输入，防止误用更新后的数据。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_case(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按测点位置精确抽取原始正演视电阻率。"""

    source = ROOT / case["source"]
    if sha256(source).lower() != case["source_sha256"].lower():
        raise ValueError(f"正演来源 SHA256 不符：{source}")
    with np.load(source, allow_pickle=False) as archive:
        stations = np.asarray(archive["station_x_m"], dtype=float)
        frequencies = np.asarray(archive["frequencies_hz"], dtype=float)
        rho = np.asarray(archive[case["rho_key"]], dtype=float)
    if rho.shape != (frequencies.size, stations.size):
        raise AssertionError("正演数组维度与测点、频率不一致")
    if "station_subset_x_m" in case:
        requested = np.asarray(case["station_subset_x_m"], dtype=float)
        indices = np.array([int(np.argmin(np.abs(stations - x))) for x in requested])
        if not np.array_equal(stations[indices], requested):
            raise AssertionError("反演测点子集与正演测点没有精确对应")
        stations, rho = stations[indices], rho[:, indices]
    return stations, frequencies, rho


def save_json(path: Path, payload: dict | list) -> None:
    """保存可检查的中文诊断信息。"""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_case(case: dict, config: Yu2017Config, neighborhood_m: float) -> dict:
    """保存每条线的全部尺度 α 与每个负尺度对，不生成最终分类。"""

    stations, frequencies, rho = read_case(case)
    result = detect_yu2017(stations, frequencies, rho, config)
    output = OUTPUT_ROOT / case["case_id"]
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = result.diagnostics
    occupancy = np.zeros((frequencies.size, stations.size), dtype=bool)
    for frequency_index, events in enumerate(result.negative_pairs):
        for event in events:
            occupancy[frequency_index, event.nearest_station_index] = True
    np.savez_compressed(
        output / "diagnostics.npz",
        station_x_m=stations,
        frequencies_hz=frequencies,
        rho_a=rho,
        sample_x_m=diagnostics[0].sample_x_m,
        signal_profiles=np.stack([item.signal_profile for item in diagnostics]),
        wavelet_coefficients=np.stack([item.coefficients for item in diagnostics]),
        modulus_maxima=np.stack([item.modulus_maxima for item in diagnostics]),
        event_station_occupancy=occupancy,
    )
    saved_lines = []
    for detail in diagnostics:
        saved_lines.append({
            "frequency_hz": detail.frequency_hz,
            "lines": [
                {
                    "accepted": line.accepted,
                    "rejection_reason": line.rejection_reason,
                    "alpha_pairs": line.alpha_pairs,
                    "nodes": [asdict(node) for node in line.nodes],
                }
                for line in detail.maxima_lines
            ],
        })
    save_json(output / "maxima_lines.json", saved_lines)
    with (output / "negative_alpha_pairs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(result.negative_pairs[0][0])) if result.negative_pairs[0] else [
            "frequency_hz", "line_index", "smaller_scale", "larger_scale", "alpha",
            "candidate_x_m", "nearest_station_index", "nearest_station_x_m", "mapping_distance_m",
        ])
        writer.writeheader()
        for events in result.negative_pairs:
            for event in events:
                writer.writerow(asdict(event))
    targets = np.asarray(case["target_x_m"], dtype=float)
    all_events = [event for events in result.negative_pairs for event in events]
    evaluation = []
    for target in targets:
        at_station = [event for event in all_events if event.nearest_station_x_m == target]
        near_position = [event for event in all_events if abs(event.candidate_x_m - target) <= neighborhood_m]
        evaluation.append({
            "target_x_m": float(target),
            "negative_pair_count_mapped_exactly": len(at_station),
            "frequency_count_mapped_exactly": len({event.frequency_hz for event in at_station}),
            "negative_pair_count_position_within_neighborhood": len(near_position),
            "frequency_count_position_within_neighborhood": len({event.frequency_hz for event in near_position}),
        })
    outside = [
        event for event in all_events
        if targets.size == 0 or np.min(np.abs(targets - event.candidate_x_m)) > neighborhood_m
    ]
    summary = {
        "case_id": case["case_id"],
        "source": case["source"],
        "source_sha256": case["source_sha256"],
        "rho_key": case["rho_key"],
        "config": asdict(config),
        "station_count": int(stations.size),
        "frequency_count": int(frequencies.size),
        "station_x_m": stations.tolist(),
        "negative_pair_count": len(all_events),
        "frequency_count_with_negative_pair": sum(bool(events) for events in result.negative_pairs),
        "event_frequency_counts_by_nearest_station": occupancy.sum(axis=0).astype(int).tolist(),
        "evaluation_neighborhood_m": neighborhood_m,
        "evaluation_only": evaluation,
        "negative_pair_count_outside_all_target_neighborhoods": len(outside),
        "final_static_sites": None,
        "classification_reason": "于生宝第二章未定义多尺度整线合并和 MT 跨频最终测点规则",
    }
    save_json(output / "summary.json", summary)
    plot_case(result, targets, output)
    return summary


def main() -> None:
    """用固定来源和固定参数运行原文模式的逐频识别。"""

    parser = argparse.ArgumentParser(description="于生宝第二章逐尺度静态候选识别验证")
    parser.add_argument("--case", action="append", help="仅运行指定场景，可重复传入")
    args = parser.parse_args()
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = Yu2017Config(**payload["detection"])
    chosen = set(args.case or [case["case_id"] for case in payload["cases"]])
    known = {case["case_id"] for case in payload["cases"]}
    if not chosen <= known:
        raise ValueError(f"未知场景：{sorted(chosen - known)}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = [
        save_case(case, config, float(payload["evaluation_neighborhood_m"]))
        for case in payload["cases"] if case["case_id"] in chosen
    ]
    save_json(OUTPUT_ROOT / "comparison.json", {
        "reference": payload["reference"],
        "scope": payload["scope"],
        "explicit_project_choices": payload["explicit_project_choices"],
        "cases": summaries,
    })
    for item in summaries:
        print(item["case_id"], "negative_pairs=", item["negative_pair_count"],
              "frequencies=", item["frequency_count_with_negative_pair"])


if __name__ == "__main__":
    main()
