"""读取旧 S0–S3 正演数组，运行论文定义的基础测点识别。"""

from __future__ import annotations

import csv
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


CONFIG_PATH = PROJECT_ROOT / "configs" / "section_22" / "site_types.json"
INPUT_ROOT = PROJECT_ROOT / "data" / "synthetic" / "section_22" / "221_site_types"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "section_22" / "221_site_types" / "legacy_table2_replay"


def load_case(case_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """读取旧正演的三类输入，不导入也不执行旧正演程序。"""

    source = INPUT_ROOT / case_name
    frequencies = np.asarray(np.load(source / "frequencies_8stations.npy"), dtype=float)
    stations = np.asarray(np.load(source / "stations_8stations.npy"), dtype=float)
    full = np.asarray(np.load(source / "data_static.npy"), dtype=float)
    expected = frequencies.size * 2 * stations.size
    if full.size != expected:
        raise ValueError(f"{case_name} 的输入长度应为 {expected}，实际为 {full.size}")
    rho_a = full.reshape(frequencies.size, 2, stations.size)[:, 0, :]
    setup = json.loads((source / "scenario_setup.json").read_text(encoding="utf-8"))
    return stations, frequencies, rho_a, setup


def safe_number(value: float) -> float | None:
    """将非有限浮点数转为 JSON 可表示的空值。"""

    return float(value) if np.isfinite(value) else None


def line_record(line) -> dict:
    """保留每条传播线的节点、符号、α 和淘汰原因。"""

    return {
        "nodes": [
            {**asdict(node), "modulus": node.modulus, "sign": node.sign}
            for node in line.nodes
        ],
        "accepted": bool(line.accepted),
        "rejection_reason": line.rejection_reason,
        "alpha": None if line.alpha is None else safe_number(line.alpha),
        "alpha_pairs": line.alpha_pairs,
        "static_candidate": line.static_candidate,
        "candidate_x_m": line.candidate_x_m,
        "mapped_station_index": line.mapped_station_index,
        "mapping_distance_m": line.mapping_distance_m,
    }


def classification_metrics(predicted: np.ndarray, known: np.ndarray) -> dict:
    """仅对相同粒度的布尔矩阵计算 TP、FP、FN。"""

    tp = int(np.count_nonzero(predicted & known))
    fp = int(np.count_nonzero(predicted & ~known))
    fn = int(np.count_nonzero(~predicted & known))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def save_case(case_name: str, result, setup: dict, old_summary: dict, old_detection: dict) -> dict:
    """保存紧凑数值数组、完整传播线和新旧对照摘要。"""

    output_dir = OUTPUT_ROOT / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    frequency_details = result.frequency_diagnostics
    np.savez_compressed(
        output_dir / "detection_diagnostics.npz",
        stations_m=result.station_x_m,
        station_ids=np.asarray(result.station_ids),
        frequencies_hz=result.frequencies_hz,
        sample_x_m=frequency_details[0].sample_x_m,
        signal_at_stations=np.stack([item.signal_at_stations for item in frequency_details]),
        signal_profiles=np.stack([item.signal_profile for item in frequency_details]),
        wavelet_coefficients=np.stack([item.coefficients for item in frequency_details]),
        modulus_maxima=np.stack([item.modulus_maxima for item in frequency_details]),
        selected_maxima=np.stack([item.selected_maxima for item in frequency_details]),
        per_frequency_votes=result.per_frequency_votes,
        vote_counts=result.vote_counts,
        median_alpha=result.median_alpha,
    )
    line_details = [
        {
            "frequency_hz": item.frequency_hz,
            "candidate_x_m": item.candidate_x_m,
            "candidate_station_indices": item.candidate_station_indices,
            "maxima_lines": [line_record(line) for line in item.maxima_lines],
        }
        for item in frequency_details
    ]
    (output_dir / "maxima_lines.json").write_text(
        json.dumps(line_details, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "static_candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "frequency_hz", "candidate_x_m", "station_index", "station_id",
            "station_x_m", "mapping_distance_m", "alpha_pairs",
        ])
        for item in frequency_details:
            for line in item.maxima_lines:
                if not line.static_candidate or line.mapped_station_index is None:
                    continue
                index = line.mapped_station_index
                writer.writerow([
                    item.frequency_hz, line.candidate_x_m, index,
                    result.station_ids[index], result.station_x_m[index],
                    line.mapping_distance_m, json.dumps(line.alpha_pairs),
                ])
    known = set(int(index) for index in setup["known_static_indices"])
    known_mask = np.zeros_like(result.per_frequency_votes, dtype=bool)
    known_mask[:, sorted(known)] = True
    old_votes = np.asarray(old_detection["per_frequency_votes"], dtype=bool)
    if old_votes.shape != result.per_frequency_votes.shape:
        raise ValueError(f"{case_name} 的旧逐频诊断形状与新结果不一致")
    old_alpha = {
        int(row["station_index"]): row["median_alpha"]
        for row in old_summary["score_table"]
    }
    alpha_comparison = []
    candidate_indices = set(int(index) for index in np.flatnonzero(result.vote_counts))
    for station_index in sorted(known | candidate_indices | set(old_summary["detected_static_indices"])):
        previous = old_alpha.get(station_index)
        current = safe_number(result.median_alpha[station_index])
        pair_records = [
            {"frequency_hz": item.frequency_hz, "alpha_pairs": line.alpha_pairs}
            for item in frequency_details
            for line in item.maxima_lines
            if line.static_candidate and line.mapped_station_index == station_index
        ]
        alpha_comparison.append(
            {
                "station_index": station_index,
                "station_x_m": float(result.station_x_m[station_index]),
                "old_alpha": previous,
                "new_alpha": current,
                "new_alpha_pairs_by_frequency": pair_records,
                "new_minus_old": None if previous is None or current is None else current - previous,
                "new_frequency_votes": int(result.vote_counts[station_index]),
            }
        )
    summary = {
        "experiment_id": setup["experiment_id"],
        "input_source": str((INPUT_ROOT / case_name).resolve()),
        "algorithm": "项目配置：自然对数、一阶高斯导数、五尺度传播线、逐对 alpha",
        "config": asdict(result.config),
        "known_static_indices": sorted(known),
        "known_static_x_m": [float(result.station_x_m[index]) for index in sorted(known)],
        "final_static_sites": None,
        "final_classification_reason": "参考方法只定义逐频空间曲线识别，未定义 MT 跨频汇总",
        "candidate_station_indices_by_frequency": [
            item.candidate_station_indices for item in frequency_details
        ],
        "candidate_x_m_by_frequency": [item.candidate_x_m for item in frequency_details],
        "frequency_occurrence_counts_diagnostic": result.vote_counts.astype(int).tolist(),
        "new_per_frequency_metrics": classification_metrics(result.per_frequency_votes, known_mask),
        "old_per_frequency_metrics": classification_metrics(old_votes, known_mask),
        "old_detected_static_indices": old_summary["detected_static_indices"],
        "old_true_positive": old_summary["true_positive"],
        "old_false_positive": old_summary["false_positive"],
        "old_false_negative": old_summary["false_negative"],
        "alpha_comparison": alpha_comparison,
        "diagnostics": {
            "npz": "detection_diagnostics.npz",
            "lines": "maxima_lines.json",
            "candidates": "static_candidates.csv",
            "figures": ["空间剖面诊断.png", "模极大值传播线.png", "逐频alpha诊断.png", "逐频静态候选.png"],
        },
    }
    (output_dir / "detection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def aggregate_old_station_metrics(rows: list[dict]) -> dict:
    """按四组旧场景合计测点级指标。"""

    tp = sum(row["old_true_positive"] for row in rows)
    fp = sum(row["old_false_positive"] for row in rows)
    fn = sum(row["old_false_negative"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def aggregate_frequency_metrics(rows: list[dict], key: str) -> dict:
    """合计相同频率与测点粒度的识别指标。"""

    totals = {name: sum(row[key][name] for row in rows) for name in ("tp", "fp", "fn")}
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    totals.update(
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    )
    return totals


def main() -> None:
    """仅重算 S0–S3 的识别；读取旧摘要用于对照。"""

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = DetectionConfig(**payload["detection"])
    old_root = PROJECT_ROOT / payload["old_result_root"]
    cases = payload["cases"]
    rows = []
    for case_name in cases:
        stations, frequencies, rho_a, setup = load_case(case_name)
        old_summary = json.loads(
            (old_root / case_name / "scenario_summary.json").read_text(encoding="utf-8")
        )
        old_detection = json.loads(
            (old_root / case_name / "detected_static_sites.json").read_text(encoding="utf-8")
        )
        result = detect_static_sites(stations, frequencies, rho_a, config)
        plot_case_diagnostics(
            result, rho_a, setup["known_static_indices"], OUTPUT_ROOT / case_name
        )
        rows.append(save_case(case_name, result, setup, old_summary, old_detection))
    comparison = {
        "cases": [
            {
                "experiment_id": row["experiment_id"],
                "old_detected_static_indices": row["old_detected_static_indices"],
                "new_candidate_station_indices_by_frequency": row["candidate_station_indices_by_frequency"],
                "alpha_comparison": row["alpha_comparison"],
                "old_per_frequency_metrics": row["old_per_frequency_metrics"],
                "new_per_frequency_metrics": row["new_per_frequency_metrics"],
            }
            for row in rows
        ],
        "old_station_metrics": aggregate_old_station_metrics(rows),
        "old_per_frequency_metrics": aggregate_frequency_metrics(rows, "old_per_frequency_metrics"),
        "new_per_frequency_metrics": aggregate_frequency_metrics(rows, "new_per_frequency_metrics"),
        "new_station_metrics": None,
        "station_metric_comparison": "不可直接比较：参考文献未定义跨频最终测点分类",
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved comparison: {OUTPUT_ROOT / 'comparison.json'}")


if __name__ == "__main__":
    main()
