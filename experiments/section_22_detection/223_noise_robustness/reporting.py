"""保存每次含噪识别的完整诊断及跨重复统计。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import candidate_alpha_min, distribution, wilson_interval


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def write_json(path: Path, payload: dict | list) -> None:
    """严格禁止非标准 JSON 浮点值并保留中文。"""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_trial(
    destination: Path,
    result,
    record: dict,
    clean_rho: np.ndarray,
    noisy_rho: np.ndarray,
    unchanged_phase: np.ndarray,
    standard_normal: np.ndarray,
    noise_factor: np.ndarray,
    target_indices: list[int],
    plot_diagnostics,
    make_figures: bool,
) -> None:
    """分别保存输入、五尺度中间量、完整传播线和逐频候选。"""

    destination.mkdir(parents=True)
    np.savez_compressed(
        destination / "input.npz",
        station_x_m=result.station_x_m,
        frequencies_hz=result.frequencies_hz,
        target_station_indices=np.asarray(target_indices, dtype=int),
        clean_apparent_resistivity=clean_rho,
        noisy_apparent_resistivity=noisy_rho,
        unchanged_phase_deg=unchanged_phase,
        standard_normal_draws=standard_normal,
        noise_factor=noise_factor,
        noise_fraction=np.asarray(record["noise_fraction"]),
    )
    details = result.frequency_diagnostics
    np.savez_compressed(
        destination / "detection_diagnostics.npz",
        station_x_m=result.station_x_m,
        frequencies_hz=result.frequencies_hz,
        sample_x_m=details[0].sample_x_m,
        signal_at_stations=np.stack([item.signal_at_stations for item in details]),
        signal_profiles=np.stack([item.signal_profile for item in details]),
        wavelet_coefficients=np.stack([item.coefficients for item in details]),
        modulus_maxima=np.stack([item.modulus_maxima for item in details]),
        selected_maxima=np.stack([item.selected_maxima for item in details]),
        per_frequency_votes=result.per_frequency_votes,
        occurrence_counts=result.vote_counts,
        candidate_alpha_min=candidate_alpha_min(result),
    )
    frequency_lines = []
    with (destination / "static_candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "frequency_hz", "candidate_x_m", "station_index", "station_x_m",
            "mapping_distance_m", "alpha_2_4", "alpha_4_8", "alpha_8_16", "alpha_16_32",
        ])
        for item in details:
            lines = []
            for line in item.maxima_lines:
                lines.append({
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
                        result.station_x_m[index], line.mapping_distance_m, *line.alpha_pairs,
                    ])
            frequency_lines.append({"frequency_hz": item.frequency_hz, "maxima_lines": lines})
    write_json(destination / "maxima_lines.json", frequency_lines)
    write_json(destination / "trial_summary.json", record)
    if make_figures:
        known = target_indices if record["arm"] == "with_shallow" else []
        plot_diagnostics(result, noisy_rho, known, destination)


def aggregate_records(records: list[dict], noise_fractions: list[float]) -> list[dict]:
    """按静态体有无与噪声级别汇总逐次指标分布。"""

    groups = []
    for arm in ("with_shallow", "without_shallow"):
        for sigma in noise_fractions:
            subset = [row for row in records if row["arm"] == arm and row["noise_fraction"] == sigma]
            if not subset:
                raise ValueError("缺少指定噪声级别或对照组")
            fields = ("tp", "fp", "fn", "false_positive_rate", "total_frequency_station_candidates")
            if arm == "with_shallow":
                fields += ("precision", "recall", "f1", "neighbor_false_candidates", "outside_region_candidates")
            group = {
                "arm": arm,
                "noise_fraction": sigma,
                "repetition_count": len(subset),
                "metrics": {field: distribution([row["metrics"][field] for row in subset]) for field in fields},
                "redrawn_sample_count_total": int(sum(row["redrawn_sample_count"] for row in subset)),
                "realized_relative_rms": distribution([row["realized_relative_rms"] for row in subset]),
            }
            if arm == "with_shallow":
                group["target_frequency_counts"] = [
                    distribution([row["metrics"]["target_frequency_counts"][index] for row in subset])
                    for index in (0, 1)
                ]
                group["region_frequency_counts"] = [
                    distribution([row["metrics"]["region_frequency_counts"][index] for row in subset])
                    for index in (0, 1)
                ]
                successes = sum(bool(row["metrics"]["both_targets_detected_any_frequency"]) for row in subset)
                group["both_targets_any_frequency"] = {
                    "successes": successes,
                    "fraction": successes / len(subset),
                    "wilson_95_interval": wilson_interval(successes, len(subset)),
                    "note": "仅为宽松逐次诊断，不是跨频最终测点分类",
                }
            groups.append(group)
    return groups


def save_summary(destination: Path, records: list[dict], groups: list[dict], provenance: dict) -> None:
    """保存逐次总表、分布摘要及两组配对对照图。"""

    write_json(destination / "summary.json", {"groups": groups, "provenance": provenance})
    with (destination / "repetitions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "arm", "noise_fraction", "repetition", "tp", "fp", "fn", "precision", "recall", "f1",
            "false_positive_rate", "hits_minus_5km", "hits_plus_5km", "region_hits_minus_5km",
            "region_hits_plus_5km", "neighbor_false_candidates", "outside_region_candidates",
            "both_targets_any_frequency", "redrawn_sample_count", "realized_relative_rms",
        ])
        for row in records:
            metrics = row["metrics"]
            counts = metrics["target_frequency_counts"] or [None, None]
            region = metrics["region_frequency_counts"] or [None, None]
            writer.writerow([
                row["arm"], row["noise_fraction"], row["repetition"], metrics["tp"], metrics["fp"],
                metrics["fn"], metrics["precision"], metrics["recall"], metrics["f1"],
                metrics["false_positive_rate"], *counts, *region, metrics["neighbor_false_candidates"],
                metrics["outside_region_candidates"], metrics["both_targets_detected_any_frequency"],
                row["redrawn_sample_count"], row["realized_relative_rms"],
            ])
    with (destination / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "arm", "noise_fraction", "repetitions", "f1_q25", "f1_median", "f1_q75",
            "recall_q25", "recall_median", "recall_q75", "fp_q25", "fp_median", "fp_q75",
            "target_minus_5km_median", "target_plus_5km_median", "both_targets_any_fraction",
            "redrawn_sample_count_total",
        ])
        for group in groups:
            metrics = group["metrics"]
            is_positive = group["arm"] == "with_shallow"
            f1 = metrics.get("f1")
            recall = metrics.get("recall")
            target = group.get("target_frequency_counts", [None, None])
            writer.writerow([
                group["arm"], group["noise_fraction"], group["repetition_count"],
                *([f1[key] for key in ("q25", "median", "q75")] if is_positive else [None] * 3),
                *([recall[key] for key in ("q25", "median", "q75")] if is_positive else [None] * 3),
                *[metrics["fp"][key] for key in ("q25", "median", "q75")],
                *([value["median"] for value in target] if is_positive else [None, None]),
                group["both_targets_any_frequency"]["fraction"] if is_positive else None,
                group["redrawn_sample_count_total"],
            ])
    positive = [group for group in groups if group["arm"] == "with_shallow"]
    negative = [group for group in groups if group["arm"] == "without_shallow"]
    x = [group["noise_fraction"] * 100 for group in positive]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for field, color, label in (("recall", "#216c9e", "召回率"), ("f1", "#b44b38", "F1")):
        median = [group["metrics"][field]["median"] for group in positive]
        lower = [group["metrics"][field]["q25"] for group in positive]
        upper = [group["metrics"][field]["q75"] for group in positive]
        axes[0].plot(x, median, "o-", color=color, label=label)
        axes[0].fill_between(x, lower, upper, color=color, alpha=0.16)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("逐频指标")
    axes[0].set_title("双浅部体识别表现")
    for group_set, color, label in ((positive, "#b44b38", "双浅部体"), (negative, "#515d70", "无浅部体对照")):
        median = [group["metrics"]["fp"]["median"] for group in group_set]
        lower = [group["metrics"]["fp"]["q25"] for group in group_set]
        upper = [group["metrics"]["fp"]["q75"] for group in group_set]
        axes[1].plot(x, median, "o-", color=color, label=label)
        axes[1].fill_between(x, lower, upper, color=color, alpha=0.16)
    axes[1].set_ylabel("逐频误候选数量")
    axes[1].set_title("噪声与误候选")
    for ax in axes:
        ax.set_xlabel("视电阻率相对噪声标准差（%）")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "噪声水平与识别鲁棒性.png", dpi=180)
    plt.close(fig)
