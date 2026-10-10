"""保存强度实验的完整数值、传播线和诊断图。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def write_json(path: Path, payload: dict | list) -> None:
    """以 UTF-8 保存不含非标准浮点数的 JSON。"""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_case(
    destination: Path,
    record: dict,
    result,
    alpha_min: np.ndarray,
    baseline_rho: np.ndarray,
    shifted_rho: np.ndarray,
    baseline_phase: np.ndarray,
    target_indices: list[int],
    provenance: dict,
    plot_case_diagnostics,
) -> None:
    """保存可重算的输入、完整诊断以及每个频率的传播线。"""

    destination.mkdir(parents=True)
    np.savez_compressed(
        destination / "input.npz",
        station_x_m=result.station_x_m,
        frequencies_hz=result.frequencies_hz,
        target_station_indices=np.asarray(target_indices, dtype=int),
        baseline_apparent_resistivity=baseline_rho,
        shifted_apparent_resistivity=shifted_rho,
        unchanged_phase_deg=baseline_phase,
        cs=np.asarray(record["cs"]),
        delta_s=np.asarray(record["delta_s"]),
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
        candidate_alpha_min=alpha_min,
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
    write_json(destination / "case_summary.json", {**record, **provenance})
    plot_case_diagnostics(result, shifted_rho, [] if record["is_negative_control"] else target_indices, destination)


def save_summary(destination: Path, records: list[dict], boundaries: dict, provenance: dict) -> None:
    """保存跨强度表格与分支边界诊断图。"""

    write_json(destination / "summary.json", {"cases": records, "boundaries": boundaries, **provenance})
    with (destination / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "cs", "delta_s", "is_negative_control", "success_both_targets",
            "hits_minus_5km", "hits_plus_5km", "alpha_min_minus_5km", "alpha_min_plus_5km",
            "tp", "fp", "fn", "precision", "recall", "f1",
        ])
        for row in records:
            metrics = row["metrics"]
            writer.writerow([
                row["cs"], row["delta_s"], row["is_negative_control"],
                row["all_targets_detected_any_frequency"], *row["target_frequency_counts"],
                *row["alpha_min_at_targets"], metrics["tp"], metrics["fp"], metrics["fn"],
                metrics["precision"], metrics["recall"], metrics["f1"],
            ])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    for column, (branch_name, condition) in enumerate((
        ("Cs < 1", lambda cs: cs < 1), ("Cs > 1", lambda cs: cs > 1),
    )):
        branch = sorted((row for row in records if condition(row["cs"])), key=lambda row: row["delta_s"])
        delta = [row["delta_s"] for row in branch]
        for target, color, label in ((0, "#1768a0", "−5 km"), (1, "#c24732", "+5 km")):
            axes[0, column].plot(
                delta, [row["target_frequency_counts"][target] / provenance["frequency_count"] for row in branch],
                "o-", color=color, label=label,
            )
        axes[0, column].set_title(branch_name)
        axes[0, column].set_ylabel("目标测点命中频率比例")
        axes[0, column].set_ylim(-0.05, 1.05)
        axes[0, column].legend()
        axes[1, column].plot(delta, [row["metrics"]["f1"] for row in branch], "o-", color="#654296")
        axes[1, column].set_ylabel("逐频 F1")
        axes[1, column].set_ylim(-0.05, 1.05)
        axes[1, column].set_xlabel("静态位移对数幅度 Δs")
        for row_axes in axes[:, column]:
            row_axes.grid(alpha=0.2)
    fig.suptitle("静态位移强度与基础识别表现")
    fig.tight_layout()
    fig.savefig(destination / "强度与识别指标.png", dpi=180)
    plt.close(fig)
