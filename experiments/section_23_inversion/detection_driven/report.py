"""汇总三个场景的检测成败及真值掩码与检测掩码反演差异。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "results/section_23/detection_driven"
CONFIG_PATH = ROOT / "configs/section_23/detection_driven.json"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    records = []
    for case_id, case in config["cases"].items():
        detected = json.loads((OUTPUT_ROOT / case_id / "summary.json").read_text(encoding="utf-8"))
        oracle_path = ROOT / case["oracle_result_directory"] / "summary.json"
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        detected_results = detected.get("inversions", {})
        oracle_results = oracle.get("results", {})
        metrics = detected["classification_metrics"]
        records.append({
            "case_id": case_id,
            "status": detected["status"],
            "oracle_result_status": oracle.get("status", "未单独标记"),
            "frequency_count": detected["frequency_count"],
            "minimum_votes": detected["minimum_votes"],
            "truth_x_m": json.dumps(detected["truth_x_m_evaluation_only"]),
            "detected_x_m": json.dumps(detected["detected_x_m"]),
            "TP": metrics["TP"], "FP": metrics["FP"], "FN": metrics["FN"],
            "precision": metrics["precision"], "recall": metrics["recall"],
            "F1": metrics["F1"],
            "oracle_phase_roi_log10_rmse": oracle_results.get("phase_preserving", {}).get("roi_log10_rmse"),
            "detected_phase_roi_log10_rmse": detected_results.get("phase_preserving", {}).get("roi_log10_rmse"),
            "oracle_shape_roi_log10_rmse": oracle_results.get("phase_plus_shape", {}).get("roi_log10_rmse"),
            "detected_shape_roi_log10_rmse": detected_results.get("phase_plus_shape", {}).get("roi_log10_rmse"),
        })
    with (OUTPUT_ROOT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps({"cases": records}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("三场景识别与反演对照表已保存", flush=True)


if __name__ == "__main__":
    main()
