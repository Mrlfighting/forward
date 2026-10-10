"""读取独立正演场景，对 TM/TE 进行基础逐频静态候选识别。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mt_static.detection import DetectionConfig, detect_static_sites
from run_section21_forward_s2_check import file_sha256, save_detection


FORWARD_ROOT = PROJECT_ROOT / "results" / "section_21_forward" / "scenarios"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "section_22" / "221_site_types" / "scenarios"
CONFIG_PATH = PROJECT_ROOT / "configs" / "section_22" / "site_types.json"
SCENARIO_ID = re.compile(r"^[a-z][a-z0-9_]*$")


def main() -> None:
    """比较同一场景有无浅部体的逐频候选，不制定跨频票数。"""

    parser = argparse.ArgumentParser(description="读取独立正演场景并运行基础识别")
    parser.add_argument("--scenario-id", required=True, help="已完成正演的场景编号")
    parser.add_argument("--mode", choices=("TM", "TE"), default="TM", help="用于识别的极化模式")
    args = parser.parse_args()
    if SCENARIO_ID.fullmatch(args.scenario_id) is None:
        raise ValueError("场景编号格式无效")
    forward_dir = FORWARD_ROOT / args.scenario_id
    forward_path = forward_dir / "forward.npz"
    manifest = json.loads((forward_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["scenario_id"] != args.scenario_id:
        raise ValueError("场景编号与正演摘要不一致")
    config = DetectionConfig(**json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["detection"])
    with np.load(forward_path, allow_pickle=False) as saved:
        stations = np.asarray(saved["station_x_m"], dtype=float)
        frequencies = np.asarray(saved["frequencies_hz"], dtype=float)
        responses = {
            label: np.asarray(saved[f"{args.mode.lower()}_{label}_apparent_resistivity"], dtype=float)
            for label in ("without_shallow", "with_shallow")
        }
    bodies = manifest["body_geometry"]["shallow_bodies"]
    centers = [
        float((body["specified_bounds_m"][0][0] + body["specified_bounds_m"][1][0]) / 2.0)
        for body in bodies
    ]
    target_indices = [int(np.argmin(np.abs(stations - center))) for center in centers]
    output_dir = OUTPUT_ROOT / args.scenario_id / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scenario_id": args.scenario_id,
        "mode": args.mode,
        "source_forward_npz": str(forward_path.resolve()),
        "source_forward_sha256": file_sha256(forward_path),
        "source_manifest": str((forward_dir / "manifest.json").resolve()),
        "detection_config": asdict(config),
        "body_centers_x_m": centers,
        "nearest_station_x_m": [float(stations[index]) for index in target_indices],
        "results": {},
        "final_static_sites": None,
        "final_classification_reason": "当前基础算法只输出逐频候选，不设跨频最终测点规则",
    }
    for label, rho_a in responses.items():
        result = detect_static_sites(stations, frequencies, rho_a, config)
        summary["results"][label] = save_detection(
            label, result, rho_a, target_indices, output_dir / label
        )
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved detection: {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
