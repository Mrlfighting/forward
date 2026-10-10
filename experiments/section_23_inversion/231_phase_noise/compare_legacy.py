"""将四静态点重跑结果与原图保存的数值逐项对照。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "section_23" / "231_phase_noise" / "four_static_rectangles_v1"
DEFAULT_LEGACY = Path(
    r"D:\staticjiaozheng\论文整理\04_static_site_phase_interference_downweighted"
    r"\static_site_random_phase_interference_5deg_static_phase_downweighted"
)


def main() -> None:
    """保存输入一致性与四组模型差异，不据此调反演参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY)
    args = parser.parse_args()
    directory = args.input_dir.resolve()
    legacy = args.legacy_dir.resolve()
    old_report = json.loads((legacy / "static_site_random_phase_interference_summary.json").read_text(encoding="utf-8"))
    new_report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with np.load(directory / "input.npz", allow_pickle=False) as saved:
        new_observed = saved["static_full"]
        new_std = saved["static_std"]
    old_observed = np.load(legacy / "data_static_site_random_phase_interference.npy", allow_pickle=False)
    old_std = np.load(legacy / "std_static_site_random_phase_interference.npy", allow_pickle=False)
    names = (
        ("clean_full", "recovered_clean_full_reference.npy", "clean_full_reference"),
        ("static_full", "recovered_static_site_random_phase_full.npy", "static_full_static_site_phase_noise"),
        ("phase_preserving", "recovered_static_site_random_phase_zeng.npy", "zeng_static_site_phase_noise"),
        ("phase_plus_shape", "recovered_static_site_random_phase_shape.npy", "shape_static_site_phase_noise"),
    )
    rows = {}
    for method, old_filename, old_key in names:
        with np.load(directory / f"{method}.npz", allow_pickle=False) as saved:
            new_model = saved["recovered_log_conductivity"]
        old_model = np.load(legacy / old_filename, allow_pickle=False)
        if new_model.shape != old_model.shape:
            raise AssertionError(f"{method} 新旧模型形状不同")
        new_rmse = new_report["results"][method]["roi_log10_rmse"]
        old_rmse = old_report["results"][old_key]["log_rmse_roi"]
        rows[method] = {
            "new_roi_log10_rmse": new_rmse,
            "legacy_roi_log10_rmse": old_rmse,
            "roi_rmse_difference": new_rmse - old_rmse,
            "model_log_conductivity_rmse": float(np.sqrt(np.mean((new_model - old_model) ** 2))),
            "model_log_conductivity_max_abs_difference": float(np.max(np.abs(new_model - old_model))),
        }
    comparison = {
        "legacy_directory": str(legacy),
        "max_abs_observation_difference": float(np.max(np.abs(new_observed - old_observed))),
        "max_abs_standard_deviation_difference": float(np.max(np.abs(new_std - old_std))),
        "methods": rows,
        "note": "目标区误差和恢复模型只作原图复现核对；各方法数据量不同，不能用总拟合值单独排序。",
    }
    (directory / "legacy_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
