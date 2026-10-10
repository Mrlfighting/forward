"""只读对照旧图的输入、共同一维初值及四组恢复模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "section_23" / "232_layered_model" / "inputs.npz"
RESULT_DIR = PROJECT_ROOT / "results" / "section_23" / "232_layered_model"
DEFAULT_LEGACY = (
    Path(r"D:\staticjiaozheng\论文整理\05_moderate_layered_static_data_selection")
    / "moderate_layered_original_reproduced"
)
EXPECTED_HASHES = {
    "data_clean_moderate_layered.npy": "9e1cf1fd8763bc64b0f70fde0fc15877ebe465a5b1e2e21a7c7b22b713ca5b38",
    "data_static_moderate_layered.npy": "399cd4c24b9295ba24c415f118e82358d80673f1de61060f809cafb36d5cb198",
    "one_d_active_start.npy": "4b7df4de7b68ae7144739634d4c34b47901073ef275b15a0b1494aaa94ca4d2a",
    "recovered_clean_reference.npy": "91061e4213a5dd87adf023340a05f5b687cccee463071f8712e52f16a71f9da8",
    "recovered_moderate_static_full.npy": "d919b5c34fb725e53813b5665e1f5dd9b4a0a5376cf838055421637bdab4881f",
    "recovered_moderate_phase_only.npy": "25a46acd71db4904e8f276039297a71aedd1a3314f23f05131654fe12192e6d9",
    "recovered_moderate_shape.npy": "9cb2213e75a34396bad3eebbb19c759705daabdb9c0357d54f82bc5caadcd24c",
}


def compare(new: np.ndarray, old: np.ndarray) -> dict:
    """同时记录严格差值和均方根差值。"""

    if new.shape != old.shape:
        raise AssertionError(f"新旧数组尺寸不同：{new.shape} != {old.shape}")
    delta = np.asarray(new) - np.asarray(old)
    return {
        "max_abs_difference": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
    }


def main() -> None:
    """哈希验证历史文件后写入只读数值比较报告。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY)
    args = parser.parse_args()
    legacy = args.legacy_dir.resolve()
    for filename, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256((legacy / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"历史文件哈希不一致：{filename}")
    with np.load(INPUT_PATH, allow_pickle=False) as saved:
        source = {key: saved[key] for key in saved.files}
    report = {
        "legacy_directory": str(legacy),
        "clean_observation": compare(
            source["clean_full"], np.load(legacy / "data_clean_moderate_layered.npy", allow_pickle=False)
        ),
        "static_observation": compare(
            source["static_full"], np.load(legacy / "data_static_moderate_layered.npy", allow_pickle=False)
        ),
        "one_d_active_start": compare(
            source["one_d_active_start"], np.load(legacy / "one_d_active_start.npy", allow_pickle=False)
        ),
        "methods": {},
    }
    legacy_names = {
        "clean_full": "recovered_clean_reference.npy",
        "static_full": "recovered_moderate_static_full.npy",
        "phase_preserving": "recovered_moderate_phase_only.npy",
        "phase_plus_shape": "recovered_moderate_shape.npy",
    }
    for method, filename in legacy_names.items():
        with np.load(RESULT_DIR / f"{method}.npz", allow_pickle=False) as saved:
            recovered = saved["recovered_log_conductivity"]
        report["methods"][method] = compare(
            recovered, np.load(legacy / filename, allow_pickle=False)
        )
    (RESULT_DIR / "legacy_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
