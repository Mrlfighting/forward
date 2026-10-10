"""只读回放旧一维初值与观测数据，定位静态全数据组偏差。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "simpeg"))

from simpeg import data_misfit
from simpeg.data import Data

from mt_static.inversion import active_cells, build_full_survey, build_inversion_mesh, build_simulation, invert_fixed
from mt_static.inversion.solver import InversionSettings


CONFIG_PATH = PROJECT_ROOT / "configs" / "section_23" / "layered_model.json"
DEFAULT_LEGACY = (
    Path(r"D:\staticjiaozheng\论文整理\05_moderate_layered_static_data_selection")
    / "moderate_layered_original_reproduced"
)
CACHE_DIR = PROJECT_ROOT / "cache" / "section_23" / "232_layered_model"
EXPECTED_SHA256 = {
    "one_d_active_start.npy": "4b7df4de7b68ae7144739634d4c34b47901073ef275b15a0b1494aaa94ca4d2a",
    "data_static_moderate_layered.npy": "399cd4c24b9295ba24c415f118e82358d80673f1de61060f809cafb36d5cb198",
    "recovered_moderate_static_full.npy": "d919b5c34fb725e53813b5665e1f5dd9b4a0a5376cf838055421637bdab4881f",
}


def main() -> None:
    """保持新求解器不变，只替换为旧图锁定的起点和观测。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    legacy = args.legacy_dir.resolve()
    for name, expected in EXPECTED_SHA256.items():
        if hashlib.sha256((legacy / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"历史输入哈希不符：{name}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    settings = InversionSettings(**config["inversion"])
    mesh = build_inversion_mesh()
    active = active_cells(mesh, settings)
    start = np.load(legacy / "one_d_active_start.npy", allow_pickle=False)
    observed = np.load(legacy / "data_static_moderate_layered.npy", allow_pickle=False)
    old_model = np.load(legacy / "recovered_moderate_static_full.npy", allow_pickle=False)
    frequencies = np.geomspace(
        config["frequency_min_hz"], config["frequency_max_hz"], config["frequency_count"]
    )
    stations = np.asarray(config["station_x_m"], dtype=float)
    survey = build_full_survey(frequencies, stations)
    blocks = observed.reshape(frequencies.size, 2, stations.size)
    std = np.empty_like(blocks)
    std[:, 0, :] = np.hypot(
        config["noise"]["rho_error_floor"], config["noise"]["relative_rho_noise"]
    ) * np.abs(blocks[:, 0, :])
    std[:, 1, :] = np.hypot(
        config["noise"]["phase_error_floor_deg"],
        config["noise"]["phase_error_floor_deg"],
    )
    simulation = build_simulation(mesh, survey, active, settings)
    misfit = data_misfit.L2DataMisfit(
        simulation=simulation,
        data=Data(survey, dobs=observed, standard_deviation=std.ravel()),
    )
    result = invert_fixed("旧输入静态全数据诊断", misfit, simulation, mesh, active, start, settings)
    delta = result.recovered_log_conductivity - old_model
    comparison = {
        "model_log_conductivity_rmse": float(np.sqrt(np.mean(delta**2))),
        "model_max_abs_difference": float(np.max(np.abs(delta))),
        "phi_d": result.phi_d,
        "legacy_input_sha256": EXPECTED_SHA256,
        "note": "仅用于定位新旧静态全数据组差异，不替代独立重算结果。",
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "legacy_start_diagnostic.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
