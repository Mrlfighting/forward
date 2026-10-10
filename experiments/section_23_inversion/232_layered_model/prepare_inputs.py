"""按旧复杂层状实验重新正演，并由普通点求共同一维初值。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "simpeg"))

from mt_static.inversion import active_cells, build_full_survey, build_simulation
from mt_static.forward.layered_data import build_layered_observations, split_full_data
from mt_static.forward.layered_model import build_layered_forward_mesh, build_layered_models
from mt_static.inversion.one_d_background import invert_common_one_d_background
from mt_static.inversion.solver import InversionSettings, build_inversion_mesh


CONFIG_PATH = PROJECT_ROOT / "configs" / "section_23" / "layered_model.json"
INPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "section_23" / "232_layered_model"


def sha256(path: Path) -> str:
    """记录输入文件的内容哈希。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """生成可供四方法共用、与旧目录解耦的固定输入。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stations = np.asarray(config["station_x_m"], dtype=float)
    static_positions = np.asarray(config["static_station_x_m"], dtype=float)
    static_indices = np.searchsorted(stations, static_positions)
    if not np.array_equal(stations[static_indices], static_positions):
        raise AssertionError("两个静态点必须与测点精确对齐")
    normal_indices = np.setdiff1d(np.arange(stations.size), static_indices)
    frequencies = np.geomspace(
        config["frequency_min_hz"], config["frequency_max_hz"], config["frequency_count"]
    )
    settings = InversionSettings(**config["inversion"])
    forward_mesh = build_layered_forward_mesh(static_positions)
    clean_true, static_true, surface_counts = build_layered_models(
        forward_mesh,
        config["layers"],
        config["targets"],
        config["surface_bodies"],
        config["background_resistivity_ohm_m"],
    )
    forward_active = forward_mesh.cell_centers[:, 1] <= 0.0
    inversion_mesh = build_inversion_mesh()
    inversion_active = active_cells(inversion_mesh, settings)
    if tuple(forward_mesh.shape_cells) != (156, 105) or int(forward_active.sum()) != 16380:
        raise AssertionError("正演网格与旧图不一致")
    if tuple(inversion_mesh.shape_cells) != (80, 64) or int(inversion_active.sum()) != 3100:
        raise AssertionError("反演网格与旧图不一致")
    survey = build_full_survey(frequencies, stations)
    forward_simulation = build_simulation(forward_mesh, survey, forward_active, settings)
    print("正演：无浅部体层状模型", flush=True)
    clean_prediction = np.asarray(forward_simulation.dpred(clean_true[forward_active]), dtype=float)
    print("正演：两个 4 m 浅部体层状模型", flush=True)
    static_prediction = np.asarray(forward_simulation.dpred(static_true[forward_active]), dtype=float)
    noise = config["noise"]
    observations = build_layered_observations(
        clean_prediction,
        static_prediction,
        frequencies.size,
        stations.size,
        noise["relative_rho_noise"],
        noise["rho_error_floor"],
        noise["phase_error_floor_deg"],
        noise["random_seed"],
    )
    one_d = invert_common_one_d_background(
        inversion_mesh,
        inversion_active,
        frequencies,
        stations,
        normal_indices,
        observations["static"],
        observations["static_std"],
        settings.background_resistivity,
        maximum_depth_m=settings.active_depth_m,
        max_iter=config["one_d_preinversion_max_iter"],
    )
    clean_rho, _ = split_full_data(clean_prediction, frequencies.size, stations.size)
    static_rho, _ = split_full_data(static_prediction, frequencies.size, stations.size)
    shift_ratios = {}
    for index in static_indices:
        ratio = static_rho[:, index] / clean_rho[:, index]
        shift_ratios[str(int(stations[index]))] = {
            "geometric_mean": float(np.exp(np.mean(np.log(ratio)))),
            "minimum": float(ratio.min()),
            "maximum": float(ratio.max()),
            "log_standard_deviation": float(np.std(np.log(ratio))),
        }
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = INPUT_DIR / "inputs.npz"
    np.savez_compressed(
        input_path,
        frequencies_hz=frequencies,
        station_x_m=stations,
        static_station_indices=static_indices,
        normal_station_indices=normal_indices,
        clean_full=observations["clean"],
        static_full=observations["static"],
        clean_std=observations["clean_std"],
        static_std=observations["static_std"],
        clean_noiseless=clean_prediction,
        physical_static_noiseless=static_prediction,
        true_clean_log_conductivity=clean_true,
        true_static_log_conductivity=static_true,
        forward_mesh_x_nodes_m=forward_mesh.nodes_x,
        forward_mesh_z_nodes_m=forward_mesh.nodes_y,
        inversion_mesh_x_nodes_m=inversion_mesh.nodes_x,
        inversion_mesh_z_nodes_m=inversion_mesh.nodes_y,
        inversion_active_cells=inversion_active,
        one_d_profile=one_d.profile,
        one_d_depth_m=one_d.depth_m,
        one_d_active_start=one_d.active_start,
    )
    manifest = {
        "config_sha256": sha256(CONFIG_PATH),
        "input_sha256": sha256(input_path),
        "forward_mesh_shape": list(forward_mesh.shape_cells),
        "inversion_mesh_shape": list(inversion_mesh.shape_cells),
        "forward_active_cells": int(forward_active.sum()),
        "inversion_active_cells": int(inversion_active.sum()),
        "surface_body_cell_counts": surface_counts,
        "static_shift_ratios": shift_ratios,
        "rho_relative_std": observations["rho_relative_std"],
        "phase_std_deg": observations["phase_std_deg"],
        "one_d_data_count": int(frequencies.size * 2 * normal_indices.size),
        "one_d_parameter_count": int(one_d.profile.size),
        "one_d_phi_d": one_d.phi_d,
        "one_d_elapsed_seconds": one_d.elapsed_seconds,
    }
    (INPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
