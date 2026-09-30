"""组织论文 2.1 正演并保存可复核的数值结果。"""

import json
from pathlib import Path

import numpy as np
from discretize import TensorMesh

from .mesh import build_mesh
from .model import build_models, describe_anomaly_regions
from .plotting import plot_results
from .response import summarize_differences
from .settings import BACKGROUND_RESISTIVITY_OHM_M, FREQUENCIES_HZ, STATION_X_M
from .simulation import simulate_mode


def load_matching_baselines(
    result_path: Path, mesh: TensorMesh, without_shallow: np.ndarray
) -> dict[str, np.ndarray]:
    """仅在网格、深部模型和观测完全一致时复用旧基线阻抗。"""
    expected_arrays = {
        "frequencies_hz": FREQUENCIES_HZ,
        "station_x_m": STATION_X_M,
        "mesh_x_nodes_m": mesh.nodes_x,
        "mesh_z_nodes_m": mesh.nodes_y,
        "without_shallow_log_conductivity": without_shallow,
    }
    baselines = {}
    with np.load(result_path, allow_pickle=False) as existing:
        for name, expected in expected_arrays.items():
            if name not in existing or not np.array_equal(existing[name], expected):
                raise ValueError(f"已有结果的 {name} 与当前模型不一致，不能复用基线")
        expected_size = 2 * FREQUENCIES_HZ.size * STATION_X_M.size
        for mode in ("TE", "TM"):
            name = f"{mode.lower()}_without_shallow_dpred"
            if name not in existing:
                raise ValueError(f"已有结果缺少 {name}，不能复用基线")
            values = np.asarray(existing[name], dtype=float).copy()
            if values.size != expected_size or not np.all(np.isfinite(values)):
                raise ValueError(f"已有结果的 {name} 尺寸或数值无效")
            baselines[mode] = values
    return baselines


def run_forward(output_dir: Path, reuse_baseline: bool = False) -> dict:
    """计算两种模式的浅部响应，可复用经过严格核对的无浅部体基线。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = build_mesh()
    without_shallow, with_shallow, cell_counts = build_models(mesh)
    anomaly_regions = describe_anomaly_regions(mesh)
    models = {"without_shallow": without_shallow, "with_shallow": with_shallow}
    baselines = (
        load_matching_baselines(output_dir / "section_21_forward.npz", mesh, without_shallow)
        if reuse_baseline else {}
    )
    responses = {
        mode: simulate_mode(mesh, models, mode, baseline_dpred=baselines.get(mode))
        for mode in ("TE", "TM")
    }
    output_arrays = {
        "frequencies_hz": FREQUENCIES_HZ,
        "station_x_m": STATION_X_M,
        "mesh_x_nodes_m": mesh.nodes_x,
        "mesh_z_nodes_m": mesh.nodes_y,
        "without_shallow_log_conductivity": without_shallow,
        "with_shallow_log_conductivity": with_shallow,
    }
    for mode, mode_responses in responses.items():
        for model_name, data in mode_responses.items():
            for field_name, values in data.items():
                output_arrays[f"{mode.lower()}_{model_name}_{field_name}"] = values
    np.savez_compressed(output_dir / "section_21_forward.npz", **output_arrays)
    summary = {
        "schema_version": 3,
        "baseline_reused_after_exact_match": reuse_baseline,
        "mesh": {
            "horizontal_cell_count": int(mesh.shape_cells[0]),
            "vertical_cell_count": int(mesh.shape_cells[1]),
            "total_cell_count": int(mesh.n_cells),
            "x_bounds_m": [float(mesh.nodes_x.min()), float(mesh.nodes_x.max())],
            "z_bounds_m": [float(mesh.nodes_y.min()), float(mesh.nodes_y.max())],
            "surface_cell_thickness_m": float(mesh.h[1][-1]),
        },
        "models": {
            "background_resistivity_ohm_m": BACKGROUND_RESISTIVITY_OHM_M,
            "anomaly_cell_counts": cell_counts,
            "anomaly_regions": anomaly_regions,
            "different_cell_count": int(np.count_nonzero(without_shallow != with_shallow)),
            "shallow_body_bounds_m": {
                "left": next(
                    description["specified_bounds_m"]
                    for name, description in anomaly_regions.items()
                    if name.startswith("left_shallow")
                ),
                "right": next(
                    description["specified_bounds_m"]
                    for name, description in anomaly_regions.items()
                    if name.startswith("right_shallow")
                ),
            },
        },
        "stations": {"count": int(STATION_X_M.size), "x_bounds_m": [-10000, 10000], "spacing_m": 500},
        "frequencies": {"count": int(FREQUENCIES_HZ.size), "bounds_hz": [float(FREQUENCIES_HZ[0]), float(FREQUENCIES_HZ[-1])]},
        "data_conventions": {
            "te": "Zyx=Ey/Hx; Simulation2DMagneticField",
            "tm": "Zxy=Ex/Hy; Simulation2DElectricField",
            "dpred_order": "ascending frequency; all stations real, then all stations imaginary",
            "display_impedance": "negative SimPEG impedance; raw impedance is also saved",
            "apparent_resistivity": "|Z|^2/(mu_0*2*pi*f)",
            "phase": "arg(-Z) modulo 180 degrees",
            "noise": "none",
            "injected_static_shift": "none; only shallow conductivity cells differ",
        },
        "response_differences": summarize_differences(responses),
    }
    (output_dir / "section_21_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_results(responses, output_dir, FREQUENCIES_HZ, STATION_X_M)
    return summary
