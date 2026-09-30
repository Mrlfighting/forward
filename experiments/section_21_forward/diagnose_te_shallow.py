"""用固定足迹与实际几何两组浅部体区分网格和物理响应。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "simpeg"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from discretize import TensorMesh
from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import BreakingChangeWarning, model_builder

from mt_static.forward.mesh import build_mesh
from mt_static.forward.model import build_models
from mt_static.forward.settings import FREQUENCIES_HZ, VACUUM_PERMEABILITY


DEFAULT_FREQUENCY_INDICES = (0, 14, 19)
STATIONS_M = np.array([-5000.0, 5000.0])
BODY_RESISTIVITIES = {"left": 1000.0, "right": 10.0}
BODY_BOUNDS = {
    "original": {
        "left": ([-5201.0, -4.0], [-4899.0, 0.0]),
        "right": ([4899.0, -4.0], [5201.0, 0.0]),
    },
    "fixed_footprint": {
        "left": ([-5200.0, -5.0], [-4800.0, 0.0]),
        "right": ([4800.0, -5.0], [5200.0, 0.0]),
    },
}


def build_refined_mesh(horizontal_width_m: float) -> TensorMesh:
    """保持原网格核心范围与深部节点，细化浅部和核心区。"""
    original = build_mesh()
    core_cell_count = round(20400.0 / horizontal_width_m)
    horizontal_widths = [
        (200.0, 12, -1.3),
        (horizontal_width_m, core_cell_count),
        (200.0, 12, 1.3),
    ]
    deep_nodes = original.nodes_y[original.nodes_y < -80.0]
    shallow_nodes = np.arange(-80.0, 1.0, 1.0)
    vertical_nodes = np.concatenate((deep_nodes, shallow_nodes))
    centered_mesh = TensorMesh([horizontal_widths, [1.0]], x0="CN")
    return TensorMesh(
        [horizontal_widths, np.diff(vertical_nodes)],
        x0=[centered_mesh.x0[0], vertical_nodes[0]],
    )


def add_shallow_bodies(mesh: TensorMesh, background: np.ndarray, geometry: str) -> tuple[np.ndarray, dict]:
    """在共同深部模型上加入指定浅部体，并记录实际单元足迹。"""
    model = background.copy()
    footprints = {}
    for name, (lower, upper) in BODY_BOUNDS[geometry].items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BreakingChangeWarning)
            indices = np.asarray(
                model_builder.get_indices_block(np.asarray(lower), np.asarray(upper), mesh.cell_centers),
                dtype=int,
            )
        model[indices] = np.log(1.0 / BODY_RESISTIVITIES[name])
        centers = mesh.cell_centers[indices]
        x_indices = np.searchsorted(mesh.nodes_x, centers[:, 0]) - 1
        z_indices = np.searchsorted(mesh.nodes_y, centers[:, 1]) - 1
        footprints[name] = {
            "cell_count": int(indices.size),
            "x_bounds_m": [
                float(np.min(mesh.nodes_x[x_indices])),
                float(np.max(mesh.nodes_x[x_indices + 1])),
            ],
            "z_bounds_m": [
                float(np.min(mesh.nodes_y[z_indices])),
                float(np.max(mesh.nodes_y[z_indices + 1])),
            ],
        }
    return model, footprints


def build_diagnostic_survey(frequency_indices: tuple[int, ...]) -> nsem.survey.Survey:
    """仅计算低、中、高三个频率及浅部异常体正上方测点。"""
    locations = np.column_stack((STATIONS_M, np.zeros_like(STATIONS_M)))
    sources = []
    for index in frequency_indices:
        receivers = [
            nsem.receivers.Impedance(locations, orientation="yx", component=component)
            for component in ("real", "imag")
        ]
        sources.append(nsem.sources.Planewave(receivers, FREQUENCIES_HZ[index]))
    return nsem.survey.Survey(sources)


def convert_selected_response(dpred: np.ndarray, frequency_indices: tuple[int, ...]) -> dict:
    """按所选频率、实虚部、测点顺序转换为视电阻率和相位。"""
    impedance_parts = dpred.reshape(len(frequency_indices), 2, STATIONS_M.size)
    impedance = -(impedance_parts[:, 0, :] + 1j * impedance_parts[:, 1, :])
    frequencies = FREQUENCIES_HZ[list(frequency_indices)]
    apparent_resistivity = np.abs(impedance) ** 2 / (VACUUM_PERMEABILITY * 2.0 * np.pi * frequencies[:, None])
    phase = np.mod(np.angle(impedance, deg=True), 180.0)
    return {"apparent_resistivity": apparent_resistivity, "phase_deg": phase}


def run_case(horizontal_width_m: float, frequency_indices: tuple[int, ...]) -> dict:
    """计算同一细网格上的无浅部体及两种浅部体足迹。"""
    mesh = build_refined_mesh(horizontal_width_m)
    without_shallow, _, _ = build_models(mesh)
    models = {"without_shallow": without_shallow}
    footprints = {}
    for geometry in BODY_BOUNDS:
        models[geometry], footprints[geometry] = add_shallow_bodies(mesh, without_shallow, geometry)
    simulation = nsem.simulation.Simulation2DMagneticField(
        mesh, survey=build_diagnostic_survey(frequency_indices), sigmaMap=maps.ExpMap(mesh), storeInnerProduct=True
    )
    responses = {}
    for name, model in models.items():
        print(f"网格 {horizontal_width_m:g} m：TE {name}", flush=True)
        responses[name] = convert_selected_response(np.asarray(simulation.dpred(model), dtype=float), frequency_indices)
    baseline = responses["without_shallow"]
    differences = {}
    for geometry in BODY_BOUNDS:
        affected = responses[geometry]
        differences[geometry] = {
            "apparent_resistivity_ratio": (
                affected["apparent_resistivity"] / baseline["apparent_resistivity"]
            ).tolist(),
            "phase_difference_deg": (
                (affected["phase_deg"] - baseline["phase_deg"] + 90.0) % 180.0 - 90.0
            ).tolist(),
        }
    return {"mesh_cells": list(mesh.shape_cells), "footprints": footprints, "differences": differences}


def main() -> None:
    """将诊断结果单独保存，避免覆盖论文正式结果。"""
    parser = argparse.ArgumentParser(description="诊断 TE 高频浅部响应的网格与尺寸来源")
    parser.add_argument("--widths", type=float, nargs="+", default=[100.0, 50.0])
    parser.add_argument("--frequency-indices", type=int, nargs="+", default=list(DEFAULT_FREQUENCY_INDICES))
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    frequency_indices = tuple(args.frequency_indices)
    if not frequency_indices or any(index < 0 or index >= FREQUENCIES_HZ.size for index in frequency_indices):
        parser.error("频率索引必须在已有频率数组范围内")
    output_dir = PROJECT_ROOT / "results" / "section_21_forward" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_file or output_dir / "te_grid_body_size.json"
    report = {
        "frequencies_hz": FREQUENCIES_HZ[list(frequency_indices)].tolist(),
        "stations_m": STATIONS_M.tolist(),
        "cases": {},
    }
    for width in args.widths:
        report["cases"][str(int(width))] = run_case(width, frequency_indices)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"诊断结果：{output_path}")


if __name__ == "__main__":
    main()
