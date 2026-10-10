"""在局部加密网格上比较浅部体尺寸对 TE 和 TM 的影响。"""

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


STATIONS_M = np.array([-5000.0, 5000.0])
BODY_RESISTIVITIES = {"left": 1000.0, "right": 10.0}
LOCAL_CELL_WIDTHS_M = np.array([1.0] * 10 + [2.0] * 5 + [4.0] * 5 + [8.0] * 5 + [16.0] * 5 + [20.0] * 2)


def build_local_mesh(body_cell_size_m: float = 1.0) -> TensorMesh:
    """仅在两处浅部体附近细化水平单元，并将表层垂向细化至 1 m。"""
    original = build_mesh()
    offsets = np.cumsum(LOCAL_CELL_WIDTHS_M)
    if not np.isclose(offsets[-1], 200.0):
        raise AssertionError("局部水平细化范围必须为中心两侧各 200 m")
    horizontal_nodes = original.nodes_x.copy()
    for center in STATIONS_M:
        horizontal_nodes = horizontal_nodes[
            (horizontal_nodes < center - 200.0) | (horizontal_nodes > center + 200.0)
        ]
        local_nodes = np.concatenate((center - offsets[::-1], [center], center + offsets))
        horizontal_nodes = np.concatenate((horizontal_nodes, local_nodes))
        if body_cell_size_m == 0.5:
            body_nodes = np.arange(center - 2.0, center + 2.0 + 1e-9, 0.5)
            horizontal_nodes = np.concatenate((horizontal_nodes, body_nodes))
    horizontal_nodes = np.unique(np.round(horizontal_nodes, decimals=8))
    deep_nodes = original.nodes_y[original.nodes_y < -80.0]
    shallow_nodes = np.arange(-80.0, 1.0, 1.0)
    vertical_nodes = np.concatenate((deep_nodes, shallow_nodes))
    if body_cell_size_m == 0.5:
        vertical_nodes = np.unique(np.concatenate((vertical_nodes, np.arange(-4.0, 0.0 + 1e-9, 0.5))))
    return TensorMesh(
        [np.diff(horizontal_nodes), np.diff(vertical_nodes)],
        x0=[horizontal_nodes[0], vertical_nodes[0]],
    )


def add_bodies(mesh: TensorMesh, background: np.ndarray, width_m: float) -> tuple[np.ndarray, dict]:
    """两处异常体均以对应测点为中心，深度固定为 4 m。"""
    model = background.copy()
    footprints = {}
    for name, center in zip(("left", "right"), STATIONS_M, strict=True):
        lower = np.array([center - width_m / 2.0, -4.0])
        upper = np.array([center + width_m / 2.0, 0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BreakingChangeWarning)
            indices = np.asarray(model_builder.get_indices_block(lower, upper, mesh.cell_centers), dtype=int)
        if indices.size == 0:
            raise ValueError(f"{width_m:g} m 浅部体未被网格表示")
        model[indices] = np.log(1.0 / BODY_RESISTIVITIES[name])
        centers = mesh.cell_centers[indices]
        x_indices = np.searchsorted(mesh.nodes_x, centers[:, 0]) - 1
        z_indices = np.searchsorted(mesh.nodes_y, centers[:, 1]) - 1
        footprints[name] = {
            "cell_count": int(indices.size),
            "x_bounds_m": [float(np.min(mesh.nodes_x[x_indices])), float(np.max(mesh.nodes_x[x_indices + 1]))],
            "z_bounds_m": [float(np.min(mesh.nodes_y[z_indices])), float(np.max(mesh.nodes_y[z_indices + 1]))],
        }
    return model, footprints


def build_survey(mode: str, frequency_indices: tuple[int, ...]) -> nsem.survey.Survey:
    """对两处异常体上方测点构建指定模式和频率的阻抗测量。"""
    locations = np.column_stack((STATIONS_M, np.zeros_like(STATIONS_M)))
    orientation = "yx" if mode == "TE" else "xy"
    sources = []
    for index in frequency_indices:
        receivers = [
            nsem.receivers.Impedance(locations, orientation=orientation, component=component)
            for component in ("real", "imag")
        ]
        sources.append(nsem.sources.Planewave(receivers, FREQUENCIES_HZ[index]))
    return nsem.survey.Survey(sources)


def convert_response(dpred: np.ndarray, frequency_indices: tuple[int, ...]) -> dict:
    """从阻抗实虚部计算所选频率的视电阻率和相位。"""
    parts = dpred.reshape(len(frequency_indices), 2, STATIONS_M.size)
    impedance = -(parts[:, 0, :] + 1j * parts[:, 1, :])
    frequencies = FREQUENCIES_HZ[list(frequency_indices)]
    rho = np.abs(impedance) ** 2 / (VACUUM_PERMEABILITY * 2.0 * np.pi * frequencies[:, None])
    phase = np.mod(np.angle(impedance, deg=True), 180.0)
    return {"apparent_resistivity": rho, "phase_deg": phase}


def run_mode(mesh: TensorMesh, models: dict[str, np.ndarray], mode: str, frequency_indices: tuple[int, ...]) -> dict:
    """同一模式下共用网格、接收器和求解配置。"""
    simulation_type = (
        nsem.simulation.Simulation2DMagneticField
        if mode == "TE"
        else nsem.simulation.Simulation2DElectricField
    )
    simulation = simulation_type(
        mesh, survey=build_survey(mode, frequency_indices), sigmaMap=maps.ExpMap(mesh), storeInnerProduct=True
    )
    responses = {}
    for name, model in models.items():
        print(f"{mode}：计算 {name}", flush=True)
        responses[name] = convert_response(np.asarray(simulation.dpred(model), dtype=float), frequency_indices)
    baseline = responses["without_shallow"]
    differences = {}
    for name, response in responses.items():
        if name == "without_shallow":
            continue
        differences[name] = {
            "apparent_resistivity_ratio": (
                response["apparent_resistivity"] / baseline["apparent_resistivity"]
            ).tolist(),
            "phase_difference_deg": (
                (response["phase_deg"] - baseline["phase_deg"] + 90.0) % 180.0 - 90.0
            ).tolist(),
        }
    return differences


def main() -> None:
    """将尺寸敏感性结果写入独立诊断文件。"""
    parser = argparse.ArgumentParser(description="受控比较浅部体尺寸与 TE/TM 响应")
    parser.add_argument("--body-widths", type=float, nargs="+", default=[4.0, 100.0])
    parser.add_argument("--frequency-indices", type=int, nargs="+", default=[0, 19])
    parser.add_argument("--modes", nargs="+", choices=["TE", "TM"], default=["TE", "TM"])
    parser.add_argument("--body-cell-size", type=float, choices=[0.5, 1.0], default=1.0)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    if any(width <= 0 or width > 400 for width in args.body_widths):
        parser.error("浅部体宽度必须在 0～400 m 范围内")
    if any(index < 0 or index >= FREQUENCIES_HZ.size for index in args.frequency_indices):
        parser.error("频率索引超出已有频率数组范围")
    frequency_indices = tuple(args.frequency_indices)
    mesh = build_local_mesh(args.body_cell_size)
    without_shallow, _, _ = build_models(mesh)
    models = {"without_shallow": without_shallow}
    footprints = {}
    for width in args.body_widths:
        name = f"width_{width:g}_m"
        models[name], footprints[name] = add_bodies(mesh, without_shallow, width)
    report = {
        "mesh_cells": list(mesh.shape_cells),
        "frequencies_hz": FREQUENCIES_HZ[list(frequency_indices)].tolist(),
        "stations_m": STATIONS_M.tolist(),
        "body_height_m": 4.0,
        "body_cell_size_m": args.body_cell_size,
        "footprints": footprints,
        "modes": {},
    }
    output_dir = PROJECT_ROOT / "results" / "section_21_forward" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_file or output_dir / "shallow_size_local_mesh.json"
    for mode in args.modes:
        report["modes"][mode] = run_mode(mesh, models, mode, frequency_indices)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"诊断结果：{output_path}")


if __name__ == "__main__":
    main()
