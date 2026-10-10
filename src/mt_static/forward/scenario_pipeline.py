"""按场景配置运行或登记二维正演，并保存独立可追溯结果。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .mesh import build_mesh
from .scenario_config import load_base_config, load_scenario_config
from .scenario_model import build_scenario_models
from .scenario_plotting import plot_scenario_results
from .settings import FREQUENCIES_HZ, STATION_X_M
from .simulation import simulate_mode


def file_sha256(path: Path) -> str:
    """计算输入文件的哈希，便于结果溯源。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_key(mesh, baseline: np.ndarray) -> str:
    """对决定基线响应的网格、模型和观测数组计算缓存键。"""

    digest = hashlib.sha256(b"section21-v1-TE-Zyx-TM-Zxy")
    for values in (mesh.nodes_x, mesh.nodes_y, baseline, FREQUENCIES_HZ, STATION_X_M):
        array = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _expected_arrays(mesh, baseline: np.ndarray) -> dict[str, np.ndarray]:
    """定义复用正演时必须逐项完全相同的基础数组。"""

    return {
        "frequencies_hz": FREQUENCIES_HZ,
        "station_x_m": STATION_X_M,
        "mesh_x_nodes_m": mesh.nodes_x,
        "mesh_z_nodes_m": mesh.nodes_y,
        "without_shallow_log_conductivity": baseline,
    }


def _read_baselines(path: Path, expected: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """只在网格、模型和测量严格一致时读取 TE/TM 基线。"""

    with np.load(path, allow_pickle=False) as saved:
        for name, values in expected.items():
            if name not in saved or not np.array_equal(saved[name], values):
                raise ValueError(f"已有基线的 {name} 与当前场景不一致")
        baselines = {}
        expected_size = 2 * FREQUENCIES_HZ.size * STATION_X_M.size
        for mode in ("TE", "TM"):
            name = f"{mode.lower()}_without_shallow_dpred"
            if name not in saved:
                raise ValueError(f"已有基线缺少 {name}")
            values = np.asarray(saved[name], dtype=float)
            if values.size != expected_size or not np.all(np.isfinite(values)):
                raise ValueError(f"已有基线的 {name} 无效")
            baselines[mode] = values.copy()
    return baselines


def load_baseline_cache(
    cache_root: Path, key: str, mesh, baseline: np.ndarray, existing_forward: Path
) -> tuple[dict[str, np.ndarray], str]:
    """优先复用逐项核对过的缓存或现有 2.1 正演基线。"""

    expected = _expected_arrays(mesh, baseline)
    cached = cache_root / key / "baseline.npz"
    if cached.exists():
        return _read_baselines(cached, expected), str(cached)
    if existing_forward.exists():
        baselines = _read_baselines(existing_forward, expected)
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cached,
            **expected,
            te_without_shallow_dpred=baselines["TE"],
            tm_without_shallow_dpred=baselines["TM"],
        )
        return baselines, str(existing_forward)
    return {}, "computed"


def _response_from_saved(saved, mode: str, model_name: str) -> dict[str, np.ndarray]:
    """按统一字段约定读入现有正演响应。"""

    names = ("dpred", "raw_impedance", "display_impedance", "apparent_resistivity", "phase_deg")
    response = {}
    for field in names:
        name = f"{mode.lower()}_{model_name}_{field}"
        if name not in saved:
            raise ValueError(f"已有正演缺少 {name}")
        response[field] = np.asarray(saved[name]).copy()
    expected_shape = (FREQUENCIES_HZ.size, STATION_X_M.size)
    for field in ("raw_impedance", "display_impedance", "apparent_resistivity", "phase_deg"):
        if response[field].shape != expected_shape or not np.all(np.isfinite(response[field])):
            raise ValueError(f"已有正演的 {mode}/{model_name}/{field} 尺寸或数值无效")
    return response


def import_existing_forward(path: Path, mesh, baseline: np.ndarray, model: np.ndarray) -> dict:
    """只在两套地电模型与观测完全一致时登记既有正演。"""

    with np.load(path, allow_pickle=False) as saved:
        for name, values in {
            **_expected_arrays(mesh, baseline),
            "with_shallow_log_conductivity": model,
        }.items():
            if name not in saved or not np.array_equal(saved[name], values):
                raise ValueError(f"既有正演的 {name} 与场景配置不一致，不能登记")
        return {
            mode: {
                model_name: _response_from_saved(saved, mode, model_name)
                for model_name in ("without_shallow", "with_shallow")
            }
            for mode in ("TE", "TM")
        }


def summarize_scenario_responses(responses: dict, body_descriptions: list[dict]) -> dict:
    """按每个浅部体中心报告 TM/TE 视电阻率和相位变化。"""

    summary = {}
    for mode, mode_responses in responses.items():
        baseline = mode_responses["without_shallow"]
        affected = mode_responses["with_shallow"]
        log_ratio = np.log10(affected["apparent_resistivity"] / baseline["apparent_resistivity"])
        phase_delta = (affected["phase_deg"] - baseline["phase_deg"] + 90.0) % 180.0 - 90.0
        at_bodies = {}
        for body in body_descriptions:
            lower, upper = body["specified_bounds_m"]
            center = (lower[0] + upper[0]) / 2.0
            index = int(np.argmin(np.abs(STATION_X_M - center)))
            at_bodies[body["body_id"]] = {
                "center_x_m": center,
                "nearest_station_x_m": float(STATION_X_M[index]),
                "geometric_mean_rho_ratio": float(10.0 ** np.mean(log_ratio[:, index])),
                "std_log10_rho_ratio": float(np.std(log_ratio[:, index])),
                "median_absolute_phase_difference_deg": float(np.median(np.abs(phase_delta[:, index]))),
            }
        summary[mode] = {
            "median_absolute_log10_rho_ratio_all": float(np.median(np.abs(log_ratio))),
            "median_absolute_phase_difference_all_deg": float(np.median(np.abs(phase_delta))),
            "at_shallow_bodies": at_bodies,
        }
    return summary


def run_scenario(
    base_path: Path,
    scenario_path: Path,
    output_root: Path,
    cache_root: Path,
    existing_forward: Path,
    import_forward: Path | None = None,
) -> Path:
    """执行新场景或登记完全一致的现有结果，不覆盖同名场景。"""

    base = load_base_config(base_path)
    scenario = load_scenario_config(scenario_path, base["base_id"])
    destination = output_root / scenario["scenario_id"]
    if destination.exists():
        raise FileExistsError(f"场景结果已存在：{destination}；请使用新的 scenario_id")
    mesh = build_mesh()
    baseline, model, body_info = build_scenario_models(mesh, base, scenario)
    key = baseline_key(mesh, baseline)
    if import_forward is not None:
        responses = import_existing_forward(import_forward, mesh, baseline, model)
        baseline_source = str(import_forward)
        run_mode = "validated_import"
    else:
        baselines, baseline_source = load_baseline_cache(
            cache_root, key, mesh, baseline, existing_forward
        )
        models = {"without_shallow": baseline, "with_shallow": model}
        responses = {
            mode: simulate_mode(mesh, models, mode, baseline_dpred=baselines.get(mode))
            for mode in ("TE", "TM")
        }
        if not baselines:
            cache_path = cache_root / key / "baseline.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                **_expected_arrays(mesh, baseline),
                te_without_shallow_dpred=responses["TE"]["without_shallow"]["dpred"],
                tm_without_shallow_dpred=responses["TM"]["without_shallow"]["dpred"],
            )
        run_mode = "physical_forward"
    output_arrays = {
        **_expected_arrays(mesh, baseline),
        "with_shallow_log_conductivity": model,
    }
    for mode, mode_responses in responses.items():
        for model_name, data in mode_responses.items():
            for field, values in data.items():
                output_arrays[f"{mode.lower()}_{model_name}_{field}"] = values
    destination.mkdir(parents=True)
    np.savez_compressed(destination / "forward.npz", **output_arrays)
    figures = plot_scenario_results(
        base, scenario, responses, FREQUENCIES_HZ, STATION_X_M, destination / "figures"
    )
    manifest = {
        "schema_version": 1,
        "scenario_id": scenario["scenario_id"],
        "base_id": base["base_id"],
        "run_mode": run_mode,
        "base_config_path": str(base_path.resolve()),
        "base_config_sha256": file_sha256(base_path),
        "scenario_config_path": str(scenario_path.resolve()),
        "scenario_config_sha256": file_sha256(scenario_path),
        "import_source": None if import_forward is None else str(import_forward.resolve()),
        "import_source_sha256": None if import_forward is None else file_sha256(import_forward),
        "baseline_key": key,
        "baseline_source": baseline_source,
        "station_count": int(STATION_X_M.size),
        "frequency_count": int(FREQUENCIES_HZ.size),
        "mesh_shape_cells": list(mesh.shape_cells),
        "body_geometry": body_info,
        "response_differences": summarize_scenario_responses(responses, body_info["shallow_bodies"]),
        "figures": [str(path.relative_to(destination)) for path in figures],
        "data_conventions": {
            "te": "Zyx=Ey/Hx; Simulation2DMagneticField",
            "tm": "Zxy=Ex/Hy; Simulation2DElectricField",
            "dpred_order": "ascending frequency; all stations real, then all stations imaginary",
            "apparent_resistivity": "|Z|^2/(mu_0*2*pi*f)",
            "phase": "arg(-Z) modulo 180 degrees",
            "noise": "none",
            "injected_static_shift": "none; only shallow conductivity cells differ",
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
