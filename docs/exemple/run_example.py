"""按建模、正演、反演三个阶段运行二维 MT 教学案例。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from discretize import TensorMesh


EXAMPLE_DIR = Path(__file__).resolve().parent

from forward import frequencies, simulate, station_positions, verify_receiver_convention
from grid_model import active_cells, build_mesh, build_true_model, check_geometry
from inverse import invert
from plotting import plot_forward_curves, plot_inversion_fit, plot_inversion_models, plot_model_grid


CONFIG_PATH = EXAMPLE_DIR / "config.json"
DEFAULT_OUTPUT = EXAMPLE_DIR / "output"


def config_hash() -> str:
    """用配置文件内容锁定三个阶段之间的输入。"""

    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def load_config() -> dict:
    """检查教学案例的物理、观测与网格参数。"""

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    body = config["anomaly"]
    if config["mode"] != "TM":
        raise ValueError("本教学案例只演示项目约定的 TM 模式")
    if body["x_min_m"] >= body["x_max_m"] or body["z_min_m"] >= body["z_max_m"] or body["z_max_m"] > 0:
        raise ValueError("异常体边界无效")
    if config["stations"]["spacing_m"] <= 0 or config["frequencies"]["minimum_hz"] <= 0:
        raise ValueError("测点间距和频率必须为正")
    if config["frequencies"]["maximum_hz"] <= config["frequencies"]["minimum_hz"]:
        raise ValueError("最大频率必须高于最小频率")
    if config["inversion"]["rho_relative_error"] <= 0 or config["inversion"]["phase_error_deg"] <= 0:
        raise ValueError("反演误差权重必须为正")
    return config


def save_json(path: Path, payload: dict) -> None:
    """保存可机器读取的非有限值检查过的摘要。"""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def mesh_from_nodes(x_nodes: np.ndarray, z_nodes: np.ndarray) -> TensorMesh:
    """仅根据上一阶段存储的节点重建张量网格。"""

    return TensorMesh([np.diff(x_nodes), np.diff(z_nodes)], x0=[x_nodes[0], z_nodes[0]])


def read_stage(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    """拒绝将旧配置产生的中间文件用于当前阶段。"""

    if not path.exists():
        raise FileNotFoundError(f"先运行前一阶段，缺少 {path}")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: np.asarray(saved[key]).copy() for key in saved.files}
    if str(arrays["config_sha256"]) != expected_hash:
        raise ValueError(f"{path.name} 与当前 config.json 不一致，请从 model 阶段重跑")
    return arrays


def run_model(config: dict, output: Path) -> None:
    """阶段一：生成真模型、正反演网格和测点图。"""

    forward_mesh = build_mesh(config, "forward")
    inversion_mesh = build_mesh(config, "inversion")
    true_model, block = build_true_model(forward_mesh, config)
    inverse_block = build_true_model(inversion_mesh, config)[1]
    forward_geometry = check_geometry(forward_mesh, block, config)
    inverse_geometry = check_geometry(inversion_mesh, inverse_block, config)
    active = active_cells(inversion_mesh, config)
    if not np.all(active[inverse_block]):
        raise AssertionError("反演活动域没有覆盖低阻体")
    stations = station_positions(config)
    np.savez_compressed(
        output / "model.npz",
        config_sha256=config_hash(),
        forward_x_nodes_m=forward_mesh.nodes_x,
        forward_z_nodes_m=forward_mesh.nodes_y,
        inversion_x_nodes_m=inversion_mesh.nodes_x,
        inversion_z_nodes_m=inversion_mesh.nodes_y,
        true_log_conductivity=true_model,
        anomaly_cells=block,
        inversion_active_cells=active,
        station_x_m=stations,
    )
    plot_model_grid(forward_mesh, inversion_mesh, true_model, stations, config, output / "01_模型与网格.png")
    summary = {
        "config_sha256": config_hash(),
        "mode": config["mode"],
        "model": {"background_resistivity_ohm_m": config["background_resistivity_ohm_m"],
                  "anomaly": config["anomaly"]},
        "forward_mesh": {"shape": list(forward_mesh.shape_cells), "cells": int(forward_mesh.n_cells),
                         "anomaly_actual": forward_geometry},
        "inversion_mesh": {"shape": list(inversion_mesh.shape_cells), "cells": int(inversion_mesh.n_cells),
                           "active_cells": int(active.sum()), "anomaly_actual": inverse_geometry},
        "stations_m": stations.tolist(),
        "completed_steps": ["model"],
    }
    save_json(output / "summary.json", summary)
    print("建模与网格阶段完成", flush=True)


def run_forward(config: dict, output: Path) -> None:
    """阶段二：对均匀背景和低阻体模型分别做 TM 正演。"""

    model = read_stage(output / "model.npz", config_hash())
    mesh = mesh_from_nodes(model["forward_x_nodes_m"], model["forward_z_nodes_m"])
    stations = model["station_x_m"]
    frequency_values = frequencies(config)
    uniform = np.full(mesh.n_cells, np.log(1.0 / config["background_resistivity_ohm_m"]))
    print("正演均匀背景……", flush=True)
    baseline = simulate(mesh, uniform, frequency_values, stations)
    print("正演单低阻体……", flush=True)
    affected = simulate(mesh, model["true_log_conductivity"], frequency_values, stations)
    direct, convention = verify_receiver_convention(
        mesh, model["true_log_conductivity"], frequency_values, stations, affected,
    )
    np.savez_compressed(
        output / "forward.npz",
        config_sha256=config_hash(),
        frequencies_hz=frequency_values,
        station_x_m=stations,
        baseline_impedance=baseline["impedance"],
        baseline_apparent_resistivity=baseline["apparent_resistivity"],
        baseline_phase_display_deg=baseline["phase_display_deg"],
        anomaly_impedance=affected["impedance"],
        anomaly_apparent_resistivity=affected["apparent_resistivity"],
        anomaly_phase_raw_deg=affected["phase_raw_deg"],
        anomaly_phase_display_deg=affected["phase_display_deg"],
        observed_rho_phase=direct,
    )
    plot_forward_curves(frequency_values, stations, baseline, affected, output / "02_正演曲线.png")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary["frequencies_hz"] = frequency_values.tolist()
    summary["data_count"] = int(direct.size)
    summary["receiver_convention_check"] = convention
    summary["baseline_rho_median_ohm_m"] = float(np.median(baseline["apparent_resistivity"]))
    summary["completed_steps"] = ["model", "forward"]
    save_json(output / "summary.json", summary)
    print("正演与曲线阶段完成", flush=True)


def run_inversion(config: dict, output: Path) -> None:
    """阶段三：读取正演观测并用粗网格恢复低阻体。"""

    model = read_stage(output / "model.npz", config_hash())
    data = read_stage(output / "forward.npz", config_hash())
    mesh = mesh_from_nodes(model["inversion_x_nodes_m"], model["inversion_z_nodes_m"])
    result = invert(mesh, config, data["frequencies_hz"], data["station_x_m"], data["observed_rho_phase"])
    np.savez_compressed(
        output / "inversion.npz",
        config_sha256=config_hash(),
        mesh_x_nodes_m=mesh.nodes_x,
        mesh_z_nodes_m=mesh.nodes_y,
        active_cells=result["active_cells"],
        observed=result["observed"],
        standard_deviation=result["standard_deviation"],
        predicted=result["predicted"],
        recovered_log_conductivity=result["recovered_log_conductivity"],
    )
    plot_inversion_models(mesh, result["recovered_log_conductivity"], config, output / "03_反演模型.png")
    plot_inversion_fit(data["frequencies_hz"], data["station_x_m"],
                       result["observed"], result["predicted"], output / "04_反演曲线拟合.png")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary["inversion"] = {
        "errors": {"rho_relative": config["inversion"]["rho_relative_error"],
                   "phase_deg": config["inversion"]["phase_error_deg"], "injected_noise": "none"},
        "max_iter": config["inversion"]["max_iter"],
        "cg_maxiter": config["inversion"]["cg_maxiter"],
        "initial_beta": config["inversion"]["initial_beta"],
        "initial_metrics": result["initial_metrics"],
        "recovered_metrics": result["recovered_metrics"],
        "phi_d": result["phi_d"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    summary["completed_steps"] = ["model", "forward", "inversion"]
    save_json(output / "summary.json", summary)
    print("反演与结果绘图阶段完成", flush=True)


def main() -> None:
    """支持一次执行全部流程或按教程逐阶段执行。"""

    parser = argparse.ArgumentParser(description="单低阻体二维 MT 正反演教学案例")
    parser.add_argument("--step", choices=("model", "forward", "inversion", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.step in ("model", "all"):
        run_model(config, output)
    if args.step in ("forward", "all"):
        run_forward(config, output)
    if args.step in ("inversion", "all"):
        run_inversion(config, output)
    print(f"结果目录：{output}", flush=True)


if __name__ == "__main__":
    main()
