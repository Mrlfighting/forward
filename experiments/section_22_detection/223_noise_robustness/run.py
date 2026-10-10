"""固定双浅部体 TM 正演，执行视电阻率噪声条件下的基础识别。"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
PLOT_ROOT = PROJECT_ROOT / "experiments" / "section_22_detection" / "221_site_types"
for source in (SOURCE_ROOT, PLOT_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from mt_static.detection import DetectionConfig, detect_static_sites
from plot_diagnostics import plot_case_diagnostics
from metrics import trial_metrics
from noise import apply_noise, relative_gaussian_factor
from reporting import aggregate_records, save_summary, save_trial, write_json


CONFIG_PATH = PROJECT_ROOT / "configs" / "section_22" / "noise_robustness.json"
FORWARD_PATH = PROJECT_ROOT / "results" / "section_21_forward" / "structured" / "section_21_forward.npz"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "section_22" / "223_noise_robustness"


def file_sha256(path: Path) -> str:
    """读取来源文件内容哈希，用于复现时的严格核对。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experiment() -> tuple[dict, DetectionConfig, dict, np.ndarray, np.ndarray, dict, dict, list[int]]:
    """核对双浅部体、S0 对照及 2.2.1 配置均来自同一已验证正演。"""

    setup = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    levels = [float(value) for value in setup["noise_fractions"]]
    if levels != [0.0, 0.02, 0.05, 0.10, 0.15]:
        raise ValueError("噪声水平与已确认方案不一致")
    if int(setup["nonzero_repetitions"]) != 30 or int(setup["representative_repetition"]) != 0:
        raise ValueError("重复次数或诊断图选样与已确认方案不一致")
    if setup["mode"] != "TM" or float(setup["phase_noise_deg"]) != 0.0:
        raise ValueError("本节只测试 TM 视电阻率噪声，不加入相位噪声")
    if setup["nonpositive_policy"] != "redraw_individual_sample":
        raise ValueError("非正值处理必须为单样本重抽")
    if setup["target_station_x_m"] != [-5000.0, 5000.0] or float(setup["region_tolerance_m"]) != 500.0:
        raise ValueError("真值位置或区域容差与固定方案不一致")
    detection_path = PROJECT_ROOT / setup["detection_config"]
    config = DetectionConfig(**json.loads(detection_path.read_text(encoding="utf-8"))["detection"])
    s0_path = PROJECT_ROOT / setup["source_s0_summary"]
    s2_path = PROJECT_ROOT / setup["source_s2_summary"]
    s0 = json.loads(s0_path.read_text(encoding="utf-8"))
    s2 = json.loads(s2_path.read_text(encoding="utf-8"))
    source_hash = file_sha256(FORWARD_PATH)
    for record, name, scenario in ((s0, "S0", "without_shallow"), (s2, "S2", "with_shallow")):
        if record["experiment_id"] != name or record["source_scenario"] != scenario or record["mode"] != "TM":
            raise ValueError(f"{name} 摘要的正演情形或模式不一致")
        if Path(record["source_forward_npz"]).resolve() != FORWARD_PATH.resolve():
            raise ValueError(f"{name} 摘要的正演路径不一致")
        if record["source_forward_sha256"] != source_hash or record["detection_config"] != asdict(config):
            raise ValueError(f"{name} 来源哈希或 2.2.1 参数与当前文件不一致")
    if s0["known_static_x_m"] != [] or s2["known_static_x_m"] != [-5000.0, 5000.0]:
        raise ValueError("S0/S2 真值位置与噪声实验不一致")
    with np.load(FORWARD_PATH, allow_pickle=False) as saved:
        stations = np.asarray(saved["station_x_m"], dtype=float).copy()
        frequencies = np.asarray(saved["frequencies_hz"], dtype=float).copy()
        rho = {
            arm: np.asarray(saved[f"tm_{arm}_apparent_resistivity"], dtype=float).copy()
            for arm in ("without_shallow", "with_shallow")
        }
        phase = {
            arm: np.asarray(saved[f"tm_{arm}_phase_deg"], dtype=float).copy()
            for arm in ("without_shallow", "with_shallow")
        }
    if stations.size != 41 or frequencies.size != 20:
        raise ValueError("测点或频率数量与现有 2.1 正演不一致")
    for arm in rho:
        if rho[arm].shape != (20, 41) or phase[arm].shape != (20, 41):
            raise ValueError(f"{arm} 的视电阻率或相位形状无效")
        if np.any(rho[arm] <= 0) or not np.all(np.isfinite(rho[arm])) or not np.all(np.isfinite(phase[arm])):
            raise ValueError(f"{arm} 的原始响应包含无效数值")
    target_indices = [int(np.argmin(np.abs(stations - value))) for value in setup["target_station_x_m"]]
    if any(not np.isclose(stations[index], value, atol=1e-9, rtol=0)
           for index, value in zip(target_indices, setup["target_station_x_m"])):
        raise ValueError("双浅部体中心与实际测点不重合")
    log_ratio = np.log10(rho["with_shallow"][:, target_indices] / rho["without_shallow"][:, target_indices])
    provenance = {
        "source_forward_npz": str(FORWARD_PATH.resolve()),
        "source_forward_sha256": source_hash,
        "source_s0_summary": str(s0_path.resolve()),
        "source_s0_sha256": file_sha256(s0_path),
        "source_s2_summary": str(s2_path.resolve()),
        "source_s2_sha256": file_sha256(s2_path),
        "experiment_config": str(CONFIG_PATH.resolve()),
        "experiment_config_sha256": file_sha256(CONFIG_PATH),
        "detection_config": asdict(config),
        "detection_config_sha256": file_sha256(detection_path),
        "noise_model": setup["noise_model"],
        "rng_policy": "每个重复使用 SeedSequence([master_seed, repetition, 0])；相同 Z 在噪声级别及 S0/S2 间复用",
        "master_seed": int(setup["master_seed"]),
        "static_effect_policy": "固定双浅部体 TM 物理正演响应；不额外乘人工 Cs",
        "geometric_mean_rho_ratio_at_targets": (10.0 ** np.mean(log_ratio, axis=0)).tolist(),
        "std_log10_rho_ratio_at_targets": np.std(log_ratio, axis=0).tolist(),
        "phase_noise_deg": 0.0,
        "target_station_x_m": setup["target_station_x_m"],
        "region_tolerance_m": setup["region_tolerance_m"],
        "metric_unit": "frequency_times_station",
        "final_static_sites": None,
        "final_classification_reason": "基础方法仅输出逐频候选，未定义跨频最终测点分类",
    }
    return setup, config, provenance, stations, frequencies, rho, phase, target_indices


def check_zero_noise(result, arm: str, target_indices: list[int]) -> None:
    """逐项核对 0% 噪声与已保存 S0/S2 的候选矩阵。"""

    folder = "table2_s2_multiple_discrete_static_sites" if arm == "with_shallow" else "table2_s0_no_static_negative"
    path = PROJECT_ROOT / "results" / "section_22" / "221_site_types" / folder / "detection_diagnostics.npz"
    with np.load(path, allow_pickle=False) as previous:
        if not np.array_equal(previous["station_x_m"], result.station_x_m):
            raise ValueError(f"0% {arm} 测点数组未复现")
        if not np.array_equal(previous["frequencies_hz"], result.frequencies_hz):
            raise ValueError(f"0% {arm} 频率数组未复现")
        if not np.array_equal(previous["per_frequency_votes"], result.per_frequency_votes):
            raise ValueError(f"0% {arm} 逐频候选未复现")
    metrics = trial_metrics(result, target_indices, arm == "with_shallow", 500.0)
    expected = (20, 22, 20) if arm == "with_shallow" else (0, 0, 0)
    if (metrics["tp"], metrics["fp"], metrics["fn"]) != expected:
        raise ValueError(f"0% {arm} 的 TP/FP/FN 与已验证结果不一致")


def main() -> None:
    """先完成 S0/S2 门槛，再运行同底样配对噪声重复并保存完整数值。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup, config, provenance, stations, frequencies, rho, phase, target_indices = load_experiment()
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"噪声实验结果目录已有文件，拒绝覆盖：{OUTPUT_ROOT}")
    zero_results = {}
    for arm in ("without_shallow", "with_shallow"):
        result = detect_static_sites(stations, frequencies, rho[arm], config)
        check_zero_noise(result, arm, target_indices)
        zero_results[arm] = result
    print("0% 噪声已逐频复现 S0 与双浅部体 S2")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "manifest.json"
    write_json(manifest_path, {"status": "running", "provenance": provenance, "setup": setup})
    records = []
    levels = [float(value) for value in setup["noise_fractions"]]
    repeats = int(setup["nonzero_repetitions"])
    master_seed = int(setup["master_seed"])
    for level_index, sigma in enumerate(levels):
        indices = [0] if sigma == 0 else range(repeats)
        for repetition in indices:
            if sigma == 0:
                standard_normal = np.zeros((frequencies.size, stations.size), dtype=float)
                factors = np.ones_like(standard_normal)
                redraw_count = 0
            else:
                generator = np.random.default_rng(np.random.SeedSequence([master_seed, repetition, 0]))
                standard_normal = generator.standard_normal((frequencies.size, stations.size))
                repair = np.random.default_rng(np.random.SeedSequence([master_seed, repetition, level_index, 991]))
                factors, redraw_count = relative_gaussian_factor(standard_normal, sigma, repair)
            relative = factors - 1.0
            realized_mean = float(np.mean(relative))
            realized_rms = float(np.sqrt(np.mean(relative**2)))
            for arm in ("without_shallow", "with_shallow"):
                noisy_rho = apply_noise(rho[arm], factors)
                result = zero_results[arm] if sigma == 0 else detect_static_sites(stations, frequencies, noisy_rho, config)
                metrics = trial_metrics(result, target_indices, arm == "with_shallow", float(setup["region_tolerance_m"]))
                record = {
                    "arm": arm,
                    "noise_fraction": sigma,
                    "repetition": repetition,
                    "metrics": metrics,
                    "redrawn_sample_count": redraw_count,
                    "realized_relative_mean": realized_mean,
                    "realized_relative_rms": realized_rms,
                    "source_forward_sha256": provenance["source_forward_sha256"],
                    "detection_config_sha256": provenance["detection_config_sha256"],
                    "random_seed_identity": None if sigma == 0 else [master_seed, repetition, 0],
                    "repair_seed_identity": None if sigma == 0 else [master_seed, repetition, level_index, 991],
                    "phase_noise_deg": 0.0,
                }
                destination = OUTPUT_ROOT / f"noise_{int(round(sigma * 100)):02d}pct" / arm / f"rep_{repetition:03d}"
                make_figures = sigma == 0 or repetition == int(setup["representative_repetition"])
                save_trial(
                    destination, result, record, rho[arm], noisy_rho, phase[arm],
                    standard_normal, factors, target_indices, plot_case_diagnostics, make_figures,
                )
                records.append(record)
        print(f"已完成相对噪声 {sigma * 100:.0f}%：每组 {len(indices)} 次")
    groups = aggregate_records(records, levels)
    save_summary(OUTPUT_ROOT, records, groups, provenance)
    write_json(manifest_path, {
        "status": "complete",
        "provenance": provenance,
        "setup": setup,
        "trial_count": len(records),
        "summary_files": ["summary.json", "summary.csv", "repetitions.csv", "噪声水平与识别鲁棒性.png"],
    })
    print(f"已保存 {len(records)} 次逐频识别：{OUTPUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
