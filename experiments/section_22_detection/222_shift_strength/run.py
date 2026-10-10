"""在固定无浅部体 TM 数据上评估静态位移强度与基础逐频识别。"""

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
from reporting import save_case, save_summary
from strength import apply_static_factor, detection_boundary, strength_record


CONFIG_PATH = PROJECT_ROOT / "configs" / "section_22" / "shift_strength.json"
FORWARD_PATH = PROJECT_ROOT / "results" / "section_21_forward" / "structured" / "section_21_forward.npz"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "section_22" / "222_shift_strength"


def file_sha256(path: Path) -> str:
    """流式计算来源文件哈希，避免将整份正演数据读入字节缓存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs() -> tuple[dict, DetectionConfig, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], dict]:
    """核对 S0 来源、配置与正演数组，固定实验中除 Cs 外的所有条件。"""

    setup = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if setup["mode"] != "TM" or setup["noise"] != "none":
        raise ValueError("本实验仅使用无噪声 TM 基线")
    s0_path = PROJECT_ROOT / setup["source_s0_summary"]
    s0 = json.loads(s0_path.read_text(encoding="utf-8"))
    if s0["experiment_id"] != "S0" or s0["source_scenario"] != "without_shallow":
        raise ValueError("给定摘要不是当前无浅部体 S0 结果")
    if s0["mode"] != "TM" or s0["known_static_x_m"]:
        raise ValueError("S0 模式或阴性真值与本实验不一致")
    if Path(s0["source_forward_npz"]).resolve() != FORWARD_PATH.resolve():
        raise ValueError("S0 摘要所用正演路径与当前固定基线不一致")
    source_hash = file_sha256(FORWARD_PATH)
    if source_hash != s0["source_forward_sha256"]:
        raise ValueError("S0 摘要记录的正演 SHA256 与当前文件不一致")
    detection_path = PROJECT_ROOT / setup["detection_config"]
    detection_payload = json.loads(detection_path.read_text(encoding="utf-8"))["detection"]
    detection_config = DetectionConfig(**detection_payload)
    if asdict(detection_config) != s0["detection_config"]:
        raise ValueError("当前 2.2.1 参数与已保存 S0 参数不一致")
    factors = [float(value) for value in setup["static_factors"]]
    if len(factors) != 8 or len(set(factors)) != 8 or factors.count(1.0) != 1:
        raise ValueError("本实验应为七个不同强度加一个 Cs=1 对照")
    if any(not np.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("所有 Cs 必须是不同的有限正数")
    with np.load(FORWARD_PATH, allow_pickle=False) as source:
        stations = np.asarray(source["station_x_m"], dtype=float).copy()
        frequencies = np.asarray(source["frequencies_hz"], dtype=float).copy()
        rho_a = np.asarray(source["tm_without_shallow_apparent_resistivity"], dtype=float).copy()
        phase_deg = np.asarray(source["tm_without_shallow_phase_deg"], dtype=float).copy()
    if stations.size != 41 or frequencies.size != 20 or rho_a.shape != (20, 41):
        raise ValueError("测点、频率或视电阻率尺寸与固定 S0 基线不一致")
    if phase_deg.shape != rho_a.shape or not np.all(np.isfinite(phase_deg)):
        raise ValueError("S0 相位数组的尺寸或数值无效")
    target_x = [float(value) for value in setup["target_station_x_m"]]
    if target_x != [-5000.0, 5000.0]:
        raise ValueError("本实验只在 −5 km 和 +5 km 两个固定测点注入位移")
    target_indices = [int(np.argmin(np.abs(stations - value))) for value in target_x]
    if any(not np.isclose(stations[index], value, atol=1e-9, rtol=0)
           for index, value in zip(target_indices, target_x)):
        raise ValueError("目标位置与实际测点不重合")
    provenance = {
        "source_s0_summary": str(s0_path.resolve()),
        "source_s0_summary_sha256": file_sha256(s0_path),
        "source_forward_npz": str(FORWARD_PATH.resolve()),
        "source_forward_sha256": source_hash,
        "experiment_config": str(CONFIG_PATH.resolve()),
        "experiment_config_sha256": file_sha256(CONFIG_PATH),
        "detection_config": asdict(detection_config),
        "detection_config_sha256": file_sha256(detection_path),
        "mode": "TM",
        "noise": "none",
        "phase_policy": "原始无浅部体 TM 相位不变，识别算法仅输入视电阻率",
        "target_station_x_m": target_x,
        "station_count": int(stations.size),
        "frequency_count": int(frequencies.size),
        "metric_unit": "frequency_times_station",
        "final_static_sites": None,
        "final_classification_reason": "基础方法未定义跨频最终测点规则",
    }
    return setup, detection_config, stations, frequencies, rho_a, phase_deg, target_indices, provenance


def check_s0_control(result) -> None:
    """确认 Cs=1 结果逐项匹配当前已保存的 S0 阴性候选矩阵。"""

    path = PROJECT_ROOT / "results" / "section_22" / "221_site_types" / "table2_s0_no_static_negative" / "detection_diagnostics.npz"
    with np.load(path, allow_pickle=False) as saved:
        previous = np.asarray(saved["per_frequency_votes"], dtype=bool)
        if not np.array_equal(saved["station_x_m"], result.station_x_m):
            raise ValueError("S0 诊断中的测点与当前基线不一致")
        if not np.array_equal(saved["frequencies_hz"], result.frequencies_hz):
            raise ValueError("S0 诊断中的频率与当前基线不一致")
    if not np.array_equal(previous, result.per_frequency_votes) or np.any(result.per_frequency_votes):
        raise ValueError("Cs=1 未复现 S0 无候选结果，停止强度实验")


def main() -> None:
    """先完成全部计算与验收，再分别写入独立场景结果。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup, config, stations, frequencies, baseline_rho, phase_deg, target_indices, provenance = load_inputs()
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"2.2.2 结果目录已有内容，拒绝覆盖：{OUTPUT_ROOT}")
    control = detect_static_sites(stations, frequencies, baseline_rho, config)
    check_s0_control(control)
    print("Cs=1 阴性对照已复现 S0 无候选结果")
    completed = []
    for factor in (float(value) for value in setup["static_factors"]):
        shifted = apply_static_factor(baseline_rho, target_indices, factor)
        result = control if factor == 1.0 else detect_static_sites(stations, frequencies, shifted, config)
        record, alpha_min = strength_record(factor, result, target_indices)
        case_id = f"cs_{factor:.1f}".replace(".", "p")
        completed.append((case_id, record, alpha_min, shifted, result))
    records = [item[1] for item in completed]
    boundaries = {}
    for name, condition in (("cs_below_one", lambda cs: cs < 1), ("cs_above_one", lambda cs: cs > 1)):
        branch = [row for row in records if condition(row["cs"])]
        boundaries[name] = {
            "both_targets": detection_boundary(branch, None),
            "minus_5km": detection_boundary(branch, 0),
            "plus_5km": detection_boundary(branch, 1),
        }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for case_id, record, alpha_min, shifted, result in completed:
        save_case(
            OUTPUT_ROOT / case_id, record, result, alpha_min, baseline_rho,
            shifted, phase_deg, target_indices, provenance, plot_case_diagnostics,
        )
        print(f"已保存 {case_id}：TP={record['metrics']['tp']} FP={record['metrics']['fp']} FN={record['metrics']['fn']}")
    save_summary(OUTPUT_ROOT, records, boundaries, provenance)
    print(f"强度实验汇总：{OUTPUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
