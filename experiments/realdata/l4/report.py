"""在共同数据子集上计算可比拟合指标并保留分项残差。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def main() -> None:
    result_root = ROOT / "results/realdata/l4"
    data = np.load(ROOT / "data/realdata/l4/prepared.npz")
    setup = json.loads((result_root / "setup.json").read_text(encoding="utf-8"))
    summary_path = result_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = np.asarray(setup["selected_static_indices"], dtype=int)
    ordinary = np.setdiff1d(np.arange(data["stations_m"].size), selected)
    rho = data["rho_xy_ohm_m"]
    phase = data["phase_simpeg_deg"]
    rho_std = setup["inversion_parameters"]["rho_relative_error"] * rho
    phase_std = setup["inversion_parameters"]["phase_error_deg"]
    records = []
    for name in ("full", "phase", "shape"):
        if name not in summary["methods"]:
            raise RuntimeError(f"{name} 反演尚未完成")
        archive = np.load(result_root / f"inversion/{name}.npz")
        predicted = archive["predicted_full"].reshape(rho.shape[0], 2, rho.shape[1])
        rho_residual = (predicted[:, 0, :] - rho) / rho_std
        phase_residual = (predicted[:, 1, :] - phase) / phase_std
        common = {
            "all_site_phase_wrms": rms(phase_residual),
            "ordinary_site_rho_wrms": rms(rho_residual[:, ordinary]),
            "candidate_site_rho_diagnostic_wrms": rms(rho_residual[:, selected]),
            "candidate_rho_is_fitted": name == "full",
        }
        summary["methods"][name]["common_data_fit"] = common
        np.savez_compressed(
            result_root / f"inversion/{name}_residual_components.npz",
            rho_normalized=rho_residual,
            phase_normalized=phase_residual,
            ordinary_indices=ordinary,
            selected_indices=selected,
        )
        records.append({"method": name, **common})
    with (result_root / "common_data_fit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("三方法共同相位、普通点视电阻率和候选点诊断残差已保存")


if __name__ == "__main__":
    main()
