"""只读取已保存的独立场景响应，重绘核对图。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mt_static.forward.scenario_config import load_base_config, load_scenario_config
from mt_static.forward.scenario_plotting import plot_scenario_results


def file_sha256(path: Path) -> str:
    """重绘前核对原场景配置未发生变动。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """核对配置哈希后，仅更新该场景的绘图文件与图件清单。"""

    parser = argparse.ArgumentParser(description="重绘独立正演场景的模型、拟断面和曲线")
    parser.add_argument("--scenario", type=Path, required=True, help="场景 JSON 配置路径")
    parser.add_argument(
        "--base", type=Path,
        default=PROJECT_ROOT / "configs" / "section_21" / "base_v1.json",
    )
    args = parser.parse_args()
    base = load_base_config(args.base)
    scenario = load_scenario_config(args.scenario, base["base_id"])
    destination = PROJECT_ROOT / "results" / "section_21_forward" / "scenarios" / scenario["scenario_id"]
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["base_config_sha256"] != file_sha256(args.base):
        raise ValueError("公共模型配置已改变，不能据此重绘旧正演")
    if manifest["scenario_config_sha256"] != file_sha256(args.scenario):
        raise ValueError("场景配置已改变，不能据此重绘旧正演")
    with np.load(destination / "forward.npz", allow_pickle=False) as saved:
        frequencies = np.asarray(saved["frequencies_hz"], dtype=float)
        stations = np.asarray(saved["station_x_m"], dtype=float)
        responses = {
            mode: {
                model_name: {
                    field: np.asarray(saved[f"{mode.lower()}_{model_name}_{field}"], dtype=float)
                    for field in ("apparent_resistivity", "phase_deg")
                }
                for model_name in ("without_shallow", "with_shallow")
            }
            for mode in ("TE", "TM")
        }
    figures_dir = destination / "figures"
    previous = [destination / name for name in manifest["figures"]]
    current = plot_scenario_results(base, scenario, responses, frequencies, stations, figures_dir)
    current_set = {path.resolve() for path in current}
    for path in previous:
        resolved = path.resolve()
        if resolved.parent == figures_dir.resolve() and resolved not in current_set and resolved.exists():
            resolved.unlink()
    manifest["figures"] = [str(path.relative_to(destination)) for path in current]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Redrawn figures: {destination}")


if __name__ == "__main__":
    main()
