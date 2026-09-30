"""直接读取现有正演数据并重新绘制论文 2.1 图件。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mt_static.forward.plotting import plot_results


def main() -> None:
    """只执行绘图，不调用 SimPEG 正演。"""
    default_dir = PROJECT_ROOT / "results" / "section_21_forward" / "structured"
    parser = argparse.ArgumentParser(description="读取现有数据，重绘论文 2.1 正演图")
    parser.add_argument("--input", type=Path, default=default_dir / "section_21_forward.npz")
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    args = parser.parse_args()

    with np.load(args.input) as saved:
        frequencies_hz = saved["frequencies_hz"].copy()
        station_x_m = saved["station_x_m"].copy()
        responses = {
            mode: {
                model_name: {
                    field: saved[f"{mode.lower()}_{model_name}_{field}"].copy()
                    for field in ("apparent_resistivity", "phase_deg")
                }
                for model_name in ("without_shallow", "with_shallow")
            }
            for mode in ("TE", "TM")
        }
    output_paths = plot_results(responses, args.output_dir, frequencies_hz, station_x_m)
    for path in output_paths:
        print(f"已绘制：{path}")


if __name__ == "__main__":
    main()
