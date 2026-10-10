"""论文 2.1 正演运行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "simpeg"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mt_static.forward.pipeline import run_forward


def main() -> None:
    """只写入新项目的 results 目录。"""
    parser = argparse.ArgumentParser(description="运行论文 2.1 的 TE/TM 二维物理正演")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results" / "section_21_forward" / "structured",
        help="结果输出目录",
    )
    parser.add_argument(
        "--reuse-unchanged-baseline", action="store_true",
        help="网格、深部模型及观测逐项一致时复用已有无浅部体阻抗",
    )
    args = parser.parse_args()
    summary = run_forward(args.output_dir, reuse_baseline=args.reuse_unchanged_baseline)
    print(f"正演完成；结果目录：{args.output_dir}")
    print(f"网格单元数：{summary['mesh']['total_cell_count']}")
    print(f"TE/TM 差异指标：{summary['response_differences']}")


if __name__ == "__main__":
    main()
