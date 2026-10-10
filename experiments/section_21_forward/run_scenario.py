"""按配置运行独立浅部异常体正演或登记现有正演。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "simpeg"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mt_static.forward.scenario_pipeline import run_scenario


def main() -> None:
    """一个配置只生成一个独立场景目录，不覆盖已有结果。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="按浅部异常体场景配置执行二维 MT 正演")
    parser.add_argument("--scenario", type=Path, required=True, help="场景 JSON 配置路径")
    parser.add_argument(
        "--base", type=Path,
        default=PROJECT_ROOT / "configs" / "section_21" / "base_v1.json",
        help="公共模型 JSON 配置路径",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "results" / "section_21_forward" / "scenarios",
        help="独立场景的结果根目录",
    )
    parser.add_argument(
        "--import-existing", type=Path,
        help="只登记经过模型、网格与观测逐项核验的已有正演，不重新正演",
    )
    args = parser.parse_args()
    destination = run_scenario(
        base_path=args.base,
        scenario_path=args.scenario,
        output_root=args.output_root,
        cache_root=PROJECT_ROOT / "cache" / "forward" / "baselines",
        existing_forward=PROJECT_ROOT / "results" / "section_21_forward" / "structured" / "section_21_forward.npz",
        import_forward=args.import_existing,
    )
    print(f"Saved scenario: {destination}")


if __name__ == "__main__":
    main()
