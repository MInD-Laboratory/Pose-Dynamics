"""
pose_dynamics.cli.rqa_params

CLI entrypoint for RQA parameter estimation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_dynamics.rqa.params import run_rqa_params


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pose-dynamics rqa-params",
        description="Estimate AMI/FNN/epsilon sensitivity on preprocessed windows.",
    )
    p.add_argument("--pose", dest="pose_path", required=True, type=Path)
    p.add_argument("--windows", dest="windows_path", required=True, type=Path)
    p.add_argument("--config", dest="config_path", required=True, type=Path)
    p.add_argument("--out", dest="out_dir", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_rqa_params(
        pose_clean_path=args.pose_path,
        windows_path=args.windows_path,
        config=args.config_path,
        out_dir=args.out_dir,
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
