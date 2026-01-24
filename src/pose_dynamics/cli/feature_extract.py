"""
pose_dynamics.cli.feature_extract

CLI entrypoint for feature extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_dynamics.features.api import run_feature_extract


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pose-dynamics feature-extract",
        description="Extract kinematic/geometry features from preprocessed pose data.",
    )
    p.add_argument(
        "--pose",
        dest="pose_path",
        required=True,
        type=Path,
        help="Input pose_clean.parquet from preprocess.",
    )
    p.add_argument(
        "--windows",
        dest="windows_path",
        required=True,
        type=Path,
        help="Input windows.parquet from preprocess.",
    )
    p.add_argument(
        "--config",
        dest="config_path",
        required=True,
        type=Path,
        help="Features config YAML file (features.yml).",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        type=Path,
        help="Output directory for features.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_feature_extract(
        pose_clean_path=args.pose_path,
        windows_path=args.windows_path,
        config=args.config_path,
        out_dir=args.out_dir,
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
