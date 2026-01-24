"""
pose_dynamics.cli.preprocess

CLI entrypoint for preprocessing canonical pose data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_dynamics.preprocess.api import run_preprocess


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pose-dynamics preprocess",
        description="Preprocess canonical pose parquet into clean signals and windows.",
    )
    p.add_argument(
        "--in",
        dest="pose_path",
        required=True,
        type=Path,
        help="Input pose.parquet from ingest.",
    )
    p.add_argument(
        "--recording",
        dest="recording_path",
        required=True,
        type=Path,
        help="Input recording.json from ingest.",
    )
    p.add_argument(
        "--config",
        dest="config_path",
        required=True,
        type=Path,
        help="Preprocess config YAML file (preprocess.yml).",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        type=Path,
        help="Output directory for preprocess artifacts.",
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

    run_preprocess(
        pose_path=args.pose_path,
        recording_path=args.recording_path,
        config=args.config_path,
        out_dir=args.out_dir,
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
