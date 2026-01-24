"""
pose_dynamics.cli.ingest_csv

CLI entrypoint for ingesting wide pose CSV files.

Usage examples:
  pose-dynamics ingest-csv --in data/pose_csv --out artifacts/ingest/run_001
  pose-dynamics ingest-csv --in data/pose_csv --out artifacts/ingest/run_001 --fps 30
  pose-dynamics ingest-csv --in data/pose_csv --out artifacts/ingest/run_001 --fps 30 --pattern "*.csv"

Notes:
- If a CSV has a `time` column, fps is not needed for that file.
- If a CSV lacks `time`, it must have `frame` and you must supply --fps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_dynamics.io.csv_pose import ingest_pose_csv_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pose-dynamics ingest-csv",
        description="Ingest wide pose CSV files (one CSV per trial) into canonical long-form parquet.",
    )
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        type=Path,
        help="Input directory containing CSV files (one CSV per trial).",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        type=Path,
        help="Output directory to write pose.parquet, recording.json, qc_ingest.json.",
    )
    p.add_argument(
        "--fps",
        dest="fps",
        type=float,
        default=None,
        help="Frames per second. Required for trials that do not include a `time` column and use `frame` instead.",
    )
    p.add_argument(
        "--pattern",
        dest="pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for input CSVs (default: *.csv).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    ingest_pose_csv_dir(
        in_path=args.in_path,
        out_dir=args.out_dir,
        fps=args.fps,
        glob_pattern=args.pattern,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
