"""
pose_dynamics.cli.list_keypoints

List keypoints available in pose_clean.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pose-dynamics list-keypoints",
        description="List keypoints available in pose_clean.parquet.",
    )
    p.add_argument("--pose", dest="pose_path", required=True, type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.pose_path)
    kps = sorted(df["keypoint"].dropna().unique().tolist())
    for kp in kps:
        print(kp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
