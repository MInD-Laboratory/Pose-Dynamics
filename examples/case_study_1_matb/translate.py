"""
Translate OpenMATB raw pose CSVs into repo ingest format.

Input structure:
  <root>/477_M/<trial>.csv

Columns look like: x1, y1, prob1, x2, y2, prob2, ...
Output format uses: x_1, y_1, prob_1, ... and includes a `frame` column
if no `time` column is present.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

_X_RE = re.compile(r"^x(\d+)$", re.IGNORECASE)
_Y_RE = re.compile(r"^y(\d+)$", re.IGNORECASE)
_Z_RE = re.compile(r"^z(\d+)$", re.IGNORECASE)
_PROB_RE = re.compile(r"^prob(\d+)$", re.IGNORECASE)
_CONF_RE = re.compile(r"^conf(\d+)$", re.IGNORECASE)


def _detect_sep(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return ","


def _rename_columns(columns: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in columns:
        m = _X_RE.match(c)
        if m:
            out[c] = f"x_{m.group(1)}"
            continue
        m = _Y_RE.match(c)
        if m:
            out[c] = f"y_{m.group(1)}"
            continue
        m = _Z_RE.match(c)
        if m:
            out[c] = f"z_{m.group(1)}"
            continue
        m = _PROB_RE.match(c)
        if m:
            out[c] = f"prob_{m.group(1)}"
            continue
        m = _CONF_RE.match(c)
        if m:
            out[c] = f"conf_{m.group(1)}"
            continue
    return out


def translate_dir(in_root: Path, out_root: Path) -> list[Path]:
    in_root = Path(in_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    csvs = sorted(in_root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found under {in_root}")

    written: list[Path] = []
    for csv_path in csvs:
        rel = csv_path.relative_to(in_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sep = _detect_sep(csv_path)
        df = pd.read_csv(csv_path, sep=sep)

        # Rename keypoint columns to repo format
        rename_map = _rename_columns(list(df.columns))
        df = df.rename(columns=rename_map)

        # Add frame if time is not present
        if "time" not in df.columns and "frame" not in df.columns:
            df.insert(0, "frame", range(len(df)))

        df.to_csv(out_path, index=False)
        written.append(out_path)

    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Translate OpenMATB raw pose CSVs to repo ingest format."
    )
    p.add_argument(
        "--in",
        dest="in_root",
        required=True,
        type=Path,
        help="Root directory containing raw OpenMATB pose CSVs.",
    )
    p.add_argument(
        "--out",
        dest="out_root",
        required=True,
        type=Path,
        help="Output directory inside the repo (e.g., examples/case_study_1_matb/data/raw_pose_csv).",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    written = translate_dir(args.in_root, args.out_root)
    print(f"Wrote {len(written)} CSVs to {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
