"""
Translate Mosaic raw pose CSVs into repo ingest format.

Input file example:
  S003_T1_left.csv

Columns look like:
  Nose_confidence, Nose_x_offset, Nose_y_offset, ...

Output format uses:
  conf_<kp>, x_<kp>, y_<kp>

Adds identifier columns parsed from filename:
  subject_id, trial, side

Adds a `frame` column if no `time` column is present.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

_X_RE = re.compile(r"^(.+)_x_offset$", re.IGNORECASE)
_Y_RE = re.compile(r"^(.+)_y_offset$", re.IGNORECASE)
_CONF_RE = re.compile(r"^(.+)_confidence$", re.IGNORECASE)


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
            kp = m.group(1)
            out[c] = f"x_{kp}"
            continue
        m = _Y_RE.match(c)
        if m:
            kp = m.group(1)
            out[c] = f"y_{kp}"
            continue
        m = _CONF_RE.match(c)
        if m:
            kp = m.group(1)
            out[c] = f"conf_{kp}"
            continue
    return out


def _parse_identifiers(stem: str) -> tuple[str | None, str | None, str | None]:
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return None, None, None


def _iter_csvs(in_root: Path) -> tuple[Path, list[Path]]:
    in_root = Path(in_root)
    if in_root.is_file() and in_root.suffix.lower() == ".csv":
        return in_root.parent, [in_root]
    csvs = sorted(in_root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found under {in_root}")
    return in_root, csvs


def translate_dir(in_root: Path, out_root: Path) -> list[Path]:
    base_root, csvs = _iter_csvs(in_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for csv_path in csvs:
        rel = csv_path.relative_to(base_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sep = _detect_sep(csv_path)
        df = pd.read_csv(csv_path, sep=sep)

        rename_map = _rename_columns(list(df.columns))
        df = df.rename(columns=rename_map)

        subject_id, trial, side = _parse_identifiers(csv_path.stem)
        if subject_id is not None and "subject_id" not in df.columns:
            df.insert(0, "subject_id", subject_id)
        if trial is not None and "trial" not in df.columns:
            insert_at = 1 if "subject_id" in df.columns else 0
            df.insert(insert_at, "trial", trial)
        if side is not None and "side" not in df.columns:
            insert_at = 2 if "subject_id" in df.columns and "trial" in df.columns else 0
            df.insert(insert_at, "side", side)

        if "time" not in df.columns and "frame" not in df.columns:
            df.insert(0, "frame", range(len(df)))

        df.to_csv(out_path, index=False)
        written.append(out_path)

    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Translate Mosaic raw pose CSVs to repo ingest format."
    )
    p.add_argument(
        "--in",
        dest="in_root",
        required=True,
        type=Path,
        help="Input CSV file or directory containing Mosaic raw pose CSVs.",
    )
    p.add_argument(
        "--out",
        dest="out_root",
        required=True,
        type=Path,
        help="Output directory inside the repo (e.g., examples/case_study_2_mosaic/data/raw_pose_csv).",
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
