"""
Translate Mirror Game raw pose CSVs into repo ingest format.

Input file example:
  P004_T2_P2_pose_3d.csv

Columns look like:
  timestamp_ns, dt_ms, x0, y0, z0, x1, y1, z1, ...

Output format uses:
  time, x_<kp>, y_<kp>, z_<kp>

Adds identifier columns parsed from filename:
  subject_id, trial, party
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

_XYZ_RE = re.compile(r"^([xyz])(\d+)$", re.IGNORECASE)
_FILENAME_RE = re.compile(r"^(P\d{3})_(T\d+)_((?:P1|P2))_pose_3d$", re.IGNORECASE)


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
        m = _XYZ_RE.match(c)
        if m:
            axis, idx = m.group(1).lower(), m.group(2)
            out[c] = f"{axis}_{idx}"
    return out


def _parse_identifiers(stem: str) -> tuple[str | None, str | None, str | None]:
    m = _FILENAME_RE.match(stem)
    if not m:
        return None, None, None
    return m.group(1).upper(), m.group(2).upper(), m.group(3).upper()


def _iter_csvs(in_root: Path) -> tuple[Path, list[Path]]:
    in_root = Path(in_root)
    if in_root.is_file() and in_root.suffix.lower() == ".csv":
        return in_root.parent, [in_root]
    csvs = sorted(in_root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found under {in_root}")
    return in_root, csvs


def _resample_to_rate(
    df: pd.DataFrame,
    *,
    target_rate: float,
    dt_col: str = "dt_ms",
    timestamp_col: str = "timestamp_ns",
) -> pd.DataFrame:
    df = df.copy()

    if dt_col in df.columns:
        df[dt_col] = pd.to_numeric(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col])
        if df.empty:
            raise ValueError("Invalid time axis: dt_ms column has no valid values.")
        df["time_s"] = df[dt_col].cumsum() / 1000.0
        df.loc[df.index[0], "time_s"] = 0.0
    elif timestamp_col in df.columns:
        df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col])
        if df.empty:
            raise ValueError(
                "Invalid time axis: timestamp_ns column has no valid values."
            )
        t0 = df[timestamp_col].iloc[0]
        df["time_s"] = (df[timestamp_col] - t0) / 1e9
    else:
        df["time_s"] = np.arange(len(df)) / target_rate

    df = df.set_index("time_s")

    start, end = df.index.min(), df.index.max()
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("Invalid time axis: could not determine resampling range.")

    new_index = np.arange(start, end, 1 / target_rate)
    df = df.reindex(df.index.union(new_index)).interpolate("linear").loc[new_index]

    df = df.reset_index().rename(columns={"index": "time"})
    return df


def translate_dir(in_root: Path, out_root: Path, *, target_rate: float) -> list[Path]:
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

        df = _resample_to_rate(df, target_rate=target_rate)

        rename_map = _rename_columns(list(df.columns))
        df = df.rename(columns=rename_map)

        subject_id, trial, party = _parse_identifiers(csv_path.stem)
        if subject_id is not None and "subject_id" not in df.columns:
            df.insert(0, "subject_id", subject_id)
        if trial is not None and "trial" not in df.columns:
            insert_at = 1 if "subject_id" in df.columns else 0
            df.insert(insert_at, "trial", trial)
        if party is not None and "party" not in df.columns:
            insert_at = 2 if "subject_id" in df.columns and "trial" in df.columns else 0
            df.insert(insert_at, "party", party)

        df.to_csv(out_path, index=False)
        written.append(out_path)

    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Translate Mirror Game raw pose CSVs to repo ingest format."
    )
    p.add_argument(
        "--in",
        dest="in_root",
        required=True,
        type=Path,
        help="Input CSV file or directory containing Mirror Game raw pose CSVs.",
    )
    p.add_argument(
        "--out",
        dest="out_root",
        required=True,
        type=Path,
        help="Output directory inside the repo (e.g., examples/case_study_3_mirror/data/raw_pose).",
    )
    p.add_argument(
        "--target-rate",
        dest="target_rate",
        type=float,
        default=30.0,
        help="Target sampling rate in Hz (default: 30.0).",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    written = translate_dir(args.in_root, args.out_root, target_rate=args.target_rate)
    print(f"Wrote {len(written)} CSVs to {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
