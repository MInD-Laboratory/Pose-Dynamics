"""
Example converter: OpenPose face CSV -> canonical pose CSV.

Estimator-specific conversion is the user's responsibility (build plan §2, §8);
converters ship as copyable example scripts, never as core code. This one maps the
MATB OpenPose face export to the canonical schema:

    x{n}, y{n}, prob{n}   (1-based keypoint index, 'prob' = confidence)
    -> x{k}, y{k}, c{k}    (0-based keypoint index, 'c' = confidence)

Usage
-----
    python examples/openpose_to_canonical.py IN.csv OUT.csv
    # or import convert_openpose_face() and call it on a DataFrame
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_COL_RE = re.compile(r"^(x|y|prob)(\d+)$")


def convert_openpose_face(df: pd.DataFrame) -> pd.DataFrame:
    """Rename OpenPose columns to the canonical schema (prob->c, 1-based->0-based)."""
    def rename(col: str) -> str:
        m = _COL_RE.match(col)
        if not m:
            raise ValueError(f"Unexpected column {col!r}; expected x/y/prob + number.")
        axis, n = m.group(1), int(m.group(2))
        axis = "c" if axis == "prob" else axis
        return f"{axis}{n - 1}"  # 1-based -> 0-based

    return df.rename(columns={c: rename(c) for c in df.columns})


def convert_file(src: str | Path, dst: str | Path) -> Path:
    df = pd.read_csv(src)
    convert_openpose_face(df).to_csv(dst, index=False)
    return Path(dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    out = convert_file(sys.argv[1], sys.argv[2])
    print(f"wrote {out}")
