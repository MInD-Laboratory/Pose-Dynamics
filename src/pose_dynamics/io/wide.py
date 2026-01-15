from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical import validate_pose_df

_XYPROB_RE = re.compile(r"^(x|y|prob)(\d+)$")
_XYZ_RE = re.compile(r"^(x|y|z)(\d+)$")


def _time_from_timestamp_ns(ts: pd.Series) -> pd.Series:
    # convert to seconds relative to first sample
    t0 = ts.iloc[0]
    return (ts - t0) / 1e9


def _time_from_dt_ms(dt: pd.Series) -> pd.Series:
    # cumulative seconds; first sample at 0
    return pd.Series(np.r_[0.0, np.cumsum(dt.to_numpy()[:-1]) / 1000.0])


def load_pose_wide_csv(
    path: str | Path,
    *,
    fps: float | None = None,
    subject: str | None = None,
    trial: str | None = None,
) -> pd.DataFrame:
    """
    Load a wide pose CSV and return canonical long-form pose format.

    Supports:
      - 2D + confidence: xN,yN,probN
      - 3D: xN,yN,zN with optional timestamp_ns and/or dt_ms

    Canonical output columns:
      t (seconds), kp (int), x, y, optional z, conf
    """
    path = Path(path)
    wide = pd.read_csv(path)

    has_timestamp_ns = "timestamp_ns" in wide.columns
    has_dt_ms = "dt_ms" in wide.columns

    # Determine time
    if has_timestamp_ns:
        t = _time_from_timestamp_ns(wide["timestamp_ns"])
    elif has_dt_ms:
        t = _time_from_dt_ms(wide["dt_ms"])
    elif fps is not None:
        t = pd.Series(np.arange(len(wide), dtype=float) / float(fps))
    else:
        raise ValueError(
            "No time base found. Provide timestamp_ns or dt_ms column, or pass fps=..."
        )

    # Identify format by columns
    has_prob = any(re.match(r"^prob\d+$", c) for c in wide.columns)
    has_z = any(re.match(r"^z\d+$", c) for c in wide.columns)

    if has_prob:
        df = _from_xyprob(wide, t)
    elif has_z:
        df = _from_xyz(wide, t)
    else:
        raise ValueError(
            "Unrecognized wide pose format. Expected xN,yN,probN or xN,yN,zN columns."
        )

    if subject is not None:
        df["subject"] = subject
    if trial is not None:
        df["trial"] = trial

    validate_pose_df(df)
    return df


def _from_xyprob(wide: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
    # Collect per-kp columns
    kp_ids: set[int] = set()
    for c in wide.columns:
        m = _XYPROB_RE.match(c)
        if m:
            kp_ids.add(int(m.group(2)))
    if not kp_ids:
        raise ValueError("No xN/yN/probN columns found.")

    kp_sorted = sorted(kp_ids)

    # Build long-form
    out = []
    for kp in kp_sorted:
        xcol, ycol, pcol = f"x{kp}", f"y{kp}", f"prob{kp}"
        if (
            xcol not in wide.columns
            or ycol not in wide.columns
            or pcol not in wide.columns
        ):
            raise ValueError(
                f"Missing one of required columns for kp {kp}: {xcol},{ycol},{pcol}"
            )

        out.append(
            pd.DataFrame(
                {
                    "t": t.to_numpy(),
                    "kp": kp,
                    "x": wide[xcol].to_numpy(),
                    "y": wide[ycol].to_numpy(),
                    "z": np.nan,
                    "conf": wide[pcol].to_numpy(),
                }
            )
        )

    return pd.concat(out, ignore_index=True)


def _from_xyz(wide: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
    kp_ids: set[int] = set()
    for c in wide.columns:
        m = _XYZ_RE.match(c)
        if m:
            kp_ids.add(int(m.group(2)))
    if not kp_ids:
        raise ValueError("No xN/yN/zN columns found.")

    kp_sorted = sorted(kp_ids)

    out = []
    for kp in kp_sorted:
        xcol, ycol, zcol = f"x{kp}", f"y{kp}", f"z{kp}"
        if (
            xcol not in wide.columns
            or ycol not in wide.columns
            or zcol not in wide.columns
        ):
            raise ValueError(
                f"Missing one of required columns for kp {kp}: {xcol},{ycol},{zcol}"
            )

        out.append(
            pd.DataFrame(
                {
                    "t": t.to_numpy(),
                    "kp": kp,
                    "x": wide[xcol].to_numpy(),
                    "y": wide[ycol].to_numpy(),
                    "z": wide[zcol].to_numpy(),
                    "conf": np.nan,
                }
            )
        )

    return pd.concat(out, ignore_index=True)
