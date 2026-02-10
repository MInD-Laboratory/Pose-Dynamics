from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _compute_dt(t: pd.Series) -> float:
    uniq = pd.Series(t.dropna().unique()).sort_values()
    if uniq.size <= 1:
        return float("nan")
    return float(uniq.diff().dropna().median())


def kinematics_features(
    df_win: pd.DataFrame, time_col: str, dims: List[str], metrics: List[str]
) -> dict:
    out: dict[str, float] = {}
    if df_win.empty:
        return out

    t = df_win[time_col]
    dt = _compute_dt(t)
    if not np.isfinite(dt) or dt <= 0:
        return out

    vals = df_win[dims].to_numpy(dtype=float)
    mask = np.isnan(vals).any(axis=1)
    vals = vals[~mask]
    if vals.shape[0] <= 2:
        return out

    diff = np.diff(vals, axis=0) / dt
    speed = np.linalg.norm(diff, axis=1)

    if "speed" in metrics:
        out["speed_mean"] = float(np.nanmean(speed))
        out["speed_std"] = float(np.nanstd(speed))
        out["speed_rms"] = float(np.sqrt(np.nanmean(speed**2)))

    if "accel" in metrics:
        accel = np.diff(diff, axis=0) / dt
        acc_mag = np.linalg.norm(accel, axis=1)
        out["accel_mean"] = float(np.nanmean(acc_mag))
        out["accel_std"] = float(np.nanstd(acc_mag))
        out["accel_rms"] = float(np.sqrt(np.nanmean(acc_mag**2)))

    return out
