from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from scipy.stats import kurtosis, skew


def summary_stats(x: np.ndarray, stats: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if x.size == 0:
        return out
    x = x[np.isfinite(x)]
    if x.size == 0:
        return out

    stats_set = set(stats)
    if "mean" in stats_set:
        out["mean"] = float(np.mean(x))
    if "std" in stats_set:
        out["std"] = float(np.std(x))
    if "min" in stats_set:
        out["min"] = float(np.min(x))
    if "max" in stats_set:
        out["max"] = float(np.max(x))
    if "median" in stats_set:
        out["median"] = float(np.median(x))
    if "iqr" in stats_set:
        out["iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
    if "rms" in stats_set:
        out["rms"] = float(np.sqrt(np.mean(x**2)))
    if "skew" in stats_set:
        out["skew"] = float(skew(x, bias=False)) if x.size > 2 else float("nan")
    if "kurtosis" in stats_set:
        out["kurtosis"] = float(kurtosis(x, bias=False)) if x.size > 3 else float("nan")

    return out


def derivative_series(x: np.ndarray, dt: float, order: int = 1) -> np.ndarray:
    if x.size == 0 or not np.isfinite(dt) or dt <= 0:
        return np.array([])
    if order == 1:
        return np.diff(x) / dt
    if order == 2:
        return np.diff(np.diff(x) / dt) / dt
    raise ValueError("order must be 1 or 2")
