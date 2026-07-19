"""
Average Mutual Information (AMI) and cross-AMI.

AMI estimates the embedding delay ``τ`` by quantifying the information shared
between ``x(t)`` and ``x(t+τ)`` (Fraser & Swinney, 1986). ``rqa-analysis`` does not
expose this, so the framework owns it (build plan §3).

The numeric definition is ported from the prototype
(``state_space_recon.ami``): the signal is min-max scaled to ``[0, 1]`` and, for
each lag, mutual information is computed over ``k`` equal-width bins with
``k = floor(1 + log2(N) + 0.5)`` (Sturges' rule on the lagged length ``N``). The
nested per-bin loop is replaced by a 2-D histogram, which is numerically
equivalent to the prototype's equal-width binning but far faster, so AMI can be
run across hundreds of signals as the manuscript recommends.

This module only *computes the curve*. It does not pick ``τ`` — that is a
human-confirmed decision (see :mod:`.selection`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AmiCurve:
    """An AMI curve: mutual information (bits) as a function of lag."""

    lags: np.ndarray   # (L,) int
    ami: np.ndarray    # (L,) float, bits

    @property
    def n_lags(self) -> int:
        return len(self.lags)


def _sturges_bins(n: int) -> int:
    return max(2, int(np.floor(1 + np.log2(n) + 0.5)))


def _minmax(x: np.ndarray) -> np.ndarray | None:
    lo, hi = np.min(x), np.max(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return (x - lo) / (hi - lo)


def _mutual_information(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """MI in bits between two [0,1]-scaled signals over ``k`` equal-width bins."""
    edges = np.linspace(0.0, 1.0, k + 1)
    joint, _, _ = np.histogram2d(a, b, bins=[edges, edges])
    n = joint.sum()
    if n == 0:
        return 0.0
    pxy = joint / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    ratio = pxy[nz] / (px @ py)[nz]
    return float(np.sum(pxy[nz] * np.log2(ratio)))


def ami_curve(x, min_lag: int = 1, max_lag: int = 140) -> AmiCurve | None:
    """Compute the AMI curve of a 1-D signal over a range of lags.

    Parameters
    ----------
    x : array-like
        1-D signal. Non-finite samples are dropped before computation.
    min_lag, max_lag : int
        Inclusive lag range (frames).

    Returns
    -------
    AmiCurve or None
        ``None`` if the signal is too short or constant to compute AMI.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    length = x.size
    if max_lag < 1 or length < 2 * max_lag:
        return None

    scaled = _minmax(x)
    if scaled is None:
        return None

    hi = max_lag if max_lag <= (length // 2 - 1) else max(1, length // 2)
    lags = np.arange(max(1, min_lag), hi + 1)

    values = np.zeros(lags.size, dtype=float)
    for i, lag in enumerate(lags):
        n = length - lag
        if n <= 2:
            continue
        k = _sturges_bins(n)
        values[i] = _mutual_information(scaled[:n], scaled[lag:], k)

    return AmiCurve(lags=lags, ami=values)


def cross_ami_curve(x, y, min_lag: int = 1, max_lag: int = 140) -> AmiCurve | None:
    """Cross-AMI between two 1-D signals (shared-delay selection for CRQA).

    Mirrors :func:`ami_curve` but pairs ``x(t)`` with ``y(t+lag)``. Only indices
    where both signals are finite are used. Used to choose a common ``τ`` for a
    cross-recurrence signal pair (e.g. Case 1 pupil × mid-face X).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    length = min(x.size, y.size)
    if length == 0 or max_lag < 1:
        return None

    xs, ys = _minmax(x), _minmax(y)
    if xs is None or ys is None:
        return None

    hi = max_lag if max_lag <= (length // 2 - 1) else max(1, length // 2)
    lags = np.arange(max(1, min_lag), hi + 1)

    values = np.zeros(lags.size, dtype=float)
    for i, lag in enumerate(lags):
        n = length - lag
        if n <= 2:
            continue
        k = _sturges_bins(n)
        values[i] = _mutual_information(xs[:n], ys[lag:], k)

    return AmiCurve(lags=lags, ami=values)
