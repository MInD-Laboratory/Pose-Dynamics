"""
State-space reconstruction utilities for nonlinear time-series analysis.

Before recurrence quantification analysis (RQA) can be applied, the underlying
attractor must be reconstructed in a suitable phase space.  Following Takens'
embedding theorem, a scalar time series x(t) is unfolded into an m-dimensional
delay-coordinate vector:

    X(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(m−1)τ)]

where τ is the *time delay* (``tLag``) and m is the *embedding dimension*
(``eDim``).  This module provides data-driven methods for estimating these two
parameters from the data itself:

* **Auto Mutual Information (AMI)** selects τ as the lag at which the AMI
  function I(τ) reaches its first local minimum.  At this point successive
  delay coordinates carry maximally independent information, avoiding both
  redundancy (small τ, nearly identical values) and irrelevance (large τ,
  decorrelated chaos).

* **False Nearest Neighbours (FNN)** selects m by testing whether points that
  appear to be neighbours in dimension d are still neighbours when one more
  dimension is added.  The fraction of "false" neighbours falls to near zero
  at the true embedding dimension, indicating the attractor is fully unfolded.

* **Cross-AMI (X-AMI)** is the cross-signal analogue of AMI, used when a
  shared time delay must be chosen for two coupled time series (e.g., the two
  participants in a Cross-RQA dyadic analysis).

References
----------
Fraser & Swinney (1986). Independent coordinates for strange attractors from
    mutual information. *Physical Review A*, 33(2), 1134.
Kennel, Brown & Abarbanel (1992). Determining embedding dimension for
    phase-space reconstruction using a geometrical construction.
    *Physical Review A*, 45(6), 3403.
"""
from __future__ import annotations
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm



# ---------------- AMI  ----------------
def ami(timeseries, min_lag: int, max_lag: int):
    """
    Auto Mutual Information using histogram (equiprobable) binning per lag.
    Returns array shape (L, 2) with columns [lag, AMI].
    """
    if isinstance(timeseries, (pd.Series, pd.DataFrame)):
        x = timeseries.values.flatten()
    elif isinstance(timeseries, np.ndarray):
        x = timeseries.flatten()
    else:
        raise ValueError("timeseries must be a NumPy array or Pandas Series/DataFrame")

    x = x[np.isfinite(x)]
    length = len(x)
    if length < 2 * max_lag or max_lag < 1:
        return None  # too short for requested lags

    # lag vector
    if max_lag <= (length // 2 - 1):
        lag = np.arange(max(1, min_lag), max_lag + 1)
    else:
        lag = np.arange(max(1, min_lag), max(1, length // 2))

    # scale to [0,1] (guard against constant series)
    lo, hi = np.min(x), np.max(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.column_stack((lag, np.zeros_like(lag, dtype=float)))
    x = (x - lo) / (hi - lo)

    ami_values = np.zeros(len(lag), dtype=float)
    for i in tqdm(range(len(lag)), desc='Processing AMI', leave=False):
        N = length - lag[i]
        if N <= 2:
            ami_values[i] = 0.0
            continue
        k = int(np.floor(1 + np.log2(N) + 0.5))
        if k < 2 or np.var(x[:N], ddof=1) == 0:
            ami_values[i] = 0.0
            continue

        x1 = x[:N]
        x2 = x[lag[i]:]
        s = 0.0
        for k1 in range(1, k + 1):
            x1_lo, x1_hi = (k1 - 1) / k, k1 / k
            mask1 = (x1 > x1_lo) & (x1 <= x1_hi)
            if not mask1.any():
                continue
            px1 = mask1.sum() / N
            for k2 in range(1, k + 1):
                x2_lo, x2_hi = (k2 - 1) / k, k2 / k
                mask2 = (x2 > x2_lo) & (x2 <= x2_hi)
                if not mask2.any():
                    continue
                px2 = mask2.sum() / N
                pxy = (mask1 & mask2).sum() / N
                if pxy > 0:
                    s += pxy * np.log2(pxy / (px1 * px2))
        ami_values[i] = s

    return np.column_stack((lag, ami_values))


# ---------------- FNN  ----------------
def embed_time_series(data: np.ndarray, embedding_dim: int, lag: int) -> np.ndarray:
    n = len(data) - (embedding_dim - 1) * lag
    if n <= 0:
        return np.empty((0, embedding_dim))
    out = np.empty((n, embedding_dim), dtype=float)
    for c in range(embedding_dim):
        out[:, c] = data[c * lag : c * lag + n]
    return out

def fnn(timeseries, tlag: int, min_dimension: int, max_dimension: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dims, percent_false_neighbors) for a single 1D series."""
    if isinstance(timeseries, (pd.Series, pd.DataFrame)):
        x = timeseries.values.flatten().astype(float)
    elif isinstance(timeseries, np.ndarray):
        x = timeseries.flatten().astype(float)
    else:
        raise ValueError("timeseries must be a NumPy array or Pandas Series/DataFrame")

    x = x[np.isfinite(x)]
    if len(x) < (max_dimension + 1) * tlag + 1:
        # not enough samples, return empty
        return np.arange(min_dimension, max_dimension + 1), np.full((max_dimension - min_dimension + 1,), np.nan, dtype=float)

    dims = np.arange(min_dimension, max_dimension + 1, dtype=int)
    pct  = np.ones_like(dims, dtype=float)

    mean_x = np.mean(x)
    Ra = np.sqrt(np.mean((x - mean_x) ** 2))  # Abar in Kennel et al.

    for i, d in enumerate(tqdm(dims, desc="Processing FNN", leave=False)):
        max_l = len(x) - d * tlag
        emb = embed_time_series(x, d, tlag)
        if emb.shape[0] < 2:
            pct[i] = np.nan
            continue
        tree = KDTree(emb)
        number_false = 0
        for idx in range(max_l):
            # nearest neighbor excluding itself
            dist, nn_idx = tree.query(emb[idx], k=2)
            nearest_d = dist[1]
            nn = nn_idx[1]
            if nn >= max_l or nearest_d <= 0:
                # degenerate case
                test1 = 1.0
                test2 = 0.0
            else:
                test1 = abs(x[idx + d * tlag] - x[nn + d * tlag]) / nearest_d
                test2 = abs(x[idx + d * tlag] - x[nn + d * tlag]) / (Ra if Ra > 0 else 1e-12)
            number_false += (test1 >= 15) or (test2 >= 2)
        pct[i] = 100.0 * number_false / max_l
    return dims, pct

# ---------------- Cross-AMI  ----------------

def _safe_minmax_scale(a: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=float)
    lo, hi = np.nanmin(a[finite]), np.nanmax(a[finite])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a, dtype=float)
    out = (a - lo) / (hi - lo)
    out[~finite] = np.nan
    return out

def cross_ami(timeseries1, timeseries2, min_lag: int, max_lag: int) -> np.ndarray:
    """
    Cross Average Mutual Information (X-AMI) between timeseries1 and timeseries2.

    Returns an (L,2) array: column0 = lag, column1 = ami.
    """
    # convert inputs to 1D numpy arrays
    if isinstance(timeseries1, (pd.Series, pd.DataFrame)):
        timeseries1 = timeseries1.values.flatten()
    if isinstance(timeseries2, (pd.Series, pd.DataFrame)):
        timeseries2 = timeseries2.values.flatten()
    x = np.asarray(timeseries1).astype(float)
    y = np.asarray(timeseries2).astype(float)

    # keep only indices where both finite (aligned)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]; y = y[finite]
    length = min(len(x), len(y))
    if length == 0:
        return np.column_stack((np.arange(0), np.zeros(0)))

    # lag vector
    if max_lag <= (length // 2 - 1):
        lag = np.arange(max(1, min_lag), max_lag + 1)
    else:
        lag = np.arange(0, max(1, length // 2))

    # normalize
    x = _safe_minmax_scale(x)
    y = _safe_minmax_scale(y)

    ami_values = np.zeros(len(lag), dtype=float)

    for i in range(len(lag)):
        L = int(lag[i])
        N = length - L
        if N <= 2:
            ami_values[i] = 0.0
            continue

        k = int(np.floor(1 + np.log2(N) + 0.5))
        if k < 2 or np.nanvar(x[:N], ddof=1) == 0 or np.nanvar(y[L:], ddof=1) == 0:
            ami_values[i] = 0.0
            continue

        xw = x[:N]
        yw = y[L:]

        ami_sum = 0.0
        for k1 in range(1, k + 1):
            x_lo, x_hi = (k1 - 1) / k, k1 / k
            mask_x = (xw > x_lo) & (xw <= x_hi)
            if not mask_x.any():
                continue
            px1 = mask_x.sum() / N

            for k2 in range(1, k + 1):
                y_lo, y_hi = (k2 - 1) / k, k2 / k
                mask_y = (yw > y_lo) & (yw <= y_hi)
                if not mask_y.any():
                    continue
                py2 = mask_y.sum() / N

                ppp = (mask_x & mask_y).sum() / N
                if ppp > 0 and px1 > 0 and py2 > 0:
                    ami_sum += ppp * np.log2(ppp / (px1 * py2))

        ami_values[i] = ami_sum

    return np.column_stack((lag, ami_values))

# ---------------- Plots for Estimation  ----------------

def compute_ami_curve(
    x,
    min_lag: int = 1,
    max_lag: int = 140,
):
    """
    Compute AMI curve over a range of lags.

    Parameters
    ----------
    x : array-like or pandas Series/DataFrame
        1D time series.
    min_lag : int
        Minimum lag to evaluate (inclusive).
    max_lag : int
        Maximum lag to evaluate (inclusive).

    Returns
    -------
    result : dict
        {
          "lags": np.ndarray[int],   # shape (L,)
          "ami":  np.ndarray[float], # shape (L,)
          "raw":  np.ndarray[float], # original ami() output, shape (L,2)
        }
        or None if AMI could not be computed.
    """
    arr = ami(x, min_lag=min_lag, max_lag=max_lag)
    if arr is None or arr.size == 0:
        return None

    lags = arr[:, 0].astype(int)
    vals = arr[:, 1].astype(float)

    return {
        "lags": lags,
        "ami": vals,
        "raw": arr,
    }


def compute_fnn_curve(
    x,
    tau: int,
    min_dim: int = 1,
    max_dim: int = 10,
):
    """
    Compute FNN curve for a chosen delay tau.

    Parameters
    ----------
    x : array-like or pandas Series/DataFrame
        1D time series.
    tau : int
        Embedding delay to use in FNN calculation.
        (You choose this by inspecting the AMI plot.)
    min_dim : int
        Minimum embedding dimension to test (inclusive).
    max_dim : int
        Maximum embedding dimension to test (inclusive).

    Returns
    -------
    result : dict
        {
          "dims":     np.ndarray[int],   # embedding dimensions tested
          "pct_fnn":  np.ndarray[float], # % false neighbors per dim
        }
        or None if FNN could not be computed (e.g., series too short).
    """
    dims, pct = fnn(x, tlag=tau, min_dimension=min_dim, max_dimension=max_dim)
    # If everything is NaN, treat as failure
    if pct is None or np.all(~np.isfinite(pct)):
        return None

    return {
        "dims": dims.astype(int),
        "pct_fnn": pct.astype(float),
    }


def estimate_embedding_curves(
    x,
    min_lag: int = 1,
    max_lag: int = 140,
    tau_for_fnn: int | None = None,
    min_dim: int = 1,
    max_dim: int = 10,
):
    """
    High-level helper: compute AMI and (optionally) FNN curves
    without making any automatic decisions about tau or m.

    Parameters
    ----------
    x : array-like
        1D time series.
    min_lag, max_lag : int
        Range of lags for AMI.
    tau_for_fnn : int or None
        If provided, compute FNN curve at this tau.
        If None, FNN is skipped (you can call compute_fnn_curve later).
    min_dim, max_dim : int
        Dimension range for FNN if tau_for_fnn is given.

    Returns
    -------
    result : dict
        {
          "ami": dict | None,  # from compute_ami_curve()
          "fnn": dict | None,  # from compute_fnn_curve(), or None if tau_for_fnn is None
        }
    """
    ami_res = compute_ami_curve(x, min_lag=min_lag, max_lag=max_lag)

    fnn_res = None
    if tau_for_fnn is not None:
        fnn_res = compute_fnn_curve(
            x,
            tau=tau_for_fnn,
            min_dim=min_dim,
            max_dim=max_dim,
        )

    return {
        "ami": ami_res,
        "fnn": fnn_res,
    }
