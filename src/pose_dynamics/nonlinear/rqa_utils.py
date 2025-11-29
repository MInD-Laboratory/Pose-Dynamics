# pose_dynamics/nonlinear/rqa.py
from __future__ import annotations
from typing import Dict, Literal, Tuple, Sequence

import numpy as np

from pose_dynamics.nonlinear import norm_utils
from pose_dynamics.rqa import rqa_utils_cpp  # <- rqa submodule backend

RqaMode = Literal["auto", "cross"]


# ---------------------------------------------------------------------
# Common param factory
# ---------------------------------------------------------------------
def make_rqa_params(
    eDim: int = 3,
    tLag: int = 10,
    radius: float = 0.1,
    norm: str = "euclidean",
    rescaleNorm: bool = False,
    tw: int = 1,
    minl: int = 2,
) -> Dict:
    return {
        "eDim": eDim,
        "tLag": tLag,
        "radius": radius,
        "norm": norm,
        "rescaleNorm": rescaleNorm,
        "tw": tw,
        "minl": minl,
    }


# ---------------------------------------------------------------------
# Fixed %REC
# ---------------------------------------------------------------------

def _radius_for_target_recurrence(
    D: np.ndarray,
    target_rec: float,
    rescale_norm: bool = False,
    theiler: int = 0,
) -> float:
    """
    Compute radius such that the recurrence rate (excluding theiler window)
    is approximately target_rec.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (N x N).
    target_rec : float
        Target recurrence rate. If >1, treated as percentage (e.g. 5 -> 0.05).
    rescale_norm : bool
        If True, match rqa_stats behavior by rescaling by max(D) before
        computing the radius.
    theiler : int
        Theiler window (number of diagonals to exclude around main diagonal).

    Returns
    -------
    radius : float
        Radius to pass into rqa_stats.
    """
    D = np.asarray(D, dtype=float)
    if target_rec > 1.0:  # allow 5 = 5%
        target_rec = target_rec / 100.0

    if target_rec <= 0.0 or target_rec >= 1.0:
        raise ValueError("target_rec must be in (0,1) or (0,100).")

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix.")

    n = D.shape[0]

    # build mask for off-diagonal elements outside Theiler window
    i, j = np.indices((n, n))
    mask = np.ones((n, n), dtype=bool)
    # exclude main diagonal and theiler band
    if theiler >= 0:
        mask &= np.abs(i - j) > theiler
    else:
        raise ValueError("theiler must be >= 0")

    vals = D[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No valid distances to compute radius from.")

    if rescale_norm:
        maxv = np.nanmax(vals)
        if maxv > 0:
            vals = vals / maxv
        else:
            # all zeros => any small positive radius yields 100% recurrence;
            # just return 0.
            return 0.0

    # quantile corresponding to target_rec
    q = np.quantile(vals, target_rec)
    return float(q)


# ---------------------------------------------------------------------
# Univariate / embedded RQA
# ---------------------------------------------------------------------
def run_rqa(
    x: np.ndarray,
    params: Dict,
    y: np.ndarray | None = None,
    mode: RqaMode = "auto",
    return_mats: bool = True,
    target_rec: float | None = None,
) -> Tuple[Dict, Dict, Dict | None, int]:
    """
    Wrapper around rqa_utils_cpp.rqa_dist + rqa_utils_cpp.rqa_stats.

    Either:
      - use params['radius'] (fixed radius), or
      - use target_rec (fixed recurrence rate) to choose radius, but not both.
    """
    x = np.asarray(x, dtype=float)
    x = norm_utils.normalize_data(x, params["norm"])

    if mode == "cross":
        if y is None:
            raise ValueError("y must be provided for cross-RQA")
        y = np.asarray(y, dtype=float)
        y = norm_utils.normalize_data(y, params["norm"])
    else:
        y = x

    # 1) distance / embedding
    ds = rqa_utils_cpp.rqa_dist(
        x, y,
        dim=params["eDim"],
        lag=params["tLag"],
    )
    D = ds["d"]

    # 2) choose radius
    radius_param = params.get("radius", None)
    if target_rec is not None and radius_param is not None:
        raise ValueError("Provide either 'radius' in params or 'target_rec', not both.")

    if target_rec is not None:
        rad = _radius_for_target_recurrence(
            D,
            target_rec=target_rec,
            rescale_norm=params.get("rescaleNorm", False),
            theiler=params.get("tw", 1),
        )
    elif radius_param is not None:
        rad = radius_param
    else:
        raise ValueError("Must provide either params['radius'] or target_rec.")

    # 3) stats
    td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(
        D,
        rescale=params["rescaleNorm"],
        rad=rad,
        diag_ignore=params["tw"],
        minl=params["minl"],
        rqa_mode=mode,
    )

    # you may want to expose the chosen radius to the caller:
    td["radius_used"] = rad

    if not return_mats:
        mats = None

    return td, rs, mats, err_code

def auto_rqa(x, params, return_mats=True, target_rec=None):
    return run_rqa(x, params, y=None, mode="auto",
                   return_mats=return_mats, target_rec=target_rec)

def cross_rqa(x, y, params, return_mats=True, target_rec=None):
    return run_rqa(x, params, y=y, mode="cross",
                   return_mats=return_mats, target_rec=target_rec)


# ---------------------------------------------------------------------
# Multivariate RQA 
# ---------------------------------------------------------------------

def _to_mv_array(data: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """
    Convert:
      - list of 1D arrays  -> (T, dims)
      - 2D array           -> (T, dims)
    and sanity-check time alignment.
    """
    if isinstance(data, np.ndarray):
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Multivariate data must have shape (T, dims)")
        if arr.shape[1] < 2:
            raise ValueError("Multivariate RQA requires at least 2 dimensions.")
        return arr

    # assume list/sequence of 1D arrays
    series = [np.asarray(d, dtype=float).flatten() for d in data]
    if len(series) < 2:
        raise ValueError("Multivariate RQA requires at least 2 dimensions.")

    lens = {len(s) for s in series}
    if len(lens) != 1:
        raise ValueError(f"All dimensions must have same length, got {lens}")

    return np.column_stack(series)


def run_mv_rqa(
    data: np.ndarray | Sequence[np.ndarray],
    params: Dict,
    data2: np.ndarray | Sequence[np.ndarray] | None = None,
    mode: RqaMode = "auto",
    return_mats: bool = True,
    target_rec: float | None = None,
) -> Tuple[Dict, Dict, Dict | None, int]:
    X1 = _to_mv_array(data)
    X1 = norm_utils.normalize_data(X1, params["norm"])

    if mode == "cross":
        if data2 is None:
            raise ValueError("data2 must be provided for multivariate cross-RQA")
        X2 = _to_mv_array(data2)
        X2 = norm_utils.normalize_data(X2, params["norm"])
    else:
        X2 = X1

    ds = rqa_utils_cpp.rqa_dist_multivariate(
        X1.astype(np.float32),
        X2.astype(np.float32),
    )
    D = ds["d"]

    radius_param = params.get("radius", None)
    if target_rec is not None and radius_param is not None:
        raise ValueError("Provide either 'radius' or 'target_rec', not both.")

    if target_rec is not None:
        # For MV cross, Theiler window across series = 0 (as in your original code)
        theiler = 0 if mode == "cross" else params.get("tw", 1)
        rad = _radius_for_target_recurrence(
            D,
            target_rec=target_rec,
            rescale_norm=params.get("rescaleNorm", False),
            theiler=theiler,
        )
    elif radius_param is not None:
        rad = radius_param
    else:
        raise ValueError("Must provide either params['radius'] or target_rec.")

    diag_ignore = 0 if mode == "cross" else params.get("tw", 1)

    td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(
        D,
        rescale=params["rescaleNorm"],
        rad=rad,
        diag_ignore=diag_ignore,
        minl=params["minl"],
        rqa_mode=mode,
    )

    td["radius_used"] = rad

    if not return_mats:
        mats = None

    return td, rs, mats, err_code


def mv_auto_rqa(
    data: np.ndarray | Sequence[np.ndarray],
    params: Dict,
    return_mats: bool = True,
):
    """Convenience alias: multivariate auto-RQA."""
    return run_mv_rqa(data, params, data2=None, mode="auto", return_mats=return_mats)


def mv_cross_rqa(
    data1: np.ndarray | Sequence[np.ndarray],
    data2: np.ndarray | Sequence[np.ndarray],
    params: Dict,
    return_mats: bool = True,
):
    """Convenience alias: multivariate cross-RQA."""
    return run_mv_rqa(data1, params, data2=data2, mode="cross", return_mats=return_mats)
