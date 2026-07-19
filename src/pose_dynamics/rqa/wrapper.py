"""
The RQA wrapper: routing, radius modes, and the radius search.

This is the single boundary to ``rqa-analysis``. It wraps only the core functions
(``autoRQA``, ``crossRQA``, ``multivariateCrossRQA``, and the DRP functions),
always calls them with ``plotMode='none'``, ``saveFig=False``, ``doStatsFile=False``
(the framework owns plotting and output), and routes by analysis type:

- single per-person 1-D signal      -> auto-RQA
- paired 1-D signals                -> cross-RQA
- paired multi-dimensional streams  -> multivariate cross-RQA (no delay embedding)

Normalization is a single decision: the input signals are passed **raw** and
``params.norm`` is handed to rqa-analysis, which normalizes exactly once. Do not
pre-normalize the signals yourself (that would be an implicit double
normalization); if a signal is already normalized, set ``norm="none"``.

Two radius modes (see :class:`RqaParams`):

- ``fixed_radius`` — the radius is supplied; %REC is an outcome.
- ``fixed_rrec`` — the target %REC is supplied and the radius is found by
  bisection (monotone: larger radius -> higher %REC). The achieved radius is the
  first-class output; non-convergence is reported, never hidden.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

try:
    from rqa_analysis import (
        DRP,
        autoRQA,
        crossDRP,
        crossRQA,
        multivariateCrossRQA,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "The 'rqa-analysis' package (with its compiled C++ core) is required. "
        "Install it with `pip install rqa-analysis` (a C/C++ compiler must be "
        "available to build its extension)."
    ) from exc

from .params import RqaParams
from .result import METRIC_KEYS, RqaResult


def _evaluate(fn: Callable, args: tuple, params: RqaParams, radius: float, tw: int):
    return fn(*args, params.lib_params(radius, tw))


def _solve_radius(fn: Callable, args: tuple, params: RqaParams, tw: int):
    """Bisection on radius to hit the target %REC. Returns (radius, %REC, converged, n_iter)."""
    target = params.target_rec
    tol = params.bisect_tol

    # Expand the upper bound until it reaches (or exceeds) the target %REC.
    hi = params.radius_hi
    _, rs, _, _ = _evaluate(fn, args, params, hi, tw)
    rec_hi = float(rs["perc_recur"])
    expansions = 0
    while rec_hi < target and expansions < 20:
        hi *= 2.0
        _, rs, _, _ = _evaluate(fn, args, params, hi, tw)
        rec_hi = float(rs["perc_recur"])
        expansions += 1

    if abs(rec_hi - target) <= tol or rec_hi < target:
        # Either already within tolerance, or the target is unreachable even at
        # the largest radius (report the best we have).
        converged = abs(rec_hi - target) <= tol
        return hi, rec_hi, converged, 0

    lo = 0.0
    best_r, best_rec = hi, rec_hi
    for n in range(1, params.bisect_max_iter + 1):
        mid = 0.5 * (lo + hi)
        _, rs, _, _ = _evaluate(fn, args, params, mid, tw)
        rec = float(rs["perc_recur"])
        best_r, best_rec = mid, rec
        if abs(rec - target) <= tol:
            return mid, rec, True, n
        if rec < target:
            lo = mid
        else:
            hi = mid
    return best_r, best_rec, False, params.bisect_max_iter


def _run(
    analysis: str,
    fn: Callable,
    args: tuple,
    params: RqaParams,
    label: str | None,
    meta: dict[str, Any] | None,
) -> RqaResult:
    tw = params.theiler_for(analysis)

    if params.radius_mode == "fixed_radius":
        radius_used, converged, n_iter = float(params.radius), True, 0
    else:
        radius_used, _, converged, n_iter = _solve_radius(fn, args, params, tw)
        if not converged:
            warnings.warn(
                f"radius search did not reach target %REC={params.target_rec} within "
                f"tolerance {params.bisect_tol}; using radius={radius_used:.4g}. "
                "The achieved %REC is reported in the result.",
                stacklevel=3,
            )

    td, rs, _mats, err = _evaluate(fn, args, params, radius_used, tw)
    metrics = {k: float(rs[k]) for k in METRIC_KEYS if k in rs}
    return RqaResult(
        analysis=analysis,
        metrics=metrics,
        radius_used=radius_used,
        rec_rate=float(rs["perc_recur"]),
        converged=converged,
        n_iter=n_iter,
        params=params.to_dict(),
        err_code=int(err),
        label=label,
        meta=dict(meta or {}),
        _matrix=td,
    )


# ----------------------------------------------------------------------
# Public routing entry points
# ----------------------------------------------------------------------
def run_auto_rqa(x, params: RqaParams, label: str | None = None, meta=None) -> RqaResult:
    """Auto-RQA of a single per-person 1-D signal."""
    x = np.asarray(x, dtype=float).ravel()
    return _run("auto", autoRQA, (x,), params, label, meta)


def run_cross_rqa(x, y, params: RqaParams, label: str | None = None, meta=None) -> RqaResult:
    """Cross-RQA of two 1-D signals (Theiler window forced to 0)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"cross-RQA signals must be equal length; got {x.size} and {y.size}.")
    return _run("cross", crossRQA, (x, y), params, label, meta)


def run_multivariate_cross_rqa(X, Y, params: RqaParams, label: str | None = None, meta=None) -> RqaResult:
    """Multivariate cross-RQA of two ``(T, d)`` streams (no delay embedding)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("multivariate cross-RQA needs 2-D (T, d) arrays.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"streams must share dimension count; got {X.shape[1]} and {Y.shape[1]}.")
    return _run("multivariate_cross", multivariateCrossRQA, (X, Y), params, label, meta)


def run_drp(x, params: RqaParams, label: str | None = None):
    """Diagonal Recurrence Profile (auto). Returns the raw rqa-analysis profile."""
    x = np.asarray(x, dtype=float).ravel()
    tw = params.theiler_for("auto")
    radius = params.radius if params.radius_mode == "fixed_radius" else params.radius_hi
    return DRP(x, params.lib_params(radius, tw))


def run_cross_drp(x, y, params: RqaParams, label: str | None = None):
    """Cross Diagonal Recurrence Profile. Returns the raw rqa-analysis profile."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return crossDRP(x, y, params.lib_params(
        params.radius if params.radius_mode == "fixed_radius" else params.radius_hi,
        params.theiler_for("cross"),
    ))
