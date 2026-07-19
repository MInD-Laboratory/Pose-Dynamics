"""
Tidy RQA result: metrics + achieved radius + the full resolved parameter set.

The achieved radius is a first-class output column (build plan §5): under
``fixed_rrec`` mode %REC is pinned to the target and becomes a convergence check,
so the informative quantities are the radius itself plus the determinism and
line-length family. Every result carries the parameters that produced it, so any
row is traceable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The recurrence metrics surfaced from the rqa-analysis result dict (`rs`).
METRIC_KEYS = (
    "perc_recur",
    "perc_determ",
    "laminarity",
    "mean_line_length",
    "std_line_length",
    "maxl_found",
    "entropy",
    "trapping_time",
    "vmax",
    "divergence",
    "complexity",
    "trend_lower_diag",
    "trend_upper_diag",
)


@dataclass
class RqaResult:
    """One RQA computation's outcome, ready for a tidy results table.

    Attributes
    ----------
    analysis : {"auto", "cross", "multivariate_cross"}
        Which recurrence routine produced this.
    metrics : dict
        Recurrence metrics (keys in :data:`METRIC_KEYS`).
    radius_used : float
        The radius actually applied (solved under ``fixed_rrec``; supplied under
        ``fixed_radius``).
    rec_rate : float
        Achieved recurrence rate (%REC).
    converged : bool
        For ``fixed_rrec``: whether the radius search hit the target within
        tolerance. Always ``True`` for ``fixed_radius``.
    n_iter : int
        Radius-search iterations used (0 for ``fixed_radius``).
    params : dict
        The resolved :class:`RqaParams` that produced this result.
    err_code : int
        rqa-analysis error code (0 = success).
    label : str or None
        Free-text label (e.g. feature / keypoint / window).
    meta : dict
        Extra tidy-table columns (trial, person, condition, window index, ...).
    """

    analysis: str
    metrics: dict[str, float]
    radius_used: float
    rec_rate: float
    converged: bool
    n_iter: int
    params: dict[str, Any]
    err_code: int = 0
    label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    _matrix: Any = field(default=None, repr=False)

    def to_row(self) -> dict[str, Any]:
        """Flat one-row dict: metadata + metrics + achieved radius + parameters."""
        row: dict[str, Any] = {"analysis": self.analysis, "label": self.label}
        row.update(self.meta)
        row.update({k: self.metrics.get(k) for k in METRIC_KEYS})
        row["radius_used"] = self.radius_used
        row["rec_rate"] = self.rec_rate
        row["converged"] = self.converged
        row["n_iter"] = self.n_iter
        row["err_code"] = self.err_code
        # Parameters, prefixed so they never collide with metric columns.
        row.update({f"param_{k}": v for k, v in self.params.items()})
        return row

    def plot(self, ax=None):
        """Checkpoint visual: the recurrence plot (framework-owned rendering)."""
        import numpy as np
        import matplotlib.pyplot as plt  # lazy import

        if self._matrix is None:
            raise ValueError("No recurrence matrix was retained for this result.")
        rp = np.asarray(self._matrix)
        if rp.ndim == 1:  # square it if the backend returned a flat buffer
            n = int(round(rp.size ** 0.5))
            rp = rp.reshape(n, n)
        if ax is None:
            _, ax = plt.subplots(figsize=(5.2, 5))
        ax.imshow(rp, origin="lower", cmap="binary", interpolation="nearest")
        ax.set_xlabel("time index i")
        ax.set_ylabel("time index j")
        title = self.label or self.analysis
        ax.set_title(f"{title}  (%REC={self.rec_rate:.2f}, r={self.radius_used:.3g})")
        return ax

    def __repr__(self) -> str:
        conv = "" if self.converged else " NOT-CONVERGED"
        return (
            f"RqaResult({self.analysis}, %REC={self.rec_rate:.2f}, "
            f"%DET={self.metrics.get('perc_determ', float('nan')):.2f}, "
            f"r={self.radius_used:.3g}{conv})"
        )
