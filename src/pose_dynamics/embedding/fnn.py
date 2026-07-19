"""
False Nearest Neighbours (FNN).

FNN estimates the embedding dimension ``m`` by testing whether points that are
neighbours in dimension ``d`` remain neighbours in ``d+1`` (Kennel, Brown &
Abarbanel, 1992). The fraction of "false" neighbours falls toward zero once the
attractor is adequately unfolded. ``rqa-analysis`` does not expose this, so the
framework owns it (build plan §3).

The numeric definition is ported verbatim from the prototype
(``state_space_recon.fnn``): forward delay embedding, nearest neighbour by
Euclidean distance (excluding self), and the two Kennel tests with the recovered
thresholds ``R_tol = 15`` and ``A_tol = 2`` (numeric inventory §4.4). This module
computes the curve only; it does not pick ``m``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

# Kennel et al. thresholds recovered from the prototype (numeric inventory §4.4).
_R_TOL = 15.0
_A_TOL = 2.0


@dataclass(frozen=True)
class FnnCurve:
    """An FNN curve: percentage of false neighbours per embedding dimension."""

    dims: np.ndarray      # (D,) int
    pct_false: np.ndarray  # (D,) float, percent
    tau: int

    @property
    def n_dims(self) -> int:
        return len(self.dims)


def _embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Forward delay embedding: rows are [x(t), x(t+tau), ..., x(t+(m-1)tau)]."""
    n = len(x) - (m - 1) * tau
    if n <= 0:
        return np.empty((0, m))
    out = np.empty((n, m), dtype=float)
    for c in range(m):
        out[:, c] = x[c * tau : c * tau + n]
    return out


def fnn_curve(
    x,
    tau: int,
    min_dim: int = 1,
    max_dim: int = 10,
) -> FnnCurve | None:
    """Compute the FNN curve of a 1-D signal at a given delay ``tau``.

    Parameters
    ----------
    x : array-like
        1-D signal. Non-finite samples are dropped first.
    tau : int
        Embedding delay to use (chosen from AMI evidence).
    min_dim, max_dim : int
        Inclusive embedding-dimension range to test.

    Returns
    -------
    FnnCurve or None
        ``None`` if the signal is too short for the requested dimensions.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < (max_dim + 1) * tau + 1:
        return None

    dims = np.arange(min_dim, max_dim + 1, dtype=int)
    pct = np.full(dims.size, np.nan, dtype=float)

    ra = np.sqrt(np.mean((x - x.mean()) ** 2))  # Kennel's "A" scale
    ra = ra if ra > 0 else 1e-12

    for i, d in enumerate(dims):
        max_l = len(x) - d * tau  # index range with a valid (d+1)-th coordinate
        emb = _embed(x, d, tau)
        if emb.shape[0] < 2 or max_l <= 0:
            continue
        tree = KDTree(emb)

        # Batch nearest-neighbour query over all points at once (k=2: self +
        # nearest). Restricting to the first max_l points keeps the (d+1)-th
        # coordinate in range.
        idx = np.arange(max_l)
        dist, nn_idx = tree.query(emb[idx], k=2)
        nearest_d = dist[:, 1]
        nn = nn_idx[:, 1]

        degenerate = (nn >= max_l) | (nearest_d <= 0)
        nn_safe = np.minimum(nn, max_l - 1)  # keep indexing in range; masked below
        added = np.abs(x[idx + d * tau] - x[nn_safe + d * tau])
        with np.errstate(divide="ignore", invalid="ignore"):
            test1 = np.where(degenerate, np.inf, added / nearest_d)
        test2 = added / ra
        false = degenerate | (test1 >= _R_TOL) | (test2 >= _A_TOL)
        pct[i] = 100.0 * np.count_nonzero(false) / max_l

    return FnnCurve(dims=dims, pct_false=pct, tau=int(tau))
