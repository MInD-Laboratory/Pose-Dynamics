"""
Principal Component Analysis as a diagnostic decomposition.

PCA characterizes dominant modes of coordinated movement across landmarks —
"Principal Movements" (PMs) — and doubles as a preprocessing-quality diagnostic
(build plan §5, manuscript §PCA). It is a decomposition, not inferential
statistics, so it lives in the core alongside AMI/FNN.

Ported from the prototype ``features/dimred.fit_pca``: an SVD of the mean-centred
data; the rows of ``Vᵀ`` are the principal directions (each a flattened spatial
pattern), ordered by explained variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAModel:
    """A fitted PCA decomposition.

    Attributes
    ----------
    mean_ : np.ndarray, shape (D,)
        Per-feature mean (the mean pose) subtracted before projection.
    components_ : np.ndarray, shape (K, D)
        Principal directions (unit eigenvectors), ordered by descending variance.
    explained_variance_ : np.ndarray, shape (K,)
        Variance of each component.
    explained_variance_ratio_ : np.ndarray, shape (K,)
        Fraction of total variance per component.
    """

    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project ``(T, D)`` data into ``(T, K)`` PC-score space."""
        return (np.asarray(X, float) - self.mean_) @ self.components_.T

    def n_components_for(self, variance: float) -> int:
        """Smallest number of components whose cumulative ratio reaches ``variance``."""
        cum = np.cumsum(self.explained_variance_ratio_)
        return int(np.searchsorted(cum, variance) + 1)

    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio_)


def fit_pca(X: np.ndarray, n_components: int | None = None, center: bool = True):
    """Fit PCA on ``(T, D)`` data. Returns ``(scores, model)``.

    Parameters
    ----------
    X : np.ndarray
        Observations x features (e.g. frames x flattened keypoint coords).
    n_components : int, optional
        Number of components to keep; ``None`` keeps ``min(T, D)``.
    center : bool
        Subtract the per-feature mean first (default True).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (T, D).")
    T, D = X.shape

    mean = X.mean(axis=0) if center else np.zeros(D)
    Xc = X - mean

    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S ** 2) / (T - 1)

    max_k = min(T, D)
    k = max_k if n_components is None else int(min(max(1, n_components), max_k))

    components = Vt[:k, :]
    model = PCAModel(
        mean_=mean,
        components_=components,
        explained_variance_=eigvals[:k],
        explained_variance_ratio_=eigvals[:k] / eigvals.sum(),
    )
    return Xc @ components.T, model
