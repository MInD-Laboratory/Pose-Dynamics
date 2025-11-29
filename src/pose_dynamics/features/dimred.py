# features/dimred.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PCAModel:
    mean_: np.ndarray           # (D,)
    components_: np.ndarray     # (K, D)
    explained_variance_: np.ndarray      # (K,)
    explained_variance_ratio_: np.ndarray  # (K,)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project new data using fitted PCA model.

        X: (T, D) -> returns (T, K)
        """
        Xc = X - self.mean_
        return Xc @ self.components_.T


def fit_pca(
    X: np.ndarray,
    n_components: Optional[int] = None,
    center: bool = True,
) -> Tuple[np.ndarray, PCAModel]:
    """
    Fit PCA on (T, D) data and return (scores, model).

    Parameters
    ----------
    X : array, shape (T, D)
        Observations x features (e.g. time x keypoints/axes).
    n_components : int or None
        Number of PCs to keep. If None, keep all (min(T, D)).
    center : bool
        Whether to subtract the mean per feature.

    Returns
    -------
    scores : np.ndarray, shape (T, K)
        Time series in PC space.
    model : PCAModel
        Object with mean_, components_, explained_variance_, explained_variance_ratio_.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (T, D).")

    T, D = X.shape

    if center:
        mean = X.mean(axis=0)
        Xc = X - mean
    else:
        mean = np.zeros(D, dtype=float)
        Xc = X

    # SVD: Xc = U S V^T  -> rows of V are principal directions
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # eigenvalues of covariance
    eigvals = (S**2) / (T - 1)

    max_k = min(T, D)
    if n_components is None:
        k = max_k
    else:
        k = int(min(max(1, n_components), max_k))

    components = Vt[:k, :]          # (K, D)
    scores = Xc @ components.T      # (T, K)
    explained_variance = eigvals[:k]
    explained_variance_ratio = explained_variance / eigvals.sum()

    model = PCAModel(
        mean_=mean,
        components_=components,
        explained_variance_=explained_variance,
        explained_variance_ratio_=explained_variance_ratio,
    )
    return scores, model
