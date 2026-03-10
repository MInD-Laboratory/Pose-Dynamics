"""
Dimensionality reduction for high-dimensional pose data.

Principal Component Analysis (PCA) is used as a pre-processing step before
recurrence quantification to compress the high-dimensional keypoint
representation (T × n_keypoints × 3 flattened) into a compact set of
orthogonal *principal movements* (PMs).  Each principal component (PC) captures
an independent, dominant mode of body motion — for example, arm extension,
torso rotation, or postural sway — allowing recurrence analysis to operate on
behaviorally interpretable, low-dimensional trajectories rather than raw
keypoint coordinates.

The lightweight ``PCAModel`` dataclass stores the fitted decomposition so the
same spatial projection can be applied to held-out windows or cross-condition
data without refitting, preserving comparability across the analysis.

Typical workflow
----------------
    scores, model = fit_pca(X_aligned, n_components=6)
    # scores : (T, 6) — time series of principal-movement scores
    # model.components_ : (6, D) — spatial loadings for each PM
    new_scores = model.transform(X_new)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PCAModel:
    """
    Lightweight container for a fitted PCA decomposition.

    Attributes
    ----------
    mean_ : np.ndarray, shape (D,)
        Per-feature mean subtracted before projection (the mean pose).
    components_ : np.ndarray, shape (K, D)
        Rows are the K principal directions (eigenvectors of the covariance
        matrix) ordered by descending explained variance.  Each row is a
        flattened spatial pattern describing one principal movement.
    explained_variance_ : np.ndarray, shape (K,)
        Variance explained by each component (eigenvalues of the covariance).
    explained_variance_ratio_ : np.ndarray, shape (K,)
        Fraction of total variance captured by each component, used to decide
        how many PCs to retain.
    """
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
