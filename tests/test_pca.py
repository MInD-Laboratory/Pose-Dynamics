"""Tests for the core PCA diagnostic."""
import numpy as np
from pose_dynamics.diagnostics import fit_pca


def test_pca_recovers_dominant_direction():
    rng = np.random.default_rng(0)
    # data mostly along one axis
    t = rng.standard_normal(500)
    X = np.column_stack([3 * t, 0.1 * rng.standard_normal(500), 0.1 * rng.standard_normal(500)])
    scores, model = fit_pca(X, n_components=3)
    assert model.explained_variance_ratio_[0] > 0.9   # first PC dominates
    assert abs(abs(model.components_[0, 0]) - 1) < 0.05  # aligned with axis 0
    assert scores.shape == (500, 3)


def test_pca_variance_ratio_sums_to_one_and_orders():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 10))
    _, model = fit_pca(X)
    assert np.isclose(model.explained_variance_ratio_.sum(), 1.0)
    assert np.all(np.diff(model.explained_variance_ratio_) <= 1e-9)  # descending
    assert model.n_components_for(0.95) <= 10


def test_pca_transform_roundtrip_shapes():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 6))
    _, model = fit_pca(X, n_components=3)
    assert model.transform(X).shape == (100, 3)
    assert np.isclose(model.cumulative_variance()[-1], model.explained_variance_ratio_.sum())
