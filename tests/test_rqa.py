"""Tests for the RQA wrapper: params, routing, radius modes, tidy output."""
from __future__ import annotations

import numpy as np
import pytest

from pose_dynamics.embedding import EmbeddingParams
from pose_dynamics.rqa import (
    METRIC_KEYS,
    RqaParams,
    RqaResult,
    run_auto_rqa,
    run_cross_rqa,
    run_multivariate_cross_rqa,
)


def _sine(n=800, period=40, noise=0.05, seed=0, phase=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period + phase) + noise * rng.normal(size=n)


# --------------------------------------------------------------------------
# Params validation
# --------------------------------------------------------------------------
def test_fixed_radius_requires_radius():
    with pytest.raises(ValueError, match="requires a radius"):
        RqaParams(eDim=4, tLag=20, radius_mode="fixed_radius")


def test_fixed_rrec_requires_target():
    with pytest.raises(ValueError, match="requires target_rec"):
        RqaParams(eDim=4, tLag=20, radius_mode="fixed_rrec")


def test_cannot_provide_both():
    with pytest.raises(ValueError, match="not both"):
        RqaParams(eDim=4, tLag=20, radius_mode="fixed_radius", radius=0.2, target_rec=2.5)


def test_rescale_code_mapping():
    assert RqaParams(eDim=4, tLag=20, radius=0.2, radius_mode="fixed_radius", rescale="mean").rescale_code == 1
    assert RqaParams(eDim=4, tLag=20, radius=0.2, radius_mode="fixed_radius", rescale="max").rescale_code == 2


def test_theiler_rule():
    p = RqaParams(eDim=4, tLag=20, radius=0.2, radius_mode="fixed_radius")
    assert p.theiler_for("auto") == 20          # defaults to tLag
    assert p.theiler_for("cross") == 0          # forced 0 across signals
    assert p.theiler_for("multivariate_cross") == 0


def test_from_embedding():
    emb = EmbeddingParams(tau=20, m=4)
    p = RqaParams.from_embedding(emb, radius_mode="fixed_radius", radius=0.2)
    assert (p.eDim, p.tLag) == (4, 20)


# --------------------------------------------------------------------------
# Auto-RQA (fixed radius)
# --------------------------------------------------------------------------
def test_auto_rqa_fixed_radius_runs():
    p = RqaParams(eDim=4, tLag=20, radius_mode="fixed_radius", radius=0.2, min_line=2)
    res = run_auto_rqa(_sine(), p, label="feat1")
    assert isinstance(res, RqaResult)
    assert res.analysis == "auto"
    assert res.err_code == 0
    assert res.radius_used == 0.2
    assert res.converged and res.n_iter == 0
    assert 0 <= res.rec_rate <= 100
    assert set(res.metrics).issuperset({"perc_recur", "perc_determ", "laminarity"})


# --------------------------------------------------------------------------
# Fixed-%REC bisection
# --------------------------------------------------------------------------
def test_fixed_rrec_hits_target():
    p = RqaParams(eDim=4, tLag=20, radius_mode="fixed_rrec", target_rec=5.0,
                  bisect_tol=0.1, min_line=2)
    res = run_auto_rqa(_sine(period=40), p)
    assert res.converged
    assert abs(res.rec_rate - 5.0) <= 0.1
    assert res.radius_used > 0
    assert res.n_iter >= 1  # bisection actually searched


def test_fixed_rrec_larger_target_needs_larger_radius():
    r = {}
    for target in (3.0, 8.0):
        p = RqaParams(eDim=4, tLag=20, radius_mode="fixed_rrec", target_rec=target, bisect_tol=0.1)
        r[target] = run_auto_rqa(_sine(period=40), p).radius_used
    assert r[8.0] > r[3.0]  # monotonic: higher %REC target -> larger radius


def test_non_convergence_is_reported_not_hidden():
    # An impossibly tight tolerance with a hard iteration cap cannot converge;
    # the wrapper must warn and flag it, never silently return a wrong radius.
    p = RqaParams(eDim=4, tLag=20, radius_mode="fixed_rrec", target_rec=5.0,
                  bisect_tol=1e-9, bisect_max_iter=3)
    with pytest.warns(UserWarning, match="did not reach target"):
        res = run_auto_rqa(_sine(period=40), p)
    assert not res.converged
    assert res.n_iter == 3  # exhausted the cap
    # The achieved %REC is still surfaced honestly.
    assert 0 <= res.rec_rate <= 100


# --------------------------------------------------------------------------
# Cross-RQA and multivariate cross-RQA routing
# --------------------------------------------------------------------------
def test_cross_rqa_runs():
    p = RqaParams(eDim=4, tLag=10, radius_mode="fixed_radius", radius=0.2, min_line=2)
    x = _sine(period=40, seed=1)
    y = _sine(period=40, seed=2, phase=0.3)
    res = run_cross_rqa(x, y, p, label="pair")
    assert res.analysis == "cross"
    assert res.err_code == 0


def test_cross_rqa_length_mismatch_errors():
    p = RqaParams(eDim=4, tLag=10, radius_mode="fixed_radius", radius=0.2)
    with pytest.raises(ValueError, match="equal length"):
        run_cross_rqa(_sine(n=500), _sine(n=400), p)


def test_multivariate_cross_rqa_runs():
    p = RqaParams(eDim=4, tLag=20, radius_mode="fixed_rrec", target_rec=2.5,
                  rescale="mean", multivariate=True, bisect_tol=0.1)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5))
    Y = X + 0.1 * rng.normal(size=(500, 5))
    res = run_multivariate_cross_rqa(X, Y, p, label="mirror")
    assert res.analysis == "multivariate_cross"
    assert res.err_code == 0
    assert abs(res.rec_rate - 2.5) <= 0.2


def test_multivariate_dim_mismatch_errors():
    p = RqaParams(eDim=4, tLag=20, radius_mode="fixed_radius", radius=0.3)
    with pytest.raises(ValueError, match="share dimension"):
        run_multivariate_cross_rqa(np.zeros((100, 3)), np.zeros((100, 5)), p)


# --------------------------------------------------------------------------
# Tidy output + provenance
# --------------------------------------------------------------------------
def test_to_row_has_metrics_radius_and_params():
    p = RqaParams(eDim=4, tLag=20, radius_mode="fixed_radius", radius=0.2)
    res = run_auto_rqa(_sine(), p, label="blink", meta={"trial": "t1", "person": "p1"})
    row = res.to_row()
    assert row["label"] == "blink"
    assert row["trial"] == "t1"
    assert "perc_determ" in row
    assert row["radius_used"] == 0.2
    assert row["param_tLag"] == 20          # parameters travel with the result
    assert row["param_radius_mode"] == "fixed_radius"
    for k in METRIC_KEYS:
        assert k in row


def test_single_normalization_is_logged_once():
    # The one normalization decision is params.norm, surfaced in the row.
    p = RqaParams(eDim=4, tLag=20, radius_mode="fixed_radius", radius=0.2, norm="zscore")
    row = run_auto_rqa(_sine(), p).to_row()
    assert row["param_norm"] == "zscore"
