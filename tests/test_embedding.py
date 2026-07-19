"""Tests for AMI/FNN and the human-in-the-loop embedding-selection stage."""
from __future__ import annotations

import numpy as np
import pytest

from pose_dynamics.embedding import (
    EmbeddingEvidence,
    EmbeddingParams,
    Signal,
    ami_curve,
    cross_ami_curve,
    fnn_curve,
    select_embedding,
)
from pose_dynamics.embedding.selection import _suggest_tau


def _sine(n=1000, period=40, noise=0.0, seed=0, phase=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period + phase)
    if noise:
        x = x + rng.normal(scale=noise, size=n)
    return x


# --------------------------------------------------------------------------
# AMI
# --------------------------------------------------------------------------
def test_ami_curve_shape_and_positivity():
    c = ami_curve(_sine(period=40), min_lag=1, max_lag=140)
    assert c is not None
    assert c.lags[0] == 1 and c.n_lags == c.ami.size
    assert c.ami[0] > 0  # some shared information at small lag


def test_ami_first_minimum_near_quarter_period():
    # For a sine, AMI's first minimum sits near a quarter period (here 40/4 = 10).
    c = ami_curve(_sine(period=40, noise=0.02), min_lag=1, max_lag=60)
    info = _suggest_tau(c.lags, c.ami, rel_frac=1 / np.e)
    assert info["first_local_min"] is not None
    assert 6 <= info["first_local_min"] <= 14


def test_ami_none_for_constant_signal():
    assert ami_curve(np.ones(500), max_lag=100) is None


def test_ami_none_when_too_short():
    assert ami_curve(_sine(n=50), max_lag=100) is None


def test_cross_ami_runs():
    x = _sine(period=40, seed=1)
    y = _sine(period=40, seed=2, phase=0.5)
    c = cross_ami_curve(x, y, max_lag=60)
    assert c is not None and c.ami.size == c.lags.size


# --------------------------------------------------------------------------
# FNN
# --------------------------------------------------------------------------
def test_fnn_drops_for_low_dimensional_signal():
    c = fnn_curve(_sine(period=40, noise=0.01), tau=10, min_dim=1, max_dim=8)
    assert c is not None
    # A clean oscillator unfolds by ~2-3 dimensions: FNN should be small there.
    assert np.nanmin(c.pct_false) < 10.0
    # and it should be (weakly) non-increasing overall from dim 1 to the minimum
    assert c.pct_false[0] >= np.nanmin(c.pct_false)


def test_fnn_none_when_too_short():
    assert fnn_curve(_sine(n=50), tau=10, max_dim=8) is None


# --------------------------------------------------------------------------
# select_embedding: proposal, no auto-commit, human commit
# --------------------------------------------------------------------------
def _signal_set(n_signals=12, period=48):
    rng = np.random.default_rng(3)
    sigs = []
    for i in range(n_signals):
        p = period + rng.integers(-4, 5)  # slight variation -> spread
        kp = f"kp{i % 4}"
        sigs.append(
            Signal(f"s{i}", _sine(n=1200, period=p, noise=0.05, seed=i),
                    group={"keypoint": kp, "trial": f"trial{i % 3}"})
        )
    return sigs


def test_select_embedding_returns_evidence_within_grid():
    ev = select_embedding(_signal_set(), tau_grid=(10, 25), m_grid=(3, 6))
    assert isinstance(ev, EmbeddingEvidence)
    assert 10 <= ev.proposed_tau <= 25
    assert 3 <= ev.proposed_m <= 6
    # evidence carries per-signal suggestions for the variability diagnostic
    assert ev.per_signal_tau.shape[0] == ev.n_signals_used
    assert "proposal" in ev.justification.lower() or "propose" in ev.justification.lower()


def test_evidence_does_not_auto_commit():
    ev = select_embedding(_signal_set())
    # There is no EmbeddingParams until the human calls commit().
    assert not isinstance(ev, EmbeddingParams)
    params = ev.commit(tau=20, m=4, notes="looks stable across keypoints")
    assert isinstance(params, EmbeddingParams)
    assert (params.tau, params.m) == (20, 4)
    assert params.chosen_by == "human_confirmed"
    assert params.proposed_tau == ev.proposed_tau
    assert params.max_interp_gap == (4 - 1) * 20  # (m-1)tau cap
    assert params.theiler_window == 20


def test_commit_warns_outside_grid():
    ev = select_embedding(_signal_set(), tau_grid=(10, 25))
    with pytest.warns(UserWarning, match="outside the presented grid"):
        ev.commit(tau=40, m=4)


def test_commit_warns_below_proposed_m():
    ev = select_embedding(_signal_set(), m_grid=(3, 6))
    with pytest.warns(UserWarning, match="below the proposed"):
        ev.commit(tau=ev.proposed_tau, m=ev.proposed_m - 1)


def test_subset_sampling_is_logged():
    sigs = _signal_set(n_signals=20)
    ev = select_embedding(sigs, subset=8, seed=42)
    assert ev.n_signals_total == 20
    assert ev.n_signals_used == 8
    assert ev.subset_seed == 42


def test_multivariate_params_skip_embedding():
    p = EmbeddingParams(tau=0, m=0, multivariate=True, chosen_by="n/a")
    assert p.multivariate
    assert p.to_dict()["multivariate"] is True


def test_empty_signals_raises():
    with pytest.raises(ValueError, match="at least one signal"):
        select_embedding([])
