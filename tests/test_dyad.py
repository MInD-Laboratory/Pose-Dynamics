"""Tests for the Dyad container and its shared-clock contract."""
from __future__ import annotations

import numpy as np
import pytest

from pose_dynamics.data import Dyad, PoseSequence, SharedClockError


def _seq(n_frames=10, fps=30.0, dims=2, n_kp=4, seed=0):
    rng = np.random.default_rng(seed)
    return PoseSequence(
        coords=rng.normal(size=(n_frames, n_kp, dims)),
        keypoint_names=[f"kp{i}" for i in range(n_kp)],
        frame_rate=fps,
    )


def test_valid_dyad_constructs():
    d = Dyad(a=_seq(), b=_seq(seed=1), dyad_id="pair01")
    assert d.frame_rate == 30.0
    assert d.dims == 2


def test_frame_rate_mismatch_raises():
    with pytest.raises(SharedClockError, match="different frame rates"):
        Dyad(a=_seq(fps=30.0), b=_seq(fps=60.0))


def test_length_mismatch_raises_by_default():
    with pytest.raises(SharedClockError, match="different lengths"):
        Dyad(a=_seq(n_frames=10), b=_seq(n_frames=12))


def test_length_mismatch_allowed_when_relaxed():
    d = Dyad(a=_seq(n_frames=10), b=_seq(n_frames=12), require_equal_length=False)
    # Constructs, but an explicit strict check still catches it before CRQA.
    with pytest.raises(SharedClockError, match="different lengths"):
        d.check_shared_clock(require_equal_length=True)


def test_map_applies_per_person_independently():
    d = Dyad(a=_seq(seed=0), b=_seq(seed=1))
    d2 = d.map(lambda s: s.with_stage("shift", {"by": 1}, coords=s.coords + 1))
    np.testing.assert_allclose(d2.a.coords, d.a.coords + 1)
    np.testing.assert_allclose(d2.b.coords, d.b.coords + 1)
    assert d2.a.provenance.stages == ["shift"]
    assert d2.b.provenance.stages == ["shift"]


def test_dims_mismatch_raises():
    with pytest.raises(SharedClockError, match="different dimensionality"):
        Dyad(a=_seq(dims=2), b=_seq(dims=3), require_equal_length=False).dims


def test_summary_covers_both_members():
    d = Dyad(a=_seq(), b=_seq(seed=1), dyad_id="p")
    summ = d.summary()
    assert summ["dyad_id"] == "p"
    assert "a" in summ and "b" in summ
    assert summ["equal_length"] is True
