"""Tests for the preprocessing stages: mask, interpolate, filter, gap policy."""
from __future__ import annotations

import numpy as np
import pytest

from pose_dynamics.data import PoseSequence
from pose_dynamics.preprocessing import (
    DataQualityReport,
    assess_quality,
    butterworth_filter,
    combine_reports,
    interpolate_gaps,
    mask_low_confidence,
    max_missing_run_per_keypoint,
)


def _seq_with_conf(conf_values, coords=None, fps=60.0):
    """Build a small 2-keypoint 2D sequence with a given confidence array."""
    conf = np.asarray(conf_values, dtype=float)
    T, K = conf.shape
    if coords is None:
        coords = np.ones((T, K, 2))
    return PoseSequence(
        coords=np.asarray(coords, float),
        keypoint_names=[f"kp{i}" for i in range(K)],
        frame_rate=fps,
        confidence=conf,
    )


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------
def test_masking_blanks_low_confidence():
    conf = np.array([[0.9, 0.1], [0.2, 0.8], [0.95, 0.95]])
    seq = _seq_with_conf(conf)
    out = mask_low_confidence(seq, threshold=0.30)
    # low-confidence coords -> NaN
    assert np.isnan(out.coords[0, 1, :]).all()
    assert np.isnan(out.coords[1, 0, :]).all()
    # high-confidence retained
    assert np.isfinite(out.coords[2, 0, :]).all()
    # mask reflects it
    assert out.mask[0, 1] == False  # noqa: E712
    assert out.mask[2, 0] == True   # noqa: E712
    assert out.provenance.stages == ["confidence_mask"]


def test_masking_noop_without_confidence():
    seq = PoseSequence(
        coords=np.ones((4, 2, 3)), keypoint_names=["a", "b"], frame_rate=30.0
    )
    out = mask_low_confidence(seq, threshold=0.30)
    assert out.provenance[0].params["applied"] is False
    np.testing.assert_array_equal(out.coords, seq.coords)


def test_masking_threshold_is_strict_less_than():
    conf = np.array([[0.30, 0.29]])
    seq = _seq_with_conf(conf)
    out = mask_low_confidence(seq, threshold=0.30)
    assert out.mask[0, 0] == True   # exactly at threshold kept  # noqa: E712
    assert out.mask[0, 1] == False  # below threshold masked      # noqa: E712


# --------------------------------------------------------------------------
# Interpolation (run-limited)
# --------------------------------------------------------------------------
def test_interpolation_fills_short_gap():
    x = np.array([0.0, np.nan, np.nan, 3.0])  # gap length 2
    coords = np.stack([x, x], axis=-1)[:, None, :]  # (4,1,2)
    seq = PoseSequence(coords=coords, keypoint_names=["a"], frame_rate=60.0)
    out = interpolate_gaps(seq, max_gap=3)
    np.testing.assert_allclose(out.coords[:, 0, 0], [0.0, 1.0, 2.0, 3.0])
    assert out.mask[:, 0].all()


def test_interpolation_leaves_long_gap():
    x = np.array([0.0, np.nan, np.nan, np.nan, 4.0])  # gap length 3
    coords = np.stack([x, x], axis=-1)[:, None, :]
    seq = PoseSequence(coords=coords, keypoint_names=["a"], frame_rate=60.0)
    out = interpolate_gaps(seq, max_gap=2)  # cap below gap length
    assert np.isnan(out.coords[1:4, 0, 0]).all()
    assert out.mask[2, 0] == False  # noqa: E712


def test_interpolation_leaves_leading_edge_gap():
    x = np.array([np.nan, np.nan, 2.0, 3.0])  # can't interpolate before first obs
    coords = np.stack([x, x], axis=-1)[:, None, :]
    seq = PoseSequence(coords=coords, keypoint_names=["a"], frame_rate=60.0)
    out = interpolate_gaps(seq, max_gap=10)
    assert np.isnan(out.coords[0:2, 0, 0]).all()


def test_interpolation_stage_name_configurable():
    x = np.array([0.0, np.nan, 2.0])
    coords = np.stack([x, x], axis=-1)[:, None, :]
    seq = PoseSequence(coords=coords, keypoint_names=["a"], frame_rate=60.0)
    out = interpolate_gaps(seq, max_gap=5, stage_name="principled_interpolation")
    assert out.provenance.stages == ["principled_interpolation"]


# --------------------------------------------------------------------------
# Filtering (temp-fill / reinstate)
# --------------------------------------------------------------------------
def test_filter_preserves_nan_positions():
    t = np.arange(300) / 60.0
    sig = np.sin(2 * np.pi * 1.0 * t) + 0.3 * np.sin(2 * np.pi * 20.0 * t)
    sig[100:105] = np.nan  # a gap that must survive filtering
    coords = np.stack([sig, sig], axis=-1)[:, None, :]  # (300,1,2)
    seq = PoseSequence(coords=coords, keypoint_names=["a"], frame_rate=60.0)
    out = butterworth_filter(seq, cutoff_hz=5.0, order=4)
    # gap reinstated at exactly the original positions
    assert np.isnan(out.coords[100:105, 0, 0]).all()
    assert np.isfinite(out.coords[0, 0, 0])


def test_filter_attenuates_high_frequency():
    t = np.arange(600) / 60.0
    low = np.sin(2 * np.pi * 1.0 * t)
    high = 0.5 * np.sin(2 * np.pi * 25.0 * t)
    sig = low + high
    coords = (low + high)[:, None, None] * np.ones((1, 1, 2))
    coords = np.stack([sig, sig], axis=-1)[:, None, :]
    seq = PoseSequence(coords=coords, keypoint_names=["a"], frame_rate=60.0)
    out = butterworth_filter(seq, cutoff_hz=5.0, order=4)
    # residual high-frequency energy should shrink markedly
    resid_before = np.std(sig - low)
    resid_after = np.std(out.coords[:, 0, 0] - low)
    assert resid_after < 0.25 * resid_before


def test_filter_rejects_cutoff_above_nyquist():
    seq = PoseSequence(coords=np.ones((50, 1, 2)), keypoint_names=["a"], frame_rate=30.0)
    with pytest.raises(ValueError, match="Nyquist"):
        butterworth_filter(seq, cutoff_hz=20.0, order=4)  # Nyquist = 15 Hz


# --------------------------------------------------------------------------
# Gap policy / data quality
# --------------------------------------------------------------------------
def test_max_missing_run():
    mask = np.array([[True], [False], [False], [False], [True]])
    seq = PoseSequence(
        coords=np.ones((5, 1, 2)), keypoint_names=["a"], frame_rate=60.0, mask=mask
    )
    assert max_missing_run_per_keypoint(seq)[0] == 3


def _seq_missing(frac_missing_per_kp, n_frames=100):
    """Build a sequence where each keypoint has a leading block missing."""
    K = len(frac_missing_per_kp)
    coords = np.ones((n_frames, K, 2))
    for k, frac in enumerate(frac_missing_per_kp):
        n_missing = int(round(frac * n_frames))
        coords[:n_missing, k, :] = np.nan
    return PoseSequence(
        coords=coords, keypoint_names=[f"kp{k}" for k in range(K)], frame_rate=60.0,
        source_file="trial.csv",
    )


def test_quality_ok_when_clean():
    seq = _seq_missing([0.0, 0.05])
    rep = assess_quality(seq, max_missing_frac=0.30)
    assert rep.status == "ok"
    assert not rep.excluded
    assert rep.flagged_keypoints == []


def test_quality_exclude_when_over_threshold():
    seq = _seq_missing([0.5, 0.5])  # 50% missing overall
    rep = assess_quality(seq, max_missing_frac=0.30, on_exceed="exclude")
    assert rep.status == "exclude"
    assert rep.excluded
    assert "exceeds trial threshold" in rep.reasons[0]


def test_quality_flags_bad_keypoint():
    seq = _seq_missing([0.0, 0.4])  # kp1 40% missing
    rep = assess_quality(
        seq, max_missing_frac=0.30, per_keypoint_max_missing_frac=0.30, on_exceed="flag"
    )
    # overall = 20% (< 30) so trial ok, but kp1 flagged
    assert rep.status == "ok"
    assert rep.flagged_keypoints == ["kp1"]


def test_combine_reports_returns_table():
    seqs = [_seq_missing([0.0, 0.0]), _seq_missing([0.5, 0.5])]
    reps = [assess_quality(s, max_missing_frac=0.30, on_exceed="exclude") for s in seqs]
    df = combine_reports(reps)
    assert list(df["status"]) == ["ok", "exclude"]
    assert df["excluded"].tolist() == [False, True]


# --------------------------------------------------------------------------
# End-to-end order and provenance
# --------------------------------------------------------------------------
def test_full_pipeline_order_and_provenance():
    rng = np.random.default_rng(0)
    T, K = 400, 3
    conf = rng.uniform(0.4, 1.0, size=(T, K))
    conf[50:60, 0] = 0.1     # short low-conf gap (fillable)
    conf[100:180, 1] = 0.1   # long low-conf gap (not fillable at cap 60)
    t = np.arange(T) / 60.0
    base = np.sin(2 * np.pi * 1.0 * t)[:, None, None] * np.ones((1, K, 2))
    seq = PoseSequence(
        coords=base + rng.normal(scale=0.01, size=(T, K, 2)),
        keypoint_names=[f"kp{k}" for k in range(K)],
        frame_rate=60.0,
        confidence=conf,
        source_file="e2e.csv",
    )

    masked = mask_low_confidence(seq, threshold=0.30)
    interp = interpolate_gaps(masked, max_gap=60)
    filt = butterworth_filter(interp, cutoff_hz=10.0, order=4)
    rep = assess_quality(interp, max_missing_frac=0.30, on_exceed="flag")

    assert filt.provenance.stages == [
        "confidence_mask",
        "provisional_interpolation",
        "butterworth_filter",
    ]
    # short gap on kp0 filled; long gap on kp1 remains
    assert np.isfinite(filt.coords[55, 0, 0])
    assert np.isnan(filt.coords[140, 1, 0])
    assert isinstance(rep, DataQualityReport)
    assert rep.status == "ok"  # overall missingness modest
