"""Tests for the PoseSequence data model and provenance."""
from __future__ import annotations

import numpy as np
import pytest

from pose_dynamics.data import PoseSequence
from pose_dynamics.data.provenance import ProvenanceLog


def _seq(dims=2, n_frames=10, n_kp=4, conf=False, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(n_frames, n_kp, dims))
    confidence = rng.uniform(size=(n_frames, n_kp)) if conf else None
    return PoseSequence(
        coords=coords,
        keypoint_names=[f"kp{i}" for i in range(n_kp)],
        frame_rate=30.0,
        confidence=confidence,
    )


def test_shape_properties():
    s = _seq(dims=3, n_frames=15, n_kp=5)
    assert s.n_frames == 15
    assert s.n_keypoints == 5
    assert s.dims == 3
    assert s.duration_s == pytest.approx(0.5)


def test_mask_initialized_from_finiteness():
    coords = np.zeros((3, 2, 2))
    coords[1, 0, 0] = np.nan
    s = PoseSequence(coords=coords, keypoint_names=["a", "b"], frame_rate=30.0)
    assert s.mask[1, 0] == False  # noqa: E712
    assert s.mask[0, 0] == True   # noqa: E712
    assert s.missing_fraction() == pytest.approx(1 / 6)


def test_rejects_bad_coord_shape():
    with pytest.raises(ValueError, match="3D array"):
        PoseSequence(coords=np.zeros((3, 2)), keypoint_names=["a", "b"], frame_rate=30.0)


def test_rejects_bad_dims():
    with pytest.raises(ValueError, match="must be 2 .* or 3"):
        PoseSequence(coords=np.zeros((3, 2, 4)), keypoint_names=["a", "b"], frame_rate=30.0)


def test_rejects_name_count_mismatch():
    with pytest.raises(ValueError, match="must match"):
        PoseSequence(coords=np.zeros((3, 2, 2)), keypoint_names=["a"], frame_rate=30.0)


def test_rejects_nonpositive_frame_rate():
    with pytest.raises(ValueError, match="positive number of Hz"):
        PoseSequence(coords=np.zeros((3, 2, 2)), keypoint_names=["a", "b"], frame_rate=0)


def test_rejects_bad_confidence_shape():
    with pytest.raises(ValueError, match="confidence must have shape"):
        PoseSequence(
            coords=np.zeros((3, 2, 2)),
            keypoint_names=["a", "b"],
            frame_rate=30.0,
            confidence=np.zeros((3, 3)),
        )


def test_with_stage_is_copy_on_write():
    s = _seq()
    original = s.coords.copy()
    new_coords = s.coords + 1.0
    s2 = s.with_stage("shift", {"by": 1.0}, coords=new_coords)
    # Original is untouched
    np.testing.assert_allclose(s.coords, original)
    assert s.provenance.stages == []
    # New object has the change and the log entry
    np.testing.assert_allclose(s2.coords, original + 1.0)
    assert s2.provenance.stages == ["shift"]
    assert s2.provenance[0].params == {"by": 1.0}


def test_with_stage_carries_arrays_forward():
    s = _seq(conf=True)
    s2 = s.with_stage("noop")
    np.testing.assert_allclose(s2.confidence, s.confidence)
    np.testing.assert_array_equal(s2.mask, s.mask)
    # arrays are copies, not aliases
    assert s2.coords is not s.coords


def test_meta_merges():
    s = _seq()
    s.meta["participant"] = "p01"
    s2 = s.with_stage("x", meta={"condition": "high"})
    assert s2.meta == {"participant": "p01", "condition": "high"}
    assert "condition" not in s.meta


def test_summary_is_serializable():
    s = _seq(conf=True)
    s2 = s.with_stage("mask", {"threshold": 0.3})
    summ = s2.summary()
    assert summ["dims"] == 2
    assert summ["stages_applied"] == ["mask"]
    assert summ["has_confidence"] is True


def test_provenance_log_immutability():
    log = ProvenanceLog()
    log2 = log.appended("a", {"p": 1}, timestamp="2020-01-01T00:00:00+00:00")
    assert len(log) == 0
    assert log2.stages == ["a"]
    assert log2[0].timestamp == "2020-01-01T00:00:00+00:00"
