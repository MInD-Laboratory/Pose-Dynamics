"""Tests for windowing, the two-eye pupil offset, and the Case-1 config."""
from __future__ import annotations

import numpy as np
import pytest

from pose_dynamics.data import PoseSequence
from pose_dynamics.features import FeaturePipeline, build_primitive
from pose_dynamics.features.types import PipelineContext
from pose_dynamics.windowing import make_windows


def _pose(coords, fps=60.0):
    coords = np.asarray(coords, float)
    return PoseSequence(coords=coords, keypoint_names=[f"k{i}" for i in range(coords.shape[1])],
                        frame_rate=fps)


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------
def test_windows_50pct_overlap():
    w = make_windows(n_frames=1000, frame_rate=10.0, window_s=20.0, overlap=0.5)
    # window = 200 frames, step = 100 -> starts at 0,100,...,800
    assert w[0].start == 0 and w[0].stop == 200
    assert w[1].start == 100
    assert all(win.length == 200 for win in w)
    assert [win.start for win in w] == list(range(0, 801, 100))


def test_windows_flag_missing():
    valid = np.ones(400, dtype=bool)
    valid[:150] = False  # first window mostly missing
    w = make_windows(400, 10.0, 20.0, 0.5, valid=valid, max_missing=0.5)
    assert w[0].missing_fraction == pytest.approx(0.75)
    assert w[0].flagged is True
    assert w[-1].flagged is False


def test_windows_reject_bad_overlap():
    with pytest.raises(ValueError, match="overlap"):
        make_windows(100, 10.0, 5.0, overlap=1.0)


# --------------------------------------------------------------------------
# Two-eye pupil offset (Case 1)
# --------------------------------------------------------------------------
def test_offset_feature_two_eye_average():
    # kp0 left pupil at (2,0); kp1 right pupil at (0,4)
    # left centre = mean(kp2,kp3) = (0,0); right centre = mean(kp4,kp5) = (0,0)
    coords = np.array([[[2.0, 0.0], [0.0, 4.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]])
    prim = build_primitive("offset_feature", {
        "name_out": "pupil", "point": [0, 1], "center": [[2, 3], [4, 5]]})
    out = prim.apply(PipelineContext(pose=_pose(coords)))
    # mean dx = (2+0)/2 = 1; mean dy = (0+4)/2 = 2
    assert out.features.get("pupil_dx")[0] == pytest.approx(1.0)
    assert out.features.get("pupil_dy")[0] == pytest.approx(2.0)
    # magnitude = mean of per-eye magnitudes = (2 + 4)/2 = 3
    assert out.features.get("pupil_mag")[0] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Case 1 config is a valid, runnable composition
# --------------------------------------------------------------------------
def test_case1_config_builds_and_runs():
    from pose_dynamics.case_studies.matb import config as C

    template = np.random.default_rng(0).normal(size=(70, 2)).tolist()
    pipe = FeaturePipeline.from_config(C.feature_pipeline_config(template))
    # dual-output procrustes present; emits geometry + parameters
    pose = _pose(np.random.default_rng(1).normal(size=(300, 70, 2)) * 100)
    ctx = pipe.run(pose)
    assert "head_motion_mag" in ctx.features.names        # parameter stream
    assert "pupil_metric_mag" in ctx.features.names        # feature on aligned geometry
    assert ctx.pose.provenance.stages[-1] == "procrustes"  # geometry stream updated
