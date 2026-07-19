"""Tests for the linear kinematic metrics."""
import numpy as np
from pose_dynamics.data import PoseSequence
from pose_dynamics.linear import per_frame_kinematics, summarise_signal, kinematic_summary


def _ramp_pose(fps=60.0):
    # keypoint 0 moves at constant 2 units/frame in x -> speed = 2*fps, accel ~ 0
    n = 100
    coords = np.zeros((n, 1, 2)); coords[:, 0, 0] = 2.0 * np.arange(n)
    return PoseSequence(coords=coords, keypoint_names=["k0"], frame_rate=fps)


def test_per_frame_kinematics_known_values():
    kin = per_frame_kinematics(_ramp_pose(fps=60.0))
    assert np.allclose(kin["displacement"], 2.0)      # 2 units/frame
    assert np.allclose(kin["speed"], 120.0)           # x fps
    assert np.allclose(kin["acceleration"], 0.0, atol=1e-9)


def test_summarise_signal_stats():
    s = summarise_signal([0.0, 2.0, 4.0], stats=("mean", "max", "rms"), prefix="speed")
    assert s["speed_mean"] == 2.0
    assert s["speed_max"] == 4.0
    assert s["speed_rms"] == np.sqrt((0 + 4 + 16) / 3)


def test_kinematic_summary_shape_and_dims_agnostic():
    df = kinematic_summary(_ramp_pose(), stats=("rms", "mean"))
    assert list(df["keypoint"]) == ["k0"]
    assert "speed_rms" in df.columns and "acceleration_mean" in df.columns
    # works for 3-D too
    coords = np.zeros((50, 2, 3)); coords[:, 0, 2] = np.arange(50)
    seq3d = PoseSequence(coords=coords, keypoint_names=["a", "b"], frame_rate=30.0)
    df3 = kinematic_summary(seq3d)
    assert len(df3) == 2
