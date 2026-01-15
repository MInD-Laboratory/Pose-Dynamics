from pathlib import Path

import numpy as np

from pose_dynamics.io import load_pose_wide_csv


def test_load_xyprob():
    p = Path("tests/data/wide2d_prob.csv")
    df = load_pose_wide_csv(p, fps=10.0)
    assert set(["t", "kp", "x", "y", "z", "conf"]).issubset(df.columns)
    # 2 frames * 2 keypoints
    assert len(df) == 4
    assert sorted(df["kp"].unique().tolist()) == [0, 1]
    assert np.isfinite(df["conf"]).all()
    assert df["z"].isna().all()


def test_load_xyz_timestamp_wins():
    p = Path("tests/data/wide3d_time.csv")
    df = load_pose_wide_csv(p)
    assert set(["t", "kp", "x", "y", "z", "conf"]).issubset(df.columns)
    assert len(df) == 4
    assert sorted(df["kp"].unique().tolist()) == [0, 1]
    # timestamps: second row is +0.01s from first
    t_unique = sorted(df["t"].unique())
    assert np.isclose(t_unique[0], 0.0)
    assert np.isclose(t_unique[1], 0.01)
    assert np.isfinite(df["z"]).all()
    assert df["conf"].isna().all()
