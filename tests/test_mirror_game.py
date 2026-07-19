"""Tests for Case-3 (Mirror Game) helpers: filename parsing, resampling, config."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pose_dynamics.case_studies.mirror_game import (
    config as C,
    load_and_resample,
    load_condition_map,
    parse_file,
)


def test_parse_filename():
    assert parse_file("P003_T7_P2_pose_3d.csv") == (3, 7, 2)
    assert parse_file("/x/P018_T12_P1_pose_3d.csv") == (18, 12, 1)


def test_condition_map(tmp_path):
    csv = tmp_path / "cond.csv"
    csv.write_text(
        "Pair,block1_lead,block1_1,block1_2,block1_3,block1_4,block1_5,block1_6,"
        "block2_1,block2_2,block2_3,block2_4,block2_5,block2_6\n"
        "1,P1,b2b,uni,f2f,uni,f2f,b2b,b2b,uni,f2f,uni,f2f,b2b\n"
    )
    cm = load_condition_map(csv)
    assert cm[(1, 1)] == "b2b"     # block1_1
    assert cm[(1, 3)] == "f2f"     # block1_3
    assert cm[(1, 7)] == "b2b"     # block2_1 (trial 7)
    assert cm[(1, 12)] == "b2b"    # block2_6 (trial 12)


def test_resampling_to_uniform_grid(tmp_path):
    # variable-rate source: a linear ramp in x0 over ~1 s
    n = 40
    ts = np.cumsum(np.random.default_rng(0).uniform(20, 80, size=n)) * 1e6  # ns
    data = {"timestamp_ns": ts.astype(np.int64), "dt_ms": np.gradient(ts) / 1e6}
    for k in range(38):
        for ax in "xyz":
            data[f"{ax}{k}"] = np.linspace(0, 1, n) if (k == 0 and ax == "x") else np.zeros(n)
    path = tmp_path / "P001_T1_P1_pose_3d.csv"
    pd.DataFrame(data).to_csv(path, index=False)

    seq = load_and_resample(path)
    assert seq.frame_rate == C.TARGET_RATE
    assert seq.dims == 3 and seq.n_keypoints == 38
    # x0 stays a monotincreasing ramp on the uniform grid
    x0 = seq.coords[:, 0, 0]
    assert np.all(np.diff(x0) >= -1e-9)
    assert seq.meta["pair"] == 1


def test_subset_indices_are_five():
    assert len(C.SUBSET_INDICES) == 5
    assert C.SUBSET["head"] == 5           # nose (numeric_inventory 9.9)
    assert C.SUBSET_INDICES == [5, 16, 17, 22, 23]


def test_cross_params_fixed_rec():
    from pose_dynamics.case_studies.mirror_game.reproduce import cross_params
    p = cross_params()
    assert p.radius_mode == "fixed_rrec"
    assert p.target_rec == 2.5
    assert p.min_line == 2
    assert (p.eDim, p.tLag) == (4, 20)
