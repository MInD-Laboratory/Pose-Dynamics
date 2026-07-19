"""Tests for the canonical CSV loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pose_dynamics.data import PoseSequence, SchemaError, load_pose_csv
from tests.conftest import make_canonical_df


def test_loads_2d_with_confidence(body2d_csv):
    seq = load_pose_csv(body2d_csv, frame_rate=60.0)
    assert isinstance(seq, PoseSequence)
    assert seq.dims == 2
    assert seq.n_keypoints == 25
    assert seq.n_frames == 120
    assert seq.has_confidence
    assert seq.confidence.shape == (120, 25)
    assert seq.frame_rate == 60.0


def test_loads_3d_without_confidence(body3d_csv):
    seq = load_pose_csv(body3d_csv, frame_rate=30.0)
    assert seq.dims == 3
    assert seq.n_keypoints == 38
    assert seq.confidence is None


def test_load_records_provenance(body3d_csv):
    seq = load_pose_csv(body3d_csv, frame_rate=30.0)
    assert seq.provenance.stages == ["load"]
    entry = seq.provenance[0]
    assert entry.params["dims"] == 3
    assert entry.params["frame_rate"] == 30.0
    assert entry.params["has_confidence"] is False


def test_default_keypoint_names(body2d_csv):
    seq = load_pose_csv(body2d_csv, frame_rate=60.0)
    assert seq.keypoint_names[:3] == ["kp0", "kp1", "kp2"]


def test_custom_keypoint_names(body3d_csv):
    names = [f"joint{i}" for i in range(38)]
    seq = load_pose_csv(body3d_csv, frame_rate=30.0, keypoint_names=names)
    assert seq.keypoint_names == names


def test_wrong_length_names_errors(body3d_csv):
    with pytest.raises(SchemaError, match="line up one-to-one"):
        load_pose_csv(body3d_csv, frame_rate=30.0, keypoint_names=["a", "b"])


def test_coordinate_values_roundtrip(tmp_path: Path):
    df = make_canonical_df(10, 3, dims=3, with_confidence=True, seed=7)
    path = tmp_path / "rt.csv"
    df.to_csv(path, index=False)
    seq = load_pose_csv(path, frame_rate=30.0)
    # keypoint 1, z axis should equal the source column z1
    np.testing.assert_allclose(seq.coords[:, 1, 2], df["z1"].to_numpy())
    np.testing.assert_allclose(seq.confidence[:, 1], df["c1"].to_numpy())


def test_blank_cells_become_nan_and_mask(tmp_path: Path):
    df = make_canonical_df(5, 2, dims=2, with_confidence=False, seed=1)
    df.loc[2, "x1"] = np.nan  # a genuine missing value
    path = tmp_path / "gap.csv"
    df.to_csv(path, index=False)
    seq = load_pose_csv(path, frame_rate=60.0)
    assert np.isnan(seq.coords[2, 1, 0])
    assert seq.mask[2, 1] == False  # noqa: E712 -- masked out
    assert seq.mask[0, 1] == True   # noqa: E712 -- present


def test_missing_file_errors(tmp_path: Path):
    with pytest.raises(SchemaError, match="File not found"):
        load_pose_csv(tmp_path / "nope.csv", frame_rate=60.0)


def test_header_only_errors(tmp_path: Path):
    path = tmp_path / "headeronly.csv"
    path.write_text("x0,y0,x1,y1\n")
    with pytest.raises(SchemaError, match="no data rows"):
        load_pose_csv(path, frame_rate=60.0)


def test_non_numeric_value_errors(tmp_path: Path):
    path = tmp_path / "bad.csv"
    # Write raw text so a genuine non-numeric value lands in a coordinate column.
    path.write_text("x0,y0,x1,y1\n0.1,0.2,0.3,0.4\n0.5,oops,0.7,0.8\n")
    with pytest.raises(SchemaError, match="non-numeric"):
        load_pose_csv(path, frame_rate=60.0)


def test_bad_header_errors(tmp_path: Path):
    path = tmp_path / "badheader.csv"
    pd.DataFrame({"x0": [1.0], "y0": [2.0], "label": ["a"]}).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="not in the expected format"):
        load_pose_csv(path, frame_rate=60.0)
