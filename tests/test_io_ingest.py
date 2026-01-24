import json
from pathlib import Path

import pandas as pd
import pytest

from pose_dynamics.io.csv_pose import ingest_pose_csv_dir, wide_pose_csv_to_long


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_single_csv_time_mode_basic(tmp_path: Path) -> None:
    # Minimal valid time-based CSV (2 keypoints, with prob on one)
    df = pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.2],
            "x_nose": [1, 2, 3],
            "y_nose": [4, 5, 6],
            "prob_nose": [0.9, 0.8, 0.7],
            "x_knee": [10, 11, 12],
            "y_knee": [20, 21, 22],
            "extra_metadata": ["a", "b", "c"],  # should be ignored
        }
    )
    csv_path = tmp_path / "in" / "trial_A.csv"
    _write_csv(csv_path, df)

    res = wide_pose_csv_to_long(csv_path, fps=None)
    out = res.df_long

    # Canonical columns: time, keypoint, x, y, and conf exists because prob_* existed for at least one kp
    assert "time" in out.columns
    assert "frame" not in out.columns
    assert set(["trial_id", "source_file", "keypoint", "x", "y"]).issubset(out.columns)
    assert "conf" in out.columns

    # trial_id and source_file derived from filename
    assert out["trial_id"].nunique() == 1
    assert out["trial_id"].iloc[0] == "trial_A"
    assert out["source_file"].iloc[0] == "trial_A.csv"

    # keypoints detected
    assert set(out["keypoint"].unique()) == {"nose", "knee"}

    # ignored columns logged
    assert "extra_metadata" in res.trial_meta["ignored_columns"]


def test_single_csv_frame_mode_requires_fps(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "x_1": [1, 2, 3],
            "y_1": [4, 5, 6],
        }
    )
    csv_path = tmp_path / "trial_B.csv"
    _write_csv(csv_path, df)

    # No time + frame => must provide fps
    with pytest.raises(
        ValueError, match=r"must provide --fps|provide --fps|You must provide --fps"
    ):
        _ = wide_pose_csv_to_long(csv_path, fps=None)

    # With fps it should work
    res = wide_pose_csv_to_long(csv_path, fps=30.0)
    out = res.df_long
    assert "frame" in out.columns
    assert "time" not in out.columns
    assert res.qc["fps_used"] == 30.0


def test_malformed_keypoint_missing_partner_errors(tmp_path: Path) -> None:
    # x_knee exists but y_knee missing -> hard error
    df = pd.DataFrame(
        {
            "time": [0.0, 0.1],
            "x_knee": [1, 2],
            "x_nose": [3, 4],
            "y_nose": [5, 6],
        }
    )
    csv_path = tmp_path / "trial_bad.csv"
    _write_csv(csv_path, df)

    with pytest.raises(
        ValueError, match=r"malformed keypoints.*knee|missing x_ or y_ partner"
    ):
        _ = wide_pose_csv_to_long(csv_path, fps=None)


def test_time_wins_over_frame_and_frame_logged(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "time": [0.0, 0.1],
            "frame": [0, 1],
            "x_nose": [1, 2],
            "y_nose": [3, 4],
        }
    )
    csv_path = tmp_path / "trial_tf.csv"
    _write_csv(csv_path, df)

    res = wide_pose_csv_to_long(csv_path, fps=999.0)  # should be irrelevant
    out = res.df_long

    assert "time" in out.columns
    assert "frame" not in out.columns
    assert "frame" in res.trial_meta["recognized_but_unused_columns"]


def test_z_and_conf_optional_handling(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.2],
            "x_a": [1, 2, 3],
            "y_a": [4, 5, 6],
            "z_a": [7, 8, 9],
            "conf_a": [0.1, 0.2, 0.3],
            "x_b": [10, 11, 12],
            "y_b": [13, 14, 15],
            # b has no z/conf
        }
    )
    csv_path = tmp_path / "trial_z.csv"
    _write_csv(csv_path, df)

    res = wide_pose_csv_to_long(csv_path, fps=None)
    out = res.df_long

    assert "z" in out.columns
    assert "conf" in out.columns

    # Keypoint b should have missing z/conf values (present but NA)
    b = out[out["keypoint"] == "b"]
    assert b["z"].isna().all()
    assert b["conf"].isna().all()


def test_directory_ingest_writes_outputs(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"

    df1 = pd.DataFrame(
        {
            "time": [0.0, 0.1],
            "x_nose": [1, 2],
            "y_nose": [3, 4],
            "notes": ["hi", "bye"],  # ignored
        }
    )
    df2 = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "x_1": [1, 2, 3],
            "y_1": [4, 5, 6],
        }
    )
    _write_csv(in_dir / "A.csv", df1)
    _write_csv(in_dir / "B.csv", df2)

    # Mixed timing modes are allowed; frame-based trial needs fps
    ingest_pose_csv_dir(in_path=in_dir, out_dir=out_dir, fps=30.0)

    pose_path = out_dir / "pose.parquet"
    rec_path = out_dir / "recording.json"
    qc_path = out_dir / "qc_ingest.json"

    assert pose_path.exists()
    assert rec_path.exists()
    assert qc_path.exists()

    pose = pd.read_parquet(pose_path)
    rec = _read_json(rec_path)
    qc = _read_json(qc_path)

    # Recording has trials and logs ignored columns for A.csv
    assert "trials" in rec
    assert len(rec["trials"]) == 2

    trialA = next(t for t in rec["trials"] if t["trial_id"] == "A")
    assert "notes" in trialA["ignored_columns"]

    # qc has one row per trial
    assert isinstance(qc, list)
    assert {row["trial_id"] for row in qc} == {"A", "B"}

    # pose has both trials
    assert set(pose["trial_id"].unique()) == {"A", "B"}
