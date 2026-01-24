import json
from pathlib import Path

import pandas as pd

from pose_dynamics.preprocess.api import run_preprocess


def test_selection_and_windowing(tmp_path: Path) -> None:
    pose_path = tmp_path / "pose.parquet"
    recording_path = tmp_path / "recording.json"
    config_path = tmp_path / "preprocess.yml"
    out_dir = tmp_path / "out"

    # Build a minimal long-form pose table
    rows = []
    for t in [0.0, 1.0, 2.0, 3.0]:
        rows.append(
            {
                "trial_id": "t1",
                "source_file": "t1.csv",
                "time": t,
                "keypoint": "kp1",
                "x": 1.0,
                "y": 2.0,
            }
        )
        rows.append(
            {
                "trial_id": "t1",
                "source_file": "t1.csv",
                "time": t,
                "keypoint": "kp2",
                "x": 3.0,
                "y": 4.0,
            }
        )

    df = pd.DataFrame(rows)
    # Introduce a missing sample for kp1 at t=1.0
    df.loc[(df["keypoint"] == "kp1") & (df["time"] == 1.0), "x"] = pd.NA

    df.to_parquet(pose_path, index=False)

    recording_payload = {
        "trials": [
            {
                "trial_id": "t1",
                "timing_mode": "time",
                "fps_used": None,
            }
        ]
    }
    recording_path.write_text(json.dumps(recording_payload, indent=2), encoding="utf-8")

    config_path.write_text(
        """
preprocess:
  selection:
    keypoints: [kp1]
    exclude_keypoints: []
    dims: xy
    require_xyz: false
    keep_unselected: false

  windowing:
    enabled: true
    units: seconds
    length_s: 2.0
    step_s: 2.0
    include_partial: false
    qc_keypoints: all
    qc_dims: xy
    drop:
      enabled: true
      missing_rule: any_dim_nan
      max_missing_frac: 0.4
      max_nans: null
      scope: aggregate
      per_keypoint_policy: any
""",
        encoding="utf-8",
    )

    outputs = run_preprocess(
        pose_path=pose_path,
        recording_path=recording_path,
        config=config_path,
        out_dir=out_dir,
        overwrite=True,
    )

    assert outputs.pose_clean_path.exists()
    assert outputs.windows_path.exists()

    df_clean = pd.read_parquet(outputs.pose_clean_path)
    assert set(df_clean["keypoint"].unique()) == {"kp1"}

    windows = pd.read_parquet(outputs.windows_path)
    assert len(windows) == 2
    assert bool(windows.loc[0, "dropped"]) is True
    assert bool(windows.loc[1, "dropped"]) is False
