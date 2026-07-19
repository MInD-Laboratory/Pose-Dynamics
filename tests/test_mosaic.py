"""Tests for Case-2 (MOSAIC) helpers: filename, conditions, ROI resolution, dyadic guard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pose_dynamics.case_studies.mosaic import (
    config as C,
    load_condition_map,
    parse_file,
    resolve_rois,
    run_reproduction,
)


def test_parse_filename():
    assert parse_file("S014_T2_right.csv") == (14, 2, "right")
    assert parse_file("/x/S007_T5_left.csv") == (7, 5, "left")


def test_condition_map(tmp_path):
    csv = tmp_path / "cond.csv"
    csv.write_text("session,trial,condition\n14,1,Office\n14,2,Food\n14,6,Party\n")
    cm = load_condition_map(csv)
    assert cm[(14, 1)] == "Office"
    assert cm[(14, 6)] == "Party"


def test_resolve_rois_maps_names_to_body25_groups():
    # a header with body-25 names + a couple of face landmarks
    body = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
            "LWrist", "MidHip", "REye", "LEye"]
    face = ["leftUpperLip", "rightBottomLip", "leftEdgeEyeLeft"]
    cols = []
    for n in body + face:
        cols += [f"{n}_confidence", f"{n}_x_offset", f"{n}_y_offset"]

    names, roi_map = resolve_rois(cols)
    # arms = the six arm keypoints (BODY_25 {2,3,4,5,6,7})
    arm_names = {names[i] for i in roi_map["arms"]}
    assert arm_names == {"RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist"}
    # upper body = Neck, R/L Shoulder, MidHip
    ub = {names[i] for i in roi_map["upper_body"]}
    assert ub == {"Neck", "RShoulder", "LShoulder", "MidHip"}
    # centre_face includes Nose/eyes plus matched face landmarks (Lip/Eye substrings)
    cf = {names[i] for i in roi_map["centre_face"]}
    assert {"Nose", "REye", "LEye"} <= cf
    assert "leftUpperLip" in cf and "leftEdgeEyeLeft" in cf


def test_dyadic_run_requires_both_cameras(tmp_path):
    # only a 'right' file present -> the dyadic reproduction must fail clearly
    cols = []
    for n in ["RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "Neck", "MidHip", "Nose", "REye", "LEye"]:
        cols += [f"{n}_confidence", f"{n}_x_offset", f"{n}_y_offset"]
    df = pd.DataFrame({c: np.ones(200) * 100 for c in cols})
    (tmp_path / "S014_T1_right.csv").write_text(df.to_csv(index=False))
    cond = tmp_path / "cond.csv"
    cond.write_text("session,trial,condition\n14,1,Office\n")
    with pytest.raises(FileNotFoundError, match="both partners"):
        run_reproduction(tmp_path, cond, progress=False)


def test_rqa_params_use_case2_values():
    from pose_dynamics.case_studies.mosaic.reproduce import auto_params, cross_params
    ap, cp = auto_params(), cross_params()
    assert (ap.eDim, ap.tLag) == (4, 10)        # numeric_inventory 8.14
    assert ap.radius == 0.2 and ap.rescale == "mean"  # 8.15
    assert cp.theiler_for("cross") == 0          # 8.17
    assert cp.min_line == 2
