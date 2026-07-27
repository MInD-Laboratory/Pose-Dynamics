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


def test_resolve_rois_excludes_lower_body_keypoints():
    # Lower-body points (legs/feet) are occluded/unreliable in a seated
    # conversation and must NOT be pulled into the Procrustes fit -- resolve_rois
    # restricts to the curated `SELECTED_KEYPOINTS`, not every header column.
    cols = []
    for n in ["Nose", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
              "Neck", "MidHip", "REye", "LEye", "rightOuterEyeBrow",
              "RKnee", "LKnee", "RAnkle", "LAnkle", "RBigToe"]:
        cols += [f"{n}_confidence", f"{n}_x_offset", f"{n}_y_offset"]

    names, roi_map = resolve_rois(cols)
    assert not ({"RKnee", "LKnee", "RAnkle", "LAnkle", "RBigToe"} & set(names))
    # a face landmark from the curated selection is kept, and (in this design,
    # every curated keypoint matches some ROI's exact list or substring match)
    # correctly resolves into centre_face via the "Eye" substring
    assert "rightOuterEyeBrow" in names
    roi_members = {n for idx in roi_map.values() for n in [names[i] for i in idx]}
    assert "rightOuterEyeBrow" in roi_members


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


def test_build_global_template_averages_and_checks_consistency():
    from pose_dynamics.data import PoseSequence
    from pose_dynamics.case_studies.mosaic.reproduce import build_global_template

    names = ["Nose", "RShoulder", "LShoulder"]
    base = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])

    def make(offset):
        coords = np.tile(base + offset, (5, 1, 1))
        return PoseSequence(coords=coords, keypoint_names=names, frame_rate=10.0)

    seqs = [make(np.array([1.0, 1.0])), make(np.array([-1.0, -1.0]))]
    template = build_global_template(seqs)
    np.testing.assert_allclose(template, base, atol=1e-9)

    mismatched = PoseSequence(
        coords=np.tile(base, (5, 1, 1)), keypoint_names=["Nose", "RShoulder", "Other"],
        frame_rate=10.0,
    )
    with pytest.raises(ValueError, match="same keypoint set"):
        build_global_template([seqs[0], mismatched])


def test_windowed_align_recovers_rotation_scale_and_translation():
    from pose_dynamics.data import PoseSequence
    from pose_dynamics.case_studies.mosaic.reproduce import windowed_align
    from pose_dynamics.case_studies.mosaic import config as C

    names = ["Nose", "RShoulder", "LShoulder", "Neck", "MidHip"]
    template = np.array([[0.0, 0.0], [2.0, 0.5], [-2.0, 0.5], [0.0, 1.0], [0.0, -1.5]])

    # Simulate one camera's fixed rotation/scale/offset relative to the template
    # (e.g. a different seating angle/distance), held constant across the trial.
    theta = 0.3
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scale = 1.4
    translation = np.array([5.0, -3.0])
    observed_pose = (template @ R) * scale + translation

    frame_rate = 10.0
    win_frames = int(round(C.WINDOW_S * frame_rate))
    n_frames = win_frames + int(round(win_frames * (1 - C.OVERLAP))) + 10
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.01, size=(n_frames, len(names), 2))
    coords = observed_pose[None, :, :] + noise

    seq = PoseSequence(coords=coords, keypoint_names=names, frame_rate=frame_rate)
    windows = windowed_align(seq, template)
    assert len(windows) >= 2  # confirms overlap actually produced >1 window
    for w, aligned in windows:
        np.testing.assert_allclose(aligned.mean(axis=0), template, atol=0.05)


def test_windowed_align_nans_whole_window_when_nose_entirely_missing():
    from pose_dynamics.data import PoseSequence
    from pose_dynamics.case_studies.mosaic.reproduce import windowed_align
    from pose_dynamics.case_studies.mosaic import config as C

    names = ["Nose", "RShoulder", "LShoulder", "Neck"]
    template = np.array([[0.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [0.0, 1.0]])
    frame_rate = 10.0
    win_frames = int(round(C.WINDOW_S * frame_rate))

    coords = np.tile(template, (win_frames, 1, 1))
    coords[:, 0, :] = np.nan  # Nose entirely missing for the whole window

    seq = PoseSequence(coords=coords, keypoint_names=names, frame_rate=frame_rate)
    windows = windowed_align(seq, template)
    assert len(windows) == 1
    _, aligned = windows[0]
    # centering is undefined without the nose, so the *entire* window -- every
    # keypoint, not just the nose -- must come back NaN.
    assert not np.any(np.isfinite(aligned))


def test_windowed_align_drops_keypoints_below_valid_frac_threshold():
    from pose_dynamics.data import PoseSequence
    from pose_dynamics.case_studies.mosaic.reproduce import windowed_align
    from pose_dynamics.case_studies.mosaic import config as C

    names = ["Nose", "RShoulder", "LShoulder", "Neck", "MidHip"]
    template = np.array([[0.0, 0.0], [2.0, 0.5], [-2.0, 0.5], [0.0, 1.0], [0.0, -1.5]])
    frame_rate = 10.0
    win_frames = int(round(C.WINDOW_S * frame_rate))

    coords = np.tile(template, (win_frames, 1, 1)).astype(float)
    # MidHip valid in only 10% of the window's frames -- below the 20% threshold,
    # so it must be excluded from the fit (but the window itself still processed).
    sparse_valid = int(win_frames * 0.10)
    coords[sparse_valid:, 4, :] = np.nan

    seq = PoseSequence(coords=coords, keypoint_names=names, frame_rate=frame_rate)
    windows = windowed_align(seq, template)
    assert len(windows) == 1
    _, aligned = windows[0]
    # the four well-covered keypoints still align correctly...
    np.testing.assert_allclose(aligned[:sparse_valid, :4, :].mean(axis=0), template[:4], atol=0.05)
    # ...MidHip wasn't dropped from the *output*, just from the fit -- its finite
    # values still got the fitted transform applied.
    assert np.all(np.isfinite(aligned[:sparse_valid, 4, :]))
