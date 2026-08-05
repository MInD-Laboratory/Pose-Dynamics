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


def test_run_reproduction_sessions_filter(tmp_path, monkeypatch):
    # sessions= should restrict which session-trials get processed, and the
    # template should be built only from that subset too.
    cols = []
    for n in ["RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "Neck", "MidHip", "Nose", "REye", "LEye"]:
        cols += [f"{n}_confidence", f"{n}_x_offset", f"{n}_y_offset"]
    df = pd.DataFrame({c: np.ones(30) * 100 for c in cols})
    for session in (1, 2, 3):
        (tmp_path / f"S{session:03}_T1_left.csv").write_text(df.to_csv(index=False))
        (tmp_path / f"S{session:03}_T1_right.csv").write_text(df.to_csv(index=False))
    cond = tmp_path / "cond.csv"
    cond.write_text("session,trial,condition\n1,1,Office\n2,1,Office\n3,1,Office\n")

    import pose_dynamics.case_studies.mosaic.reproduce as reproduce_mod

    seen_sessions = []

    def fake_process_dyad(right, left, roi_map, condition, template, session=None, trial=None, align=True):
        seen_sessions.append(session)
        return [{"session": session, "trial": trial, "condition": condition, "roi": "arms",
                 "window": 0, "cross_perc_recur": 0.0, "cross_perc_determ": 0.0,
                 "cross_lmax": 0.0, "xcorr_lag0": 0.0}]

    monkeypatch.setattr(reproduce_mod, "process_dyad", fake_process_dyad)
    out = reproduce_mod.run_reproduction(tmp_path, cond, sessions=[1, 3], progress=False)
    assert sorted(seen_sessions) == [1, 3]
    assert set(out["session"]) == {1, 3}


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


def test_build_global_template_is_frame_weighted_not_file_weighted():
    # A file with more valid frames must contribute proportionally more than a
    # mean-of-per-file-means would give it -- guards against reverting to
    # equal-per-file weighting.
    from pose_dynamics.data import PoseSequence
    from pose_dynamics.case_studies.mosaic.reproduce import build_global_template

    names = ["Nose", "RShoulder"]
    short = PoseSequence(
        coords=np.tile(np.array([[0.0, 0.0], [1.0, 0.0]]), (1, 1, 1)),
        keypoint_names=names, frame_rate=10.0,
    )
    long = PoseSequence(
        coords=np.tile(np.array([[10.0, 0.0], [11.0, 0.0]]), (9, 1, 1)),
        keypoint_names=names, frame_rate=10.0,
    )
    template = build_global_template([short, long])
    # Frame-weighted: (1*0 + 9*10)/10 = 9.0; mean-of-means would give (0+10)/2 = 5.0.
    np.testing.assert_allclose(template[:, 0], [9.0, 10.0], atol=1e-9)


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

    # align=False must skip the Procrustes fit -- the rotated/scaled pose should
    # NOT be pulled back onto the template, only nose-centred (no template needed).
    unaligned_windows = windowed_align(seq, None, align=False)
    assert len(unaligned_windows) == len(windows)
    nose_pos = observed_pose[0]
    for w, unaligned in unaligned_windows:
        assert not np.allclose(unaligned.mean(axis=0), template, atol=0.05)
        np.testing.assert_allclose(unaligned.mean(axis=0), observed_pose - nose_pos, atol=0.05)


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


def test_windowed_align_drops_keypoints_below_valid_frac_threshold(monkeypatch):
    from pose_dynamics.data import PoseSequence
    from pose_dynamics.case_studies.mosaic.reproduce import windowed_align
    from pose_dynamics.case_studies.mosaic import config as C

    # This threshold only applies to partially-observed windows, which the default
    # "all_keypoints" rule never admits -- so the behaviour is specific to "per_roi".
    monkeypatch.setattr(C, "WINDOW_COMPLETENESS", "roi_available")

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


def test_roi_complete_voids_only_the_affected_roi(monkeypatch):
    """The point of "roi_complete": a missing member voids that ROI's signal for the
    window and nothing else. Under the prototype's "all_keypoints" rule the same gap
    would also void `arms`, which never uses the missing keypoint."""
    import numpy as np
    from pose_dynamics.case_studies.mosaic import config as C
    from pose_dynamics.case_studies.mosaic.reproduce import _window_roi_speeds

    names = ["Nose", "REye", "RShoulder", "RElbow", "RWrist"]
    rois = {"centre_face": [0, 1], "arms": [2, 3, 4]}
    n = 200
    rng = np.random.default_rng(0)
    coords = np.cumsum(rng.normal(0, 0.01, (n, len(names), 2)), axis=0)
    coords[50, 1, 0] = np.nan                      # one missing frame, REye only

    monkeypatch.setattr(C, "WINDOW_COMPLETENESS", "roi_complete")
    feats = _window_roi_speeds(coords, names, rois, 30.0)
    face, arms = feats.get("centre_face_speed"), feats.get("arms_speed")
    assert np.all(np.isnan(face)), "the ROI containing the gap must be voided"
    assert np.all(np.isfinite(arms)), "an unaffected ROI must survive"

    # the permissive mode keeps both (and is where the spike artifact lives)
    monkeypatch.setattr(C, "WINDOW_COMPLETENESS", "roi_available")
    feats = _window_roi_speeds(coords, names, rois, 30.0)
    assert np.isfinite(feats.get("centre_face_speed")).any()
    assert np.all(np.isfinite(feats.get("arms_speed")))


def test_window_completeness_all_keypoints_matches_prototype_rule(monkeypatch):
    """The published prototype discarded a window outright, for every ROI, if any
    selected keypoint was missing at any frame (`window.isnull().any().any()`).
    ``WINDOW_COMPLETENESS="all_keypoints"`` must reproduce that; ``"per_roi"`` must not."""
    from pose_dynamics.data import PoseSequence
    from pose_dynamics.case_studies.mosaic import config as C
    from pose_dynamics.case_studies.mosaic.reproduce import windowed_align

    names = ["Nose", "Neck", "RShoulder", "LShoulder", "MidHip"]
    template = np.array([[0.0, 2.0], [0.0, 1.5], [-1.0, 1.5], [1.0, 1.5], [0.0, 0.0]])
    frame_rate = 30.0
    n = int(frame_rate * C.WINDOW_S * 3)          # room for several windows
    coords = np.tile(template, (n, 1, 1)).astype(float)
    coords += np.random.default_rng(0).normal(0, 0.01, coords.shape)

    # a single missing value, in one keypoint, at one frame, inside the first window
    coords[5, 4, 1] = np.nan
    seq = PoseSequence(coords=coords, keypoint_names=names, frame_rate=frame_rate)

    monkeypatch.setattr(C, "WINDOW_COMPLETENESS", "all_keypoints")
    strict = windowed_align(seq, template)
    monkeypatch.setattr(C, "WINDOW_COMPLETENESS", "roi_available")
    lenient = windowed_align(seq, template)

    # the prototype's rule voids the affected window entirely...
    assert np.all(np.isnan(strict[0][1])), "incomplete window should be voided"
    # ...including keypoints that were never missing themselves
    assert np.all(np.isnan(strict[0][1][:, 1, :])), "void must apply to every keypoint"
    # ...while later, complete windows survive
    assert np.all(np.isfinite(strict[-1][1]))
    # the per-ROI rule keeps it
    assert np.isfinite(lenient[0][1]).any(), "roi_available should retain the window"

    n_voided = sum(1 for _w, a in strict if np.all(np.isnan(a)))
    assert n_voided < len(strict), "not every window should be voided by one bad frame"


def _limb_fixture():
    """Template plus a sequence whose arms both swing and extend."""
    names = ["Nose", "Neck", "LShoulder", "LElbow", "LWrist",
             "RShoulder", "RElbow", "RWrist", "MidHip"]
    template = np.array([
        [0.0, 2.0], [0.0, 1.5], [-1.0, 1.5], [-2.0, 1.0], [-3.0, 0.5],
        [1.0, 1.5], [2.0, 1.0], [3.0, 0.5], [0.0, 0.0],
    ])
    n = 40
    coords = np.tile(template, (n, 1, 1)).astype(float)
    # stretch the left arm outward over time (radial motion) and swing the right
    t = np.linspace(0.0, 1.0, n)
    coords[:, 3, 0] = -2.0 - t              # LElbow drifts out
    coords[:, 4, 0] = -3.0 - 2 * t          # LWrist drifts out further
    coords[:, 6, 1] = 1.0 + 0.5 * t         # RElbow swings up
    return names, template, coords


def test_limb_rescaling_enforces_template_lengths_and_keeps_angles():
    from pose_dynamics.case_studies.mosaic.reproduce import (
        apply_fixed_limb_lengths, compute_reference_limb_lengths,
    )

    names, template, coords = _limb_fixture()
    refs = compute_reference_limb_lengths(template, names)
    assert [pair for pair, _ in refs] == [(2, 3), (3, 4), (5, 6), (6, 7)], "chain order matters"

    out = apply_fixed_limb_lengths(coords, refs)

    def _parallel(a, b):
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    for (i, j), target in refs:
        lengths = np.linalg.norm(out[:, j] - out[:, i], axis=1)
        np.testing.assert_allclose(lengths, target, rtol=1e-9)
        # Direction is preserved relative to the *corrected* proximal joint, which is
        # what a chained correction can offer: the distal segment is rescaled after its
        # parent has already moved, so it is NOT parallel to the original segment.
        v_out = out[:, j] - out[:, i]
        np.testing.assert_allclose(_parallel(coords[:, j] - out[:, i], v_out), 0.0, atol=1e-9)

    # For the proximal-most segment of each chain the parent never moves, so there the
    # original joint angle really is preserved.
    for i, j in ((2, 3), (5, 6)):
        np.testing.assert_allclose(
            _parallel(coords[:, j] - coords[:, i], out[:, j] - out[:, i]), 0.0, atol=1e-9)


def test_limb_rescaling_touches_only_elbows_and_wrists():
    """The reason the arms ROI is the only one affected -- asserted, not assumed."""
    from pose_dynamics.case_studies.mosaic.reproduce import (
        apply_fixed_limb_lengths, compute_reference_limb_lengths,
    )

    names, template, coords = _limb_fixture()
    out = apply_fixed_limb_lengths(coords, compute_reference_limb_lengths(template, names))

    moved = {names[k] for k in range(len(names))
             if not np.allclose(out[:, k], coords[:, k], atol=1e-12)}
    assert moved <= {"LElbow", "LWrist", "RElbow", "RWrist"}
    for untouched in ("Nose", "Neck", "LShoulder", "RShoulder", "MidHip"):
        np.testing.assert_allclose(out[:, names.index(untouched)],
                                   coords[:, names.index(untouched)], atol=1e-12)


def test_limb_rescaling_removes_radial_but_not_swing_motion():
    """The behavioural cost: reaching in/out is deleted, swinging survives."""
    from pose_dynamics.case_studies.mosaic.reproduce import (
        apply_fixed_limb_lengths, compute_reference_limb_lengths,
    )

    names, template, coords = _limb_fixture()
    refs = compute_reference_limb_lengths(template, names)
    out = apply_fixed_limb_lengths(coords, refs)

    # left arm was purely radial extension -> its length variation is gone
    before = np.ptp(np.linalg.norm(coords[:, 4] - coords[:, 2], axis=1))
    after = np.ptp(np.linalg.norm(out[:, 4] - out[:, 2], axis=1))
    assert before > 0.5 and after < 1e-9

    # right elbow swung (rotation at fixed-ish radius) -> it still moves
    assert np.ptp(out[:, 6], axis=0).max() > 1e-3


def test_windowed_align_applies_limb_rescaling_when_enabled(monkeypatch):
    """End-to-end: the stage is wired into windowed_align for both align modes."""
    from pose_dynamics.data import PoseSequence
    from pose_dynamics.case_studies.mosaic import config as C
    from pose_dynamics.case_studies.mosaic.reproduce import windowed_align

    names, template, coords = _limb_fixture()
    frame_rate, n = 30.0, int(30.0 * 130)
    coords = np.tile(coords[:1], (n, 1, 1))
    coords[:, 4, 0] = -3.0 - np.linspace(0, 2, n)      # LWrist reaches outward
    seq = PoseSequence(coords=coords, keypoint_names=names, frame_rate=frame_rate)

    for align in (True, False):
        monkeypatch.setattr(C, "APPLY_LIMB_RESCALE", True)
        on = windowed_align(seq, template, align=align)
        monkeypatch.setattr(C, "APPLY_LIMB_RESCALE", False)
        off = windowed_align(seq, template, align=align)

        _, first_on = on[0]
        _, first_off = off[0]
        len_on = np.linalg.norm(first_on[:, 4] - first_on[:, 3], axis=1)
        len_off = np.linalg.norm(first_off[:, 4] - first_off[:, 3], axis=1)
        assert np.ptp(len_on) < 1e-9, f"align={align}: rescaling not applied"
        assert np.ptp(len_off) > 1e-6, f"align={align}: control should vary"


def test_normalization_is_isotropic_and_preserves_pose_shape():
    """Coordinate normalization must divide both axes by the same constant.

    A per-axis divisor (e.g. the nominal 1920x1080) rescales x and y differently,
    which changes pose *shape*, not just units: ``windowed_align`` fits rotation +
    uniform scale and cannot undo it, and RQA's per-window z-scoring cancels an
    isotropic factor but not a per-axis one. Asserted behaviourally on a segment
    ratio so this fails if the divisors ever diverge again.
    """
    from pose_dynamics.data import PoseSequence
    from pose_dynamics.case_studies.mosaic.reproduce import preprocess_pose
    from pose_dynamics.case_studies.mosaic import config as C

    assert C.VIDEO_WIDTH == C.VIDEO_HEIGHT, "normalization divisors must be isotropic"

    names = ["Nose", "RShoulder", "LShoulder", "Neck", "MidHip"]
    # RShoulder->LShoulder is purely horizontal; Neck->MidHip purely vertical.
    # Both are exactly 100 px, so any anisotropy shows up as a ratio != 1.
    base = np.array([[850.0, 300.0], [800.0, 500.0], [900.0, 500.0],
                     [850.0, 400.0], [850.0, 500.0]])
    n_frames = 600
    coords = np.tile(base, (n_frames, 1, 1)).astype(float)
    seq = PoseSequence(coords=coords, keypoint_names=names, frame_rate=60.0,
                       confidence=np.ones((n_frames, len(names))))

    out = preprocess_pose(seq).coords
    horizontal = np.linalg.norm(out[:, 2, :] - out[:, 1, :], axis=1)
    vertical = np.linalg.norm(out[:, 4, :] - out[:, 3, :], axis=1)
    np.testing.assert_allclose(horizontal / vertical, 1.0, rtol=1e-9)


def _moving_nose_static_arms(n_frames: int = 1200, frame_rate: float = 60.0):
    """Nose translating on a slow sinusoid; arm keypoints perfectly stationary.

    Any pipeline stage that expresses keypoints relative to the *current frame's*
    nose turns the arms' zero motion into a mirror of the nose's motion, so this
    fixture separates a genuine translation-invariance fix from a change of
    reference frame.
    """
    from pose_dynamics.data import PoseSequence

    names = ["Nose", "RShoulder", "RElbow", "RWrist", "Neck"]
    base = np.array([[900.0, 400.0], [800.0, 500.0], [750.0, 620.0],
                     [700.0, 730.0], [850.0, 480.0]])
    coords = np.tile(base, (n_frames, 1, 1)).astype(float)

    t = np.arange(n_frames) / frame_rate
    nose_shift = 100.0 * np.sin(2 * np.pi * 0.5 * t)   # 0.5 Hz, well inside the 10 Hz band
    coords[:, 0, 0] += nose_shift
    coords[:, 0, 1] += 0.5 * nose_shift

    seq = PoseSequence(coords=coords, keypoint_names=names, frame_rate=frame_rate,
                       confidence=np.ones((n_frames, len(names))))
    return seq, names


def test_preprocess_pose_does_not_re_reference_keypoints_to_the_nose():
    """Regression: preprocessing must not inject the nose's motion into other
    keypoints. Case 2 centres on each window's *mean* nose position
    (``windowed_align``); a per-frame subtraction here would make every keypoint's
    velocity ``v_k - v_nose``."""
    from pose_dynamics.case_studies.mosaic.reproduce import preprocess_pose

    seq, _ = _moving_nose_static_arms()
    out = preprocess_pose(seq)

    static = out.coords[:, 1:, :]                  # everything except the nose
    assert np.all(np.isfinite(static))
    assert np.ptp(static, axis=0).max() < 1e-9, "preprocessing injected motion into static keypoints"

    nose_excursion = np.ptp(out.coords[:, 0, :], axis=0).max()
    assert nose_excursion > 1e-3, "fixture broken: the nose should still be moving"


def test_static_roi_has_zero_velocity_magnitude_while_the_nose_moves():
    """The metric the re-referencing actually corrupted: a stationary ROI must have
    zero velocity magnitude regardless of head translation."""
    from pose_dynamics.case_studies.mosaic.reproduce import _window_roi_speeds, preprocess_pose

    seq, names = _moving_nose_static_arms()
    out = preprocess_pose(seq)
    rois = {"arms": [names.index(k) for k in ("RShoulder", "RElbow", "RWrist")],
            "face": [names.index("Nose")]}

    feats = _window_roi_speeds(out.coords, out.keypoint_names, rois, out.frame_rate)
    assert np.nanmax(np.abs(feats.get("arms_speed"))) < 1e-8
    assert np.nanmax(np.abs(feats.get("face_speed"))) > 1e-3


def test_windowed_align_mean_nose_centering_leaves_roi_speeds_unchanged():
    """The per-window mean-nose offset is a constant, so it must not move the ROI
    velocity magnitudes -- the property that makes it safe where a per-frame
    subtraction is not."""
    from pose_dynamics.case_studies.mosaic.reproduce import (
        _window_roi_speeds, preprocess_pose, windowed_align,
    )

    seq, names = _moving_nose_static_arms(n_frames=60 * 130)   # > 2 windows at 60 s
    out = preprocess_pose(seq)
    rois = {"arms": [names.index(k) for k in ("RShoulder", "RElbow", "RWrist")]}

    windows = windowed_align(out, template=None, align=False)
    assert len(windows) >= 2
    for w, centred in windows:
        raw = out.coords[w.start:w.stop]
        got = _window_roi_speeds(centred, out.keypoint_names, rois, out.frame_rate).get("arms_speed")
        want = _window_roi_speeds(raw, out.keypoint_names, rois, out.frame_rate).get("arms_speed")
        np.testing.assert_allclose(got, want, atol=1e-12)


def _fake_individual_frame():
    """Minimal run_individual-shaped frame: 12 pairs x 2 partners x 6 trials x 1 ROI."""
    rng = np.random.default_rng(0)
    trial_cond = {1: "Office", 3: "Office", 5: "Office", 2: "Cafe", 4: "Food", 6: "Party"}
    effect = {"Office": 0.0, "Cafe": 0.002, "Food": 0.010, "Party": 0.020}
    rows = []
    for session in range(1, 13):
        for camera in ("left", "right"):
            who = rng.normal(0, 0.003)
            for trial, cond in trial_cond.items():
                for window in range(4):
                    base = 0.05 + who + effect[cond] + rng.normal(0, 0.001)
                    rows.append({"session": session, "trial": trial, "camera": camera,
                                 "condition": cond, "roi": "arms", "window": window,
                                 "rms": base, "mean_vel": base * 0.7, "sd_vel": base * 0.5})
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=C.CONDITION_ORDER,
                                     ordered=True)
    return df


def test_containment_df_matches_the_published_convention():
    from pose_dynamics.case_studies.mosaic.stats import containment_df
    # the individual arms model: 548 trial rows, 4 fixed effects, 93 participants
    assert containment_df(548, 4, 93) == 451      # paper reports t(451)
    # the dyadic arms model: 270 dyad-trials, 46 pairs
    assert containment_df(270, 4, 46) == 220      # paper reports t(219-221)


def test_to_trial_individual_collapses_windows_and_keys_on_participant():
    from pose_dynamics.case_studies.mosaic.stats import to_trial_individual

    df = _fake_individual_frame()
    trial = to_trial_individual(df)
    assert len(trial) == 12 * 2 * 6, "one row per participant x trial x ROI"
    assert trial["participant"].nunique() == 24
    # aggregation must not mutate the caller's frame
    assert "participant" not in df.columns
    # a trial's value is the mean of its windows
    key = dict(session="1", trial=1, camera="left", roi="arms")
    got = trial.query(" and ".join(f"{k} == @key['{k}']" for k in key))["rms"].iloc[0]
    want = df[(df.session == 1) & (df.trial == 1) & (df.camera == "left")]["rms"].mean()
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_fit_individual_recovers_effects_and_reports_containment_df():
    """Guards the patsy `C()` shadowing trap: a bare `C` in the stats module's globals
    breaks vc_formula with "'module' object is not callable"."""
    from pose_dynamics.case_studies.mosaic.stats import (
        fit_individual, to_trial_individual,
    )

    trial = to_trial_individual(_fake_individual_frame())
    coefs = fit_individual(trial)
    rms = coefs[coefs.metric == "rms"].set_index("vs_Office")
    np.testing.assert_allclose(rms.loc["Party", "beta"], 0.020, atol=0.003)
    np.testing.assert_allclose(rms.loc["Food", "beta"], 0.010, atol=0.003)
    # df is containment, not the naive n - n_fixed
    n_obs = int(rms.loc["Party", "n_obs"])
    assert rms.loc["Party", "df"] == n_obs - 4 - trial["participant"].nunique()
    assert rms.loc["Party", "df"] < n_obs - 4


def test_tukey_is_never_less_conservative_than_the_reference_contrasts():
    from pose_dynamics.case_studies.mosaic.stats import (
        METRICS_IND, check_tukey_conservative, fit_individual, to_trial_individual,
        tukey_pairwise,
    )

    trial = to_trial_individual(_fake_individual_frame())
    ref = fit_individual(trial)
    tk = tukey_pairwise(trial, METRICS_IND, "individual")
    assert len(tk) == 3 * 6, "six pairwise contrasts per metric, one ROI"
    assert check_tukey_conservative(tk, ref) == len(ref)
    # and the two tables agree on the denominator df
    assert set(tk["df"]) == set(ref["df"])


def test_source_digest_detects_a_stale_module():
    """Backs the notebook guard. A kernel keeps modules in ``sys.modules``, so an edit to
    the file does not reach a session that already imported it. Each module records a digest
    of its own source at import, and comparing that against a fresh read catches the
    mismatch -- something ``inspect.getsource`` cannot do, since it reads from disk too and
    therefore always agrees with disk regardless of what is loaded."""
    import hashlib
    from pathlib import Path

    from pose_dynamics.case_studies.mosaic import reproduce, stats

    for mod in (reproduce, stats):
        fresh = hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()[:12]
        assert mod._SOURCE_SHA == fresh, f"{mod.__name__} digest disagrees with its file"

    # and it must discriminate, or the check above would pass for any content at all
    edited = Path(stats.__file__).read_bytes() + b"\n# an edit\n"
    assert hashlib.sha256(edited).hexdigest()[:12] != stats._SOURCE_SHA


def test_coefficient_table_carries_the_intercept_so_condition_means_are_recoverable():
    """A contrast establishes that an effect exists; only the level says what shape it has,
    or whether a near-zero measure changed sign rather than magnitude."""
    from pose_dynamics.case_studies.mosaic.stats import fit_individual, to_trial_individual

    trial = to_trial_individual(_fake_individual_frame())
    coefs = fit_individual(trial)
    assert "intercept" in coefs.columns

    rms = coefs[coefs.metric == "rms"].set_index("vs_Office")
    arms = trial[trial.roi == "arms"]
    # the intercept is the Office mean, and intercept + beta is that condition's mean
    np.testing.assert_allclose(rms["intercept"].iloc[0],
                               arms[arms.condition == "Office"]["rms"].mean(), atol=2e-3)
    for cond in ("Cafe", "Food", "Party"):
        np.testing.assert_allclose(rms.loc[cond, "intercept"] + rms.loc[cond, "beta"],
                                   arms[arms.condition == cond]["rms"].mean(), atol=2e-3)


def test_boundary_warning_is_dropped_but_real_failures_are_escalated():
    """statsmodels' boundary notice fires on an absolute variance threshold of 0.01, so
    metrics on ~0.01-0.03 trip it regardless of fit quality. It is dropped; anything that
    signals an actually untrustworthy fit must raise instead."""
    import warnings

    from pose_dynamics.case_studies.mosaic.stats import (
        _BOUNDARY_MSG, _check_fit, fit_individual, to_trial_individual,
    )

    class _Res:
        def __init__(self, converged=True):
            self.converged = converged

    def _caught(msg, category=UserWarning):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            warnings.warn(msg, category)
        return list(rec)

    # benign: dropped, and not re-emitted
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _check_fit(_caught(_BOUNDARY_MSG), _Res(), "rms", "individual")
    assert rec == []

    # a non-converged optimiser is a hard error even with no warning attached
    with pytest.raises(RuntimeError, match="did not converge"):
        _check_fit([], _Res(converged=False), "rms", "individual")

    # so is a singular covariance
    with pytest.raises(RuntimeError, match="fit failed"):
        _check_fit(_caught("The random effects covariance matrix is singular"),
                   _Res(), "rms", "individual")

    # anything unrecognised is passed through rather than swallowed
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _check_fit(_caught("something new from statsmodels"), _Res(), "rms", "individual")
    assert len(rec) == 1

    # end to end: the real fits are warning-free and report convergence
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        coefs = fit_individual(to_trial_individual(_fake_individual_frame()))
    assert rec == [], f"unexpected warnings: {[str(w.message) for w in rec]}"
    assert coefs["converged"].all()
