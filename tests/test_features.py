"""
Unit tests for feature primitives against known / prototype values, plus the
config composition, validation, and the Procrustes dual-output checkpoint.
"""
from __future__ import annotations

import numpy as np
import pytest

from pose_dynamics.data import PoseSequence
from pose_dynamics.features import (
    FeaturePipeline,
    PipelineValidationError,
    StreamType,
    build_primitive,
    procrustes_anisotropic,
    procrustes_uniform,
    registered_primitives,
)
from pose_dynamics.features.types import PipelineContext


def _pose(coords, dims=None, fps=60.0, names=None, conf=None):
    coords = np.asarray(coords, float)
    K = coords.shape[1]
    return PoseSequence(
        coords=coords,
        keypoint_names=names or [f"kp{i}" for i in range(K)],
        frame_rate=fps,
        confidence=conf,
    )


def _ctx(pose):
    return PipelineContext(pose=pose, features=None)


# ======================================================================
# Geometry math vs the prototype
# ======================================================================
def test_procrustes_uniform_rotation_matches_scipy():
    # MOSAIC used scipy.linalg.orthogonal_procrustes; cross-check the rotation
    # against it (the correct convention, aligning X0 onto Y0).
    from scipy.linalg import orthogonal_procrustes

    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 2))
    Y = rng.normal(size=(8, 2))
    X0, Y0 = X - X.mean(0), Y - Y.mean(0)
    R_scipy, _ = orthogonal_procrustes(X0, Y0)
    tp = procrustes_uniform(X, Y, allow_scale=False)
    np.testing.assert_allclose(tp.R, R_scipy, atol=1e-9)


def test_procrustes_uniform_recovers_known_transform():
    # A pure rotation+translation of an ASYMMETRIC template must be undone exactly
    # (an asymmetric shape makes the alignment unique).
    theta = np.deg2rad(35.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    template = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 3.0], [0.0, 1.0]])
    X = template @ R.T + np.array([3.0, -2.0])
    tp = procrustes_uniform(X, template, allow_scale=False)
    np.testing.assert_allclose(tp.apply(X), template, atol=1e-9)


def test_procrustes_anisotropic_decomposition_golden():
    # Golden values captured from the recovered decomposition (per-axis scales,
    # rotation, translation) for a fixed input — pins the numeric definition
    # without depending on the prototype source.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 2))
    Y = rng.normal(size=(10, 2))
    tp = procrustes_anisotropic(X, Y)
    np.testing.assert_allclose(tp.scale, [0.403354, 1.113833], atol=1e-5)
    np.testing.assert_allclose(
        tp.R, [[-0.855863, -0.517203], [0.517203, -0.855863]], atol=1e-5)
    np.testing.assert_allclose(tp.t, [0.022406, -0.039587], atol=1e-5)


def test_procrustes_anisotropic_recovers_transforms():
    template = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 3.0], [0.0, 1.0]])
    theta = 0.3
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # pure rotation+translation: recovers exactly, scales ~ 1
    X = template @ R + np.array([3.0, -2.0])
    tp = procrustes_anisotropic(X, template)
    np.testing.assert_allclose(tp.scale, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(tp.apply(X), template, atol=1e-9)
    # anisotropic stretch (x by 2) then rotate: recovers, scales reflect the stretch
    X2 = (template * np.array([2.0, 1.0])) @ R + np.array([1.0, 1.0])
    tp2 = procrustes_anisotropic(X2, template)
    np.testing.assert_allclose(tp2.apply(X2), template, atol=1e-9)


# ======================================================================
# Registry is populated
# ======================================================================
def test_registry_has_all_primitives():
    reg = registered_primitives()
    expected = {
        "coordinate_normalization", "center", "canonicalise", "procrustes",
        "select_keypoints", "roi_centroid", "distance_feature", "offset_feature",
        "coordinate_magnitude", "velocity_magnitude", "keypoint_coordinates",
        "zscore", "kinematic_derivatives",
    }
    assert expected <= set(reg)


# ======================================================================
# Geometry primitives (POSE -> POSE)
# ======================================================================
def test_coordinate_normalization_unit():
    coords = np.array([[[360.0, 720.0]]])  # (1 frame, 1 kp, 2d)
    prim = build_primitive("coordinate_normalization", {"width": 720, "height": 720})
    out = prim.apply(_ctx(_pose(coords)))
    np.testing.assert_allclose(out.pose.coords[0, 0], [0.5, 1.0])


def test_coordinate_normalization_centered():
    coords = np.array([[[720.0, 0.0]]])
    prim = build_primitive("coordinate_normalization",
                           {"width": 720, "height": 720, "mode": "centered"})
    out = prim.apply(_ctx(_pose(coords)))
    np.testing.assert_allclose(out.pose.coords[0, 0], [1.0, -1.0])  # edges -> [-1, 1]


def test_center_on_keypoint():
    coords = np.array([[[1.0, 1.0], [3.0, 5.0]]])  # kp0 is reference
    prim = build_primitive("center", {"reference": 0})
    out = prim.apply(_ctx(_pose(coords)))
    np.testing.assert_allclose(out.pose.coords[0, 0], [0.0, 0.0])
    np.testing.assert_allclose(out.pose.coords[0, 1], [2.0, 4.0])


def test_canonicalise_places_pelvis_at_origin():
    # pelvis=0, L-shoulder=1, R-shoulder=2, neck=3 in a simple 3-D pose
    coords = np.array([[[0, 0, 0], [-1, 2, 0], [1, 2, 0], [0, 3, 0], [0.5, 1, 0.2]]], float)
    prim = build_primitive("canonicalise",
                           {"pelvis": 0, "left_shoulder": 1, "right_shoulder": 2, "neck": 3})
    out = prim.apply(_ctx(_pose(coords)))
    np.testing.assert_allclose(out.pose.coords[0, 0], [0, 0, 0], atol=1e-9)  # pelvis -> origin
    # shoulder axis (L->R) lies along +x after rotation into the body frame
    shoulder_vec = out.pose.coords[0, 2] - out.pose.coords[0, 1]
    assert shoulder_vec[0] > 0
    np.testing.assert_allclose(shoulder_vec[1:], 0.0, atol=1e-9)


def test_select_keypoints():
    coords = np.arange(12).reshape(1, 3, 2 * 2)[..., :2].astype(float)  # dummy
    coords = np.array([[[0, 0], [1, 1], [2, 2]]], float)
    prim = build_primitive("select_keypoints", {"indices": [0, 2], "names": ["a", "c"]})
    out = prim.apply(_ctx(_pose(coords)))
    assert out.pose.keypoint_names == ["a", "c"]
    np.testing.assert_allclose(out.pose.coords[0], [[0, 0], [2, 2]])


def test_roi_centroid():
    coords = np.array([[[0.0, 0.0], [2.0, 4.0], [10.0, 10.0]]])  # kp0,kp1 -> roiA
    prim = build_primitive("roi_centroid", {"rois": {"A": [0, 1], "B": [2]}})
    out = prim.apply(_ctx(_pose(coords)))
    assert out.pose.keypoint_names == ["A", "B"]
    np.testing.assert_allclose(out.pose.coords[0, 0], [1.0, 2.0])   # centroid of kp0,kp1
    np.testing.assert_allclose(out.pose.coords[0, 1], [10.0, 10.0])


# ======================================================================
# Feature primitives (POSE -> SIGNALS)
# ======================================================================
def test_distance_feature_euclidean_and_vertical():
    coords = np.array([[[0.0, 0.0], [3.0, 4.0]]])  # distance 5, vertical 4
    euc = build_primitive("distance_feature",
                          {"name_out": "d", "group_a": [0], "group_b": [1]})
    out = euc.apply(_ctx(_pose(coords)))
    assert out.features.get("d")[0] == pytest.approx(5.0)
    ver = build_primitive("distance_feature",
                          {"name_out": "d", "group_a": [0], "group_b": [1], "metric": "vertical"})
    assert ver.apply(_ctx(_pose(coords))).features.get("d")[0] == pytest.approx(4.0)


def test_distance_feature_averages_groups():
    # blink-style: mean of two "top" points to mean of two "bottom" points
    coords = np.array([[[0.0, 10.0], [2.0, 10.0], [0.0, 4.0], [2.0, 4.0]]])
    prim = build_primitive("distance_feature",
                           {"name_out": "blink", "group_a": [0, 1], "group_b": [2, 3]})
    out = prim.apply(_ctx(_pose(coords)))
    # mean top (1,10), mean bottom (1,4) -> distance 6
    assert out.features.get("blink")[0] == pytest.approx(6.0)


def test_offset_feature():
    # pupil at (5,5); eye centre = mean of ring [(0,0),(10,0),(0,10),(10,10)] = (5,5)
    ring = [[0, 0], [10, 0], [0, 10], [10, 10]]
    coords = np.array([[[5.0, 5.0]] + ring])
    prim = build_primitive("offset_feature",
                           {"name_out": "pupil", "point": 0, "center": [1, 2, 3, 4]})
    out = prim.apply(_ctx(_pose(coords)))
    assert out.features.get("pupil_dx")[0] == pytest.approx(0.0)
    assert out.features.get("pupil_dy")[0] == pytest.approx(0.0)
    assert out.features.get("pupil_mag")[0] == pytest.approx(0.0)


def test_coordinate_magnitude():
    coords = np.array([[[3.0, 4.0, 0.0]]])  # norm 5
    prim = build_primitive("coordinate_magnitude", {})
    out = prim.apply(_ctx(_pose(coords)))
    assert out.features.get("kp0_mag")[0] == pytest.approx(5.0)


def test_velocity_magnitude_diff():
    # constant velocity of 1 unit/frame in x; at 60 fps -> speed 60
    coords = np.arange(5, dtype=float).reshape(5, 1, 1) * np.array([1.0, 0.0])
    prim = build_primitive("velocity_magnitude", {"method": "diff"})
    out = prim.apply(_ctx(_pose(coords, fps=60.0)))
    speed = out.features.get("kp0_speed")
    assert speed[1:] == pytest.approx(60.0)  # backward diff; first frame is 0


def test_keypoint_coordinates_flatten():
    coords = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    prim = build_primitive("keypoint_coordinates", {})
    out = prim.apply(_ctx(_pose(coords)))
    assert out.features.names == ["kp0_x", "kp0_y", "kp1_x", "kp1_y"]
    np.testing.assert_allclose(out.features.values[0], [1, 2, 3, 4])


# ======================================================================
# Signal primitives (SIGNALS -> SIGNALS)
# ======================================================================
def test_zscore():
    t = np.arange(200, dtype=float)
    x = np.sin(2 * np.pi * t / 40)  # varying -> speed varies -> nonzero std
    coords = np.stack([x, np.zeros(200)], axis=-1)[:, None, :]  # (200,1,2)
    ctx = build_primitive("velocity_magnitude", {}).apply(_ctx(_pose(coords)))
    z = build_primitive("zscore", {}).apply(ctx)
    col = z.features.values[:, 0]
    assert col.mean() == pytest.approx(0.0, abs=1e-6)
    assert col.std() == pytest.approx(1.0, abs=1e-3)


def test_kinematic_derivatives_known():
    # position = constant-velocity ramp -> velocity constant, acceleration ~0
    t = np.arange(200, dtype=float)
    coords = (t * 2.0).reshape(200, 1, 1) * np.array([1.0, 0.0])  # x = 2t
    ctx = _ctx(_pose(coords, fps=1.0))
    # turn positions into a feature via keypoint_coordinates (x only matters)
    ctx = build_primitive("keypoint_coordinates", {}).apply(ctx)
    out = build_primitive("kinematic_derivatives", {"orders": [0, 1, 2]}).apply(ctx)
    assert set(out.features.names) == {"kp0_x", "kp0_y", "kp0_x_vel", "kp0_y_vel",
                                       "kp0_x_accel", "kp0_y_accel"}
    # velocity of x=2t at fps=1 is 2 (interior), acceleration ~0
    assert np.allclose(out.features.get("kp0_x_vel")[5:-5], 2.0)
    assert np.allclose(out.features.get("kp0_x_accel")[5:-5], 0.0, atol=1e-9)


# ======================================================================
# Procrustes dual-output CHECKPOINT
# ======================================================================
def _proc_pose(seed=0):
    rng = np.random.default_rng(seed)
    # asymmetric template so each frame's alignment is unique
    template = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 3.0], [0.0, 1.0]])
    T = 20
    coords = np.empty((T, template.shape[0], 2))
    for t in range(T):
        theta = 0.05 * t  # gentle rotation ramp, stays < 1 rad
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        coords[t] = template @ R.T + rng.normal(scale=0.01, size=template.shape) + np.array([t * 0.05, 0])
    return _pose(coords), template


def test_procrustes_emit_geometry_only():
    pose, template = _proc_pose()
    prim = build_primitive("procrustes", {"template": template.tolist(), "emit": "geometry"})
    assert prim.produces == frozenset({StreamType.POSE})
    out = prim.apply(_ctx(pose))
    assert out.features is None                       # no parameter stream
    assert out.pose.provenance.stages[-1] == "procrustes"
    # geometry: aligned frames sit near the template
    np.testing.assert_allclose(out.pose.coords.mean(axis=0), template, atol=0.05)


def test_procrustes_emit_parameters_only():
    pose, template = _proc_pose()
    prim = build_primitive("procrustes",
                           {"template": template.tolist(), "emit": "parameters", "prefix": "head"})
    assert prim.produces == frozenset({StreamType.SIGNALS})
    out = prim.apply(_ctx(pose))
    # geometry stream unchanged (pose coords identical to input)
    np.testing.assert_allclose(out.pose.coords, pose.coords)
    # parameter stream present
    assert {"head_tx", "head_ty", "head_rotation", "head_scale", "head_motion_mag"} <= set(out.features.names)
    # recovered rotation should track the injected 0.1 rad/frame ramp
    rot = out.features.get("head_rotation")
    assert rot[5] < rot[15]


def test_procrustes_emit_both_streams():
    pose, template = _proc_pose()
    prim = build_primitive("procrustes",
                           {"template": template.tolist(), "emit": "both",
                            "scale": "anisotropic", "prefix": "head"})
    assert prim.produces == frozenset({StreamType.POSE, StreamType.SIGNALS})
    out = prim.apply(_ctx(pose))
    # BOTH: aligned geometry AND parameter features, simultaneously (Case 1 contract)
    assert out.pose.provenance.stages[-1] == "procrustes"
    np.testing.assert_allclose(out.pose.coords.mean(axis=0), template, atol=0.05)
    assert {"head_scale_x", "head_scale_y", "head_motion_mag"} <= set(out.features.names)


def test_procrustes_motion_magnitude_formula():
    # tx=3, ty=4, sx=1, sy=1 (identity scale) -> motion_mag = 5
    pose, template = _proc_pose()
    # translate template so a rigid fit yields a known translation
    shifted = pose.coords + np.array([0.0, 0.0])
    prim = build_primitive("procrustes",
                           {"template": template.tolist(), "emit": "parameters",
                            "scale": "anisotropic"})
    out = prim.apply(_ctx(_pose(shifted)))
    tx = out.features.get("head_tx")
    ty = out.features.get("head_ty")
    sx = out.features.get("head_scale_x")
    sy = out.features.get("head_scale_y")
    mag = out.features.get("head_motion_mag")
    expect = np.sqrt(tx**2 + ty**2 + (sx - 1) ** 2 + (sy - 1) ** 2)
    np.testing.assert_allclose(mag, expect, atol=1e-9)


# ======================================================================
# Config composition + validation
# ======================================================================
def test_pipeline_from_config_and_run():
    config = [
        {"primitive": "coordinate_normalization", "params": {"width": 720, "height": 720}},
        {"primitive": "roi_centroid", "params": {"rois": {"face": [0, 1], "arm": [2, 3]}}},
        {"primitive": "velocity_magnitude", "params": {"method": "diff"}},
        {"primitive": "zscore", "params": {}},
    ]
    pipe = FeaturePipeline.from_config(config)
    pose = _pose(np.random.default_rng(0).normal(size=(50, 4, 2)) * 100)
    ctx = pipe.run(pose)
    assert ctx.features.n_features == 2  # one signal per ROI
    assert ctx.features.provenance.stages[-1] == "zscore"


def test_validation_rejects_signals_before_they_exist():
    # z-scoring before any feature primitive -> config-time error
    config = [{"primitive": "zscore", "params": {}}]
    with pytest.raises(PipelineValidationError, match="not available yet"):
        FeaturePipeline.from_config(config)


def test_validation_rejects_kinematics_before_features():
    config = [
        {"primitive": "coordinate_normalization", "params": {"width": 1, "height": 1}},
        {"primitive": "kinematic_derivatives", "params": {}},  # needs SIGNALS
    ]
    with pytest.raises(PipelineValidationError, match="kinematic_derivatives"):
        FeaturePipeline.from_config(config)


def test_unknown_primitive_errors():
    with pytest.raises(KeyError, match="Unknown primitive"):
        FeaturePipeline.from_config([{"primitive": "does_not_exist"}])


def test_config_round_trips():
    config = [
        {"primitive": "center", "params": {"reference": "centroid"}},
        {"primitive": "coordinate_magnitude", "params": {"suffix": "mag"}},
    ]
    pipe = FeaturePipeline.from_config(config)
    rebuilt = pipe.to_config()
    assert rebuilt[0]["step"] == "center"           # describe() emits "step"
    assert rebuilt[1]["params"]["suffix"] == "mag"
    FeaturePipeline.from_config(rebuilt)  # round-trips without error


def test_config_accepts_primitive_alias():
    # "primitive" is still accepted as a backwards-compatible alias for "step"
    pipe = FeaturePipeline.from_config([{"primitive": "center", "params": {}}])
    assert pipe.steps[0].name == "center"


# ======================================================================
# The three cases as pure compositions (no case-specific branching)
# ======================================================================
def test_case2_mosaic_composition_runs():
    # No Procrustes: normalize -> ROI centroid -> velocity magnitude -> zscore
    config = [
        {"primitive": "coordinate_normalization", "params": {"width": 720, "height": 720}},
        {"primitive": "roi_centroid", "params": {"rois": {"arm": [0, 1, 2], "body": [3, 4]}}},
        {"primitive": "velocity_magnitude", "params": {"method": "diff"}},
        {"primitive": "zscore", "params": {}},
    ]
    pose = _pose(np.random.default_rng(2).normal(size=(120, 5, 2)) * 100)
    ctx = FeaturePipeline.from_config(config).run(pose)
    assert ctx.features.n_features == 2


def test_case3_mirror_composition_runs_3d():
    # 3-D: canonicalise -> procrustes geometry -> select 5 kp -> keypoint coords
    config = [
        {"primitive": "canonicalise",
         "params": {"pelvis": 0, "left_shoulder": 1, "right_shoulder": 2, "neck": 3}},
        {"primitive": "procrustes", "params": {"emit": "geometry", "scale": "none"}},
        {"primitive": "select_keypoints", "params": {"indices": [0, 1, 2, 3, 4]}},
        {"primitive": "keypoint_coordinates", "params": {}},
    ]
    pose = _pose(np.random.default_rng(3).normal(size=(90, 6, 3)))
    ctx = FeaturePipeline.from_config(config).run(pose)
    assert ctx.features.n_features == 5 * 3   # 5 keypoints x 3 dims -> multivariate stream
