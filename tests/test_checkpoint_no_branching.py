"""
Build-order step 1 checkpoint.

The prototype's central structural flaw was separate code paths for 2D and 3D
data. This test is the acceptance gate for the data model: a 2D face, a 2D upper
body, and a 3D full body must ALL load into the same :class:`PoseSequence` class
and be operated on by the same code, with no branching on dimensionality.

If this passes, the flaw is fixed at the data-model level.
"""
from __future__ import annotations

import numpy as np

from pose_dynamics.data import Dyad, PoseSequence, load_pose_csv


def _identical_pipeline(seq: PoseSequence) -> PoseSequence:
    """A per-person operation written once, with no 2D/3D branch.

    It computes the per-frame centroid over keypoints and re-centers the pose,
    relying entirely on ``dims`` being data (last-axis broadcasting). The same
    line works for dims == 2 and dims == 3.
    """
    centroid = np.nanmean(seq.coords, axis=1, keepdims=True)  # (T, 1, dims)
    return seq.with_stage("center", {"ref": "centroid"}, coords=seq.coords - centroid)


def test_all_three_modalities_share_one_class(face2d_csv, body2d_csv, body3d_csv):
    face = load_pose_csv(face2d_csv, frame_rate=60.0)     # Case 1: 2D face, conf
    body = load_pose_csv(body2d_csv, frame_rate=60.0)     # Case 2: 2D upper body, conf
    full = load_pose_csv(body3d_csv, frame_rate=30.0)     # Case 3: 3D full body, no conf

    # Same concrete type for every modality.
    assert type(face) is type(body) is type(full) is PoseSequence

    # Dimensionality is data, not a subtype.
    assert (face.dims, body.dims, full.dims) == (2, 2, 3)
    assert (face.n_keypoints, body.n_keypoints, full.n_keypoints) == (70, 25, 38)
    assert face.has_confidence and body.has_confidence and not full.has_confidence

    # The exact same function processes all three with no branching.
    for seq in (face, body, full):
        out = _identical_pipeline(seq)
        assert out.dims == seq.dims
        assert out.coords.shape == seq.coords.shape
        # centroid of a centered pose is ~0 on every axis, any dims
        recentered = np.nanmean(out.coords, axis=1)
        np.testing.assert_allclose(recentered, 0.0, atol=1e-9)
        assert out.provenance.stages == ["load", "center"]


def test_dyads_work_for_2d_and_3d(body2d_csv, body3d_csv):
    # 2D dyad (Case 2 style) and 3D dyad (Case 3 style) use the identical container.
    b1 = load_pose_csv(body2d_csv, frame_rate=60.0)
    b2 = load_pose_csv(body2d_csv, frame_rate=60.0)
    dyad2d = Dyad(a=b1, b=b2, dyad_id="mosaic")

    f1 = load_pose_csv(body3d_csv, frame_rate=30.0)
    f2 = load_pose_csv(body3d_csv, frame_rate=30.0)
    dyad3d = Dyad(a=f1, b=f2, dyad_id="mirror")

    assert type(dyad2d) is type(dyad3d) is Dyad
    assert dyad2d.dims == 2 and dyad3d.dims == 3

    # Per-person mapping is one code path for both.
    processed = dyad3d.map(_identical_pipeline)
    assert processed.a.provenance.stages == ["load", "center"]
    assert processed.b.dims == 3
