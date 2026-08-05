"""Case 1 (MATB) feature extraction, ported faithfully from the parent analysis.

This module reproduces ``Pose/utils/features_utils.procrustes_features_for_file``
and ``Pose/utils/geometry_utils.procrustes_frame_to_template`` from the
Measuring_Workload_Dynamics_in_OpenMATB repository, which produced the published
Case 1 results.

It exists as a separate module rather than as a ``FeaturePipeline`` config
because four details of the original do not correspond to the generic
primitives, and each one materially changes the resulting signals:

1. **The Procrustes fit uses all 23 "relevant" landmarks**, not the four stable
   reference points. Fitting on four landmarks yields a visibly different
   transform.
2. **The transform is a least-squares linear map followed by polar
   decomposition** (``A = lstsq(Xc, Yc)``, ``A = R M``, scales from
   ``diag(R^T A)``), not a classical orthogonal-Procrustes SVD fit.
3. **The reported translation is the affine offset of the composed transform**,
   ``t = mu_Y - R S mu_X``. This moves *opposite* to the head: as the face
   translates right, ``t_x`` decreases. Treating it as head displacement
   inverts the signal.
4. **The eye contour rings are 5 points, not 6.** Landmarks 40 and 43 (1-based)
   are absent from the relevant set, so the eye centres from which pupil offsets
   are measured are means of five points.

All downstream features (blink, mouth, pupil offsets) are computed on the
*aligned* coordinates ``Xtrans``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# --- landmark groups, 1-based as in the parent repo's config ---------------
PROCRUSTES_REF = (30, 31, 37, 46)
BLINK_L_TOP, BLINK_L_BOT = (38, 39), (41, 42)
BLINK_R_TOP, BLINK_R_BOT = (44, 45), (47, 48)
HEAD_ROT = (37, 46)
MOUTH = (63, 67)
CENTER_FACE = tuple(range(28, 37))
PUPILS = (69, 70)
LEFT_EYE_RING = (37, 38, 39, 40, 41, 42)
RIGHT_EYE_RING = (43, 44, 45, 46, 47, 48)


def relevant_indices() -> list[int]:
    """The 23 landmarks the parent pipeline retains (1-based, sorted)."""
    s: set[int] = set()
    for grp in (PROCRUSTES_REF, HEAD_ROT, MOUTH, CENTER_FACE,
                BLINK_L_TOP, BLINK_L_BOT, BLINK_R_TOP, BLINK_R_BOT, PUPILS):
        s.update(grp)
    return sorted(s)


REL_IDXS = relevant_indices()


def load_global_template(path: str | Path) -> np.ndarray:
    """Load ``global_template.csv`` as an ``(n_rel, 2)`` array ordered by REL_IDXS."""
    df = pd.read_csv(path)
    return np.column_stack([
        [float(df[f"x{i}"].values[0]) for i in REL_IDXS],
        [float(df[f"y{i}"].values[0]) for i in REL_IDXS],
    ])


def procrustes_frame_to_template(frame_xy, templ_xy, available):
    """Port of ``geometry_utils.procrustes_frame_to_template``.

    Returns ``(ok, sx, sy, tx, ty, R, Xtrans)``.
    """
    idx = np.where(available)[0]
    if idx.size < 3:
        nan2 = np.full((2, 2), np.nan)
        return False, np.nan, np.nan, np.nan, np.nan, nan2, np.full_like(frame_xy, np.nan)

    X, Y = frame_xy[idx, :], templ_xy[idx, :]
    muX, muY = X.mean(axis=0, keepdims=True), Y.mean(axis=0, keepdims=True)
    Xc, Yc = X - muX, Y - muY

    A = np.linalg.lstsq(Xc, Yc, rcond=None)[0]      # best linear map
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt                                       # polar: rotation part
    M = R.T @ A
    sx, sy = np.diag(M)                              # anisotropic scales
    S = np.diag([sx, sy])

    t = muY.T - R @ S @ muX.T                        # affine offset of R S x + t
    Xtrans = (R @ S @ (frame_xy - muX).T).T + muY
    return True, float(sx), float(sy), float(t[0]), float(t[1]), R, Xtrans


def _pos(lmk: int) -> int:
    """Row of a 1-based landmark within REL_IDXS, or -1 if absent."""
    return REL_IDXS.index(lmk) if lmk in REL_IDXS else -1


def matb_features(coords: np.ndarray, template: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the Case 1 per-frame features from screen-normalized coordinates.

    ``coords`` is ``(T, 70, 2)`` in unit screen coordinates (the full OpenPose
    face model); it is reduced to REL_IDXS internally. ``template`` is the
    ``(23, 2)`` global template from :func:`load_global_template`.
    """
    rel0 = [i - 1 for i in REL_IDXS]                 # to 0-based keypoint rows
    XY = coords[:, rel0, :]
    n = XY.shape[0]

    out = {k: np.full(n, np.nan, float) for k in (
        "head_rotation", "head_tx", "head_ty", "head_scale_x", "head_scale_y",
        "head_motion_mag", "blink_aperture", "mouth_aperture",
        "pupil_metric_dx", "pupil_metric_dy", "pupil_metric_mag")}

    Ltop = [_pos(i) for i in BLINK_L_TOP]
    Lbot = [_pos(i) for i in BLINK_L_BOT]
    Rtop = [_pos(i) for i in BLINK_R_TOP]
    Rbot = [_pos(i) for i in BLINK_R_BOT]
    l_ring = [_pos(i) for i in LEFT_EYE_RING if _pos(i) >= 0]
    r_ring = [_pos(i) for i in RIGHT_EYE_RING if _pos(i) >= 0]
    m63, m67 = _pos(MOUTH[0]), _pos(MOUTH[1])
    i69, i70 = _pos(PUPILS[0]), _pos(PUPILS[1])

    tmpl_ok = np.isfinite(template).all(axis=1)

    for t in range(n):
        frame = XY[t]
        avail = np.isfinite(frame).all(axis=1) & tmpl_ok
        ok, sx, sy, tx, ty, R, Xt = procrustes_frame_to_template(frame, template, avail)
        if not ok:
            continue

        out["head_scale_x"][t] = sx
        out["head_scale_y"][t] = sy
        out["head_tx"][t] = tx
        out["head_ty"][t] = ty
        out["head_motion_mag"][t] = math.sqrt(tx * tx + ty * ty + (sx - 1.0) ** 2 + (sy - 1.0) ** 2)
        if R is not None and R.shape == (2, 2):
            out["head_rotation"][t] = math.atan2(R[1, 0], R[0, 0])

        def pair(idxs):
            pts = [Xt[i] for i in idxs if i >= 0 and np.isfinite(Xt[i]).all()]
            return np.vstack(pts) if len(pts) == 2 else None

        vals = []
        a, b = pair(Ltop), pair(Lbot)
        if a is not None and b is not None:
            vals.append(float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))))
        a, b = pair(Rtop), pair(Rbot)
        if a is not None and b is not None:
            vals.append(float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))))
        if vals:
            out["blink_aperture"][t] = float(np.mean(vals))

        if m63 >= 0 and m67 >= 0 and np.isfinite(Xt[m63]).all() and np.isfinite(Xt[m67]).all():
            out["mouth_aperture"][t] = float(np.linalg.norm(Xt[m67] - Xt[m63]))

        def centre(idxs):
            pts = [Xt[i] for i in idxs if i >= 0 and np.isfinite(Xt[i]).all()]
            return np.vstack(pts).mean(axis=0) if len(pts) >= 3 else None

        cL, cR = centre(l_ring), centre(r_ring)
        offs, mags = [], []
        if i69 >= 0 and cL is not None and np.isfinite(Xt[i69]).all():
            d = Xt[i69] - cL
            offs.append(d); mags.append(float(np.linalg.norm(d)))
        if i70 >= 0 and cR is not None and np.isfinite(Xt[i70]).all():
            d = Xt[i70] - cR
            offs.append(d); mags.append(float(np.linalg.norm(d)))
        if offs:
            out["pupil_metric_dx"][t] = float(np.mean([o[0] for o in offs]))
            out["pupil_metric_dy"][t] = float(np.mean([o[1] for o in offs]))
            out["pupil_metric_mag"][t] = float(np.mean(mags))

    out["head_translation_mag"] = np.hypot(out["head_tx"], out["head_ty"])
    return out
