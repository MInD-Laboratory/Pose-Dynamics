"""
Procrustes and body-frame geometry math (recovered from the prototype).

These are the numeric definitions behind the alignment primitives. Two scale
modes are supported, matching the recovered case studies:

- **uniform / rigid** (core ``geometry.procrustes_align``; Cases 2 & 3): Kabsch
  rotation, optional single scalar scale;
- **anisotropic** (MATB ``geometry_utils.procrustes_frame_to_template``; Case 1):
  a least-squares linear map polar-decomposed into a rotation and per-axis
  scales ``(sx, sy)``.

Also included: the 3-D body-frame canonicalisation used by Case 3
(``mirror_game.canonicalise_mean_pose``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TransformParams:
    """Per-frame transform: apply as ``points @ L + t``.

    ``L`` is the linear map (rotation * scale), ``t`` the translation. ``R`` and
    ``scale`` are retained so the parameter stream can expose rotation angle and
    per-axis scale as features.
    """

    L: np.ndarray          # (d, d) linear map
    t: np.ndarray          # (d,) translation
    R: np.ndarray          # (d, d) rotation
    scale: np.ndarray      # (d,) per-axis scales (uniform => all equal)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform a ``(K, d)`` (or ``(d,)``) set of points."""
        return np.asarray(points, float) @ self.L + self.t

    @property
    def rotation_angle(self) -> float:
        """2-D in-plane rotation angle (radians); safe (if less meaningful) in 3-D."""
        return float(np.arctan2(self.R[1, 0], self.R[0, 0]))


def procrustes_uniform(
    X: np.ndarray,
    Y: np.ndarray,
    allow_scale: bool = False,
) -> TransformParams:
    """Rigid/similarity Procrustes aligning X onto Y (Umeyama). Works in 2-D or 3-D.

    Minimises ``||Y - (s X R + t)||`` so ``aligned = X @ (sR) + t``. This uses the
    correct Kabsch/Umeyama rotation ``R = U D Vᵀ`` (with a reflection guard) — the
    same convention as scipy's ``orthogonal_procrustes`` that the MOSAIC prototype
    used. (The unused ``core/geometry.procrustes_align`` helper transposed this and
    did not actually align; see the recovery test.)
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    dim = X.shape[1]

    muX, muY = X.mean(axis=0), Y.mean(axis=0)
    X0, Y0 = X - muX, Y - muY

    H = X0.T @ Y0                      # (d, d)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(dim)
    D[-1, -1] = np.sign(np.linalg.det(U @ Vt))  # reflection guard
    R = U @ D @ Vt

    s = float(np.sum(S * np.diag(D)) / (np.sum(X0 ** 2) + 1e-12)) if allow_scale else 1.0

    L = s * R
    t = muY - muX @ L
    return TransformParams(L=L, t=t, R=R, scale=np.full(dim, s))


def procrustes_anisotropic(X: np.ndarray, Y: np.ndarray) -> TransformParams:
    """Anisotropic Procrustes (2-D), recovered from MATB ``geometry_utils``.

    Fits the best linear map by least squares, polar-decomposes it into a rotation
    ``R`` and per-axis scales ``(sx, sy) = diag(Rᵀ A)``, then applies
    ``R diag(sx, sy)`` about the centroids.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.shape[1] != 2:
        raise ValueError("anisotropic Procrustes is defined for 2-D landmarks only.")

    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc, Yc = X - muX, Y - muY

    A = np.linalg.lstsq(Xc, Yc, rcond=None)[0]  # minimises ||Xc A - Yc||
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    M = R.T @ A
    sx, sy = np.diag(M)
    S = np.diag([sx, sy])

    # Apply the shear-free rotation+axis-scale directly: aligned = (X-muX) @ (R S) + muY.
    # (The prototype transposed this and did not actually align; the decomposition
    # R, sx, sy is correct and is what the parameter stream uses.)
    L = R @ S
    t = muY - muX @ L
    return TransformParams(L=L, t=t, R=R, scale=np.array([sx, sy]))


def build_global_template(
    frames: list[np.ndarray] | np.ndarray,
    landmarks: list[int] | None = None,
) -> np.ndarray:
    """Build a template by averaging poses across frames/sequences.

    Accepts a list of ``(K, d)`` mean poses (or a ``(T, K, d)`` stack) and returns
    the mean ``(K, d)`` pose. If ``landmarks`` is given, only those keypoints are
    returned (the stable-landmark template of Case 1).
    """
    arr = np.asarray(frames, dtype=float)
    if arr.ndim == 3:
        template = np.nanmean(arr, axis=0)
    else:
        template = arr
    if landmarks is not None:
        template = template[np.asarray(landmarks, int)]
    return template


# ----------------------------------------------------------------------
# 3-D body-frame canonicalisation (Case 3)
# ----------------------------------------------------------------------
def body_frame_rotation(
    pose: np.ndarray,
    pelvis: int,
    left_shoulder: int,
    right_shoulder: int,
    neck: int,
) -> np.ndarray:
    """Return the ``(3,3)`` rotation into the body-fixed frame for a single pose.

    Recovered from ``mirror_game.canonicalise_mean_pose``: x = left->right
    shoulder, y = pelvis->neck (up), z = forward; re-orthogonalised and made
    right-handed with forward roughly +y in plot coordinates.
    """
    P = np.asarray(pose, float)
    P = P - P[pelvis]
    x_body = P[right_shoulder] - P[left_shoulder]
    y_body = P[neck] - P[pelvis]

    x = x_body / (np.linalg.norm(x_body) + 1e-12)
    y = y_body - np.dot(x, y_body) * x
    y = y / (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    y = np.cross(z, x)
    if np.dot(np.cross(x, y), z) < 0:
        z = -z
    if z[1] < 0:
        z = -z
        y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
