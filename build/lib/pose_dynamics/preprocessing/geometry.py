"""
Geometry utilities for pose data.

Pure numeric operations on 2D/3D landmarks:
- centering and coordinate transforms
- rigid / similarity Procrustes alignment
- framewise canonicalisation of pose sequences
"""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Basic coordinate utilities
# ----------------------------------------------------------------------

def compute_centroid(points: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute centroid of landmark set.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (..., n_points, dim) or (n_points, dim).
    axis : int
        Axis containing the point index.

    Returns
    -------
    np.ndarray
        Centroid array of shape (..., dim) with axis removed.
    """
    return np.nanmean(points, axis=axis)


def center_points(
    points: np.ndarray,
    ref_idx: int | None = None,
    axis_points: int = -2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center landmarks either on a reference point or on their centroid.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (..., n_points, dim).
    ref_idx : int or None
        Index of reference landmark. If None, use centroid of all points.
    axis_points : int
        Axis that indexes landmarks.

    Returns
    -------
    centered : np.ndarray
        Centered points, same shape as input.
    center : np.ndarray
        Reference point / centroid that was subtracted.
    """
    pts = np.asarray(points, dtype=float)

    if ref_idx is not None:
        # Take that specific keypoint as reference
        center = np.take(pts, indices=ref_idx, axis=axis_points)
    else:
        center = compute_centroid(pts, axis=axis_points)

    centered = pts - np.expand_dims(center, axis=axis_points)
    return centered, center


# ----------------------------------------------------------------------
# Core Procrustes / similarity transform
# ----------------------------------------------------------------------

def procrustes_align(
    X: np.ndarray,
    Y: np.ndarray,
    allow_translation: bool = True,
    allow_rotation: bool = True,
    allow_scale: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Align X to Y using a configurable Procrustes / similarity transform.

    Parameters
    ----------
    X, Y : np.ndarray
        Arrays of shape (n_points, dim). Must have same shape.
    allow_translation : bool
        If True, remove mean of X and Y and estimate translation.
        If False, no centering; translation is zero.
    allow_rotation : bool
        If True, estimate optimal rotation (Kabsch).
        If False, rotation = identity.
    allow_scale : bool
        If True, estimate global scalar scale.
        If False, scale = 1.

    Returns
    -------
    X_aligned : np.ndarray
        Transformed X in the coordinate frame of Y.
    params : dict
        Dict with keys:
        - 'R'   : rotation matrix (dim x dim)
        - 't'   : translation vector (dim,)
        - 's'   : scalar scale
        - 'fit' : sum of squared error after alignment
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: X {X.shape} vs Y {Y.shape}")

    n_points, dim = X.shape

    # --- translation (centering) ---
    if allow_translation:
        mu_X = X.mean(axis=0)
        mu_Y = Y.mean(axis=0)
        X0 = X - mu_X
        Y0 = Y - mu_Y
    else:
        mu_X = np.zeros(dim)
        mu_Y = np.zeros(dim)
        X0 = X.copy()
        Y0 = Y.copy()

    # --- rotation ---
    if allow_rotation:
        # Kabsch algorithm: R = argmin || Y0 - X0 R ||
        H = X0.T @ Y0
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Fix improper rotation (reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
    else:
        R = np.eye(dim)

    # --- scale ---
    if allow_scale:
        num = np.trace((R @ X0.T @ Y0))
        den = np.sum(X0 ** 2)
        s = float(num / (den + 1e-12))
    else:
        s = 1.0

    # --- translation to Y frame ---
    # Y ≈ s X R + t
    t = mu_Y - s * (mu_X @ R)

    X_aligned = s * (X @ R) + t

    fit = np.sum((Y - X_aligned) ** 2)

    params = {
        "R": R,
        "t": t,
        "s": s,
        "fit": float(fit),
    }
    return X_aligned, params


# ----------------------------------------------------------------------
# Sequence-level canonicalisation
# ----------------------------------------------------------------------

def procrustes_align_sequence(
    seq: np.ndarray,
    template: np.ndarray,
    allow_translation: bool = True,
    allow_rotation: bool = True,
    allow_scale: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Align a pose sequence to a template frame-by-frame.

    Parameters
    ----------
    seq : np.ndarray
        Input sequence, shape (T, n_points, dim).
    template : np.ndarray
        Template pose:
        - shape (n_points, dim): same for all frames (broadcast), or
        - shape (T, n_points, dim): one template per frame.
    allow_translation, allow_rotation, allow_scale : bool
        Passed through to `procrustes_align`.

    Returns
    -------
    aligned_seq : np.ndarray
        Sequence of shape (T, n_points, dim) aligned to template.
    info : dict
        Alignment info with keys:
        - 'params_per_frame': list of Procrustes parameter dicts
        - 'mean_fit': mean squared error across frames
    """
    seq = np.asarray(seq, dtype=float)

    if template.ndim == 2:
        # Broadcast single template to all frames
        template = np.broadcast_to(template, seq.shape)
    elif template.shape != seq.shape:
        raise ValueError(f"Template shape {template.shape} incompatible with seq {seq.shape}")

    T, n_points, dim = seq.shape
    aligned = np.zeros_like(seq)
    params_list = []
    total_fit = 0.0

    for t_idx in range(T):
        X_t = seq[t_idx]
        Y_t = template[t_idx]
        X_aligned_t, params_t = procrustes_align(
            X_t,
            Y_t,
            allow_translation=allow_translation,
            allow_rotation=allow_rotation,
            allow_scale=allow_scale,
        )
        aligned[t_idx] = X_aligned_t
        params_list.append(params_t)
        total_fit += params_t["fit"]

    mean_fit = total_fit / T if T > 0 else np.nan

    info = {
        "params_per_frame": params_list,
        "mean_fit": float(mean_fit),
    }
    return aligned, info


def build_template_from_sequences(
    sequences: list[np.ndarray],
    n_points: int,
) -> np.ndarray:
    """
    Build a canonical template by averaging centered sequences.

    Parameters
    ----------
    sequences : list of np.ndarray
        Each of shape (T, n_points * dim) or (T, n_points, dim).
        Assumed to already be roughly centered in space.
    n_points : int
        Number of landmarks.

    Returns
    -------
    template : np.ndarray
        Template pose sequence of shape (T, n_points, dim),
        where T is the minimum length across sequences.
    """
    # Convert all to (T, n_points, dim) and trim to common length
    seq_proc = []
    lengths = []

    for arr in sequences:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2:  # (T, n_points*dim)
            T, D = arr.shape
            dim = D // n_points
            seq_reshaped = arr.reshape(T, n_points, dim)
        elif arr.ndim == 3:
            T, n_pts_arr, dim = arr.shape
            if n_pts_arr != n_points:
                raise ValueError("Inconsistent n_points across sequences")
            seq_reshaped = arr
        else:
            raise ValueError("Each sequence must be (T, n_points*dim) or (T, n_points, dim)")

        seq_proc.append(seq_reshaped)
        lengths.append(seq_reshaped.shape[0])

    T_common = min(lengths)
    seq_trimmed = [s[:T_common] for s in seq_proc]

    # Simple average across sequences
    template = np.mean(np.stack(seq_trimmed, axis=0), axis=0)  # (T, n_points, dim)
    return template
