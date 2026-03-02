"""
Linear kinematic feature utilities for pose-based movement analysis.

These functions are general: they work on any pose array of shape
(T, n_keypoints, D) with D in {2, 3}. They do NOT assume any particular
dataset, skeleton, or keypoint labelling.

Typical workflow:

    coords = ...  # (T, n_keypoints, D)
    kin = extract_kinematics_per_frame(coords, fps=30.0)
    features = summarise_kinematics(kin, axes=(0, 1), prefix="global")

Where:
    - "displacement" is frame-to-frame 3D/2D step length,
    - "speed"        is |velocity| magnitude,
    - "acceleration" is |acceleration| magnitude.

"""

from __future__ import annotations
from typing import Dict, Tuple, Iterable, Optional

import numpy as np


def _validate_coords_array(coords: np.ndarray) -> Tuple[int, int, int]:
    """
    Ensure coords has shape (T, n_keypoints, D) with D in {2, 3}.
    """
    coords = np.asarray(coords)
    if coords.ndim != 3:
        raise ValueError(
            f"coords must be 3D array (T, n_keypoints, D); got shape {coords.shape}"
        )
    T, n_kp, D = coords.shape
    if D not in (2, 3):
        raise ValueError(f"Last dimension must be 2 or 3 (x,y,[z]); got D={D}")
    if T < 3:
        raise ValueError(f"Need at least 3 frames to compute velocity+accel; got T={T}")
    return T, n_kp, D


def extract_kinematics_per_frame(
    coords: np.ndarray,
    fps: float = 30.0,
    return_vectors: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute per-frame kinematic magnitudes (and optionally vectors)
    for a pose sequence.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (T, n_keypoints, D), D in {2, 3}.
    fps : float
        Sampling rate in Hz.
    return_vectors : bool, default=False
        If True, also return the full velocity and acceleration vectors.

    Returns
    -------
    kinematics : dict
        {
          "displacement": (T-1, n_keypoints)  # |Δx| between frames
          "speed":        (T-1, n_keypoints)  # |velocity|
          "acceleration": (T-2, n_keypoints)  # |acceleration|
          # optionally, if return_vectors=True:
          "vel_vec":      (T-1, n_keypoints, D),
          "acc_vec":      (T-2, n_keypoints, D),
        }
    """
    T, n_kp, D = _validate_coords_array(coords)
    dt = 1.0 / float(fps)

    # frame-to-frame displacement vectors
    disp_vec = np.diff(coords, axis=0)                    # (T-1, n_kp, D)
    disp_mag = np.linalg.norm(disp_vec, axis=-1)          # (T-1, n_kp)

    # velocities & speeds
    vel_vec = disp_vec / dt                               # (T-1, n_kp, D)
    speed = np.linalg.norm(vel_vec, axis=-1)              # (T-1, n_kp)

    # accelerations & acceleration magnitudes
    acc_vec = np.diff(vel_vec, axis=0) / dt               # (T-2, n_kp, D)
    accel = np.linalg.norm(acc_vec, axis=-1)              # (T-2, n_kp)

    out: Dict[str, np.ndarray] = {
        "displacement": disp_mag,
        "speed": speed,
        "acceleration": accel,
    }

    if return_vectors:
        out["vel_vec"] = vel_vec
        out["acc_vec"] = acc_vec

    return out


def summarise_scalar_timeseries(
    x: np.ndarray,
    axes: Tuple[int, ...] = (0, 1),
    prefix: str = "",
    stats: Iterable[str] = ("mean", "std", "min", "max", "rms"),
) -> Dict[str, float]:
    """
    Compute simple summary statistics over a scalar-valued time series.

    Parameters
    ----------
    x : np.ndarray
        Array of scalar values (any shape).
    axes : tuple of int
        Axes over which to aggregate. For example:
          - (0, 1) to collapse time and keypoints,
          - (0,)   to collapse time only (keep per-keypoint).
    prefix : str
        Prefix for feature names, e.g. "speed_global" or "left_wrist_speed".
    stats : iterable of str
        Which statistics to compute: any subset of {"mean","std","min","max"}.

    Returns
    -------
    features : dict
        e.g. {"speed_global_mean": ..., "speed_global_std": ...}
    """
    x = np.asarray(x)
    features: Dict[str, float] = {}

    if "mean" in stats:
        val = np.nanmean(x, axis=axes)
        features[f"{prefix}_mean"] = float(val)
    if "std" in stats:
        val = np.nanstd(x, axis=axes)
        features[f"{prefix}_std"] = float(val)
    if "min" in stats:
        val = np.nanmin(x, axis=axes)
        features[f"{prefix}_min"] = float(val)
    if "max" in stats:
        val = np.nanmax(x, axis=axes)
        features[f"{prefix}_max"] = float(val)
    if "rms" in stats:
        val = np.sqrt(np.nanmean(np.square(x), axis=axes))
        features[f"{prefix}_rms"] = float(val)

    return features


def summarise_kinematics(
    kinematics: Dict[str, np.ndarray],
    axes: Tuple[int, ...] = (0, 1),
    prefix: str = "",
    stats: Iterable[str] = ("mean", "std", "min", "max"),
) -> Dict[str, float]:
    """
    Convenience wrapper: summarise displacement, speed, acceleration
    magnitudes in one shot.

    Parameters
    ----------
    kinematics : dict
        Output from `extract_kinematics_per_frame`.
    axes : tuple of int
        Axes over which to aggregate (see `summarise_scalar_timeseries`).
    prefix : str
        Common prefix, e.g. "global" → global_speed_mean, etc.
    stats : iterable of str
        Stats to compute.

    Returns
    -------
    features : dict
        e.g. {
          "global_displacement_mean": ...,
          "global_speed_mean": ...,
          "global_acceleration_max": ...,
          ...
        }
    """
    features: Dict[str, float] = {}
    for name in ("displacement", "speed", "acceleration"):
        if name not in kinematics:
            continue
        key_prefix = f"{prefix}_{name}" if prefix else name
        feats = summarise_scalar_timeseries(
            kinematics[name],
            axes=axes,
            prefix=key_prefix,
            stats=stats,
        )
        features.update(feats)
    return features
