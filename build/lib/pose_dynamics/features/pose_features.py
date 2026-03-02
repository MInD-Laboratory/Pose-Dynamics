# features/pose_features.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .linear_features import extract_kinematics_per_frame


# ---------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------

def to_TKD(array: np.ndarray, n_points: Optional[int] = None, dims: int = 3) -> np.ndarray:
    """
    Ensure pose data is (T, n_points, dims).

    Accepts:
      - (T, n_points * dims)
      - (T, n_points, dims)

    Parameters
    ----------
    array : np.ndarray
        Pose data.
    n_points : int, optional
        Number of keypoints. If None and array is 2D, inferred.
    dims : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    coords : np.ndarray, shape (T, n_points, dims)
    """
    arr = np.asarray(array, dtype=float)

    if arr.ndim == 3:
        return arr

    if arr.ndim != 2:
        raise ValueError("Pose array must be 2D or 3D.")

    T, F = arr.shape

    if n_points is None:
        if F % dims != 0:
            raise ValueError(f"Cannot infer n_points: F={F} not divisible by dims={dims}.")
        n_points = F // dims

    if n_points * dims != F:
        raise ValueError(f"Expected {n_points*dims} features, got {F}.")

    return arr.reshape(T, n_points, dims)


def to_TF(array: np.ndarray) -> np.ndarray:
    """
    Ensure pose data is (T, n_features) flattened.
    Accepts (T, n_points, dims) or (T, n_features).
    """
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        T, K, D = arr.shape
        return arr.reshape(T, K * D)
    raise ValueError("Pose array must be 2D or 3D.")
    

# ---------------------------------------------------------------------
# Region-level features
# ---------------------------------------------------------------------

def region_centroid(
    coords: np.ndarray,
    indices: List[int],
) -> np.ndarray:
    """
    Compute centroid trajectory for a body region.

    Parameters
    ----------
    coords : np.ndarray, shape (T, n_points, dims)
    indices : list of int
        Keypoint indices belonging to the region.

    Returns
    -------
    centroid : np.ndarray, shape (T, dims)
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 3:
        raise ValueError("coords must have shape (T, n_points, dims).")

    idx = np.asarray(indices, dtype=int)
    region = coords[:, idx, :]   # (T, n_region, dims)
    return np.nanmean(region, axis=1)  # (T, dims)


def compute_region_kinematics(
    coords: np.ndarray,
    regions: Dict[str, List[int]],
    fps: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute basic kinematics for each region: displacement, speed, acceleration magnitude.

    Parameters
    ----------
    coords : np.ndarray, shape (T, n_points, dims)
    regions : dict
        {region_name: [keypoint_indices]}
    fps : float
        Sampling rate in Hz.

    Returns
    -------
    out : dict
        {
          region_name: {
            "disp": np.ndarray shape (T-1,),
            "speed": np.ndarray shape (T-1,),
            "accel": np.ndarray shape (T-2,),
          },
          ...
        }
    """
    coords = to_TKD(coords)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, idx in regions.items():
        cen = region_centroid(coords, idx)  # (T, dims)

        kin = extract_kinematics_per_frame(
            cen.reshape(cen.shape[0], 1, cen.shape[1]),  # (T, 1, dims)
            fps=fps,
        )
        # Extract magnitudes for that single "keypoint"
        disp = np.linalg.norm(kin["displacement"][:, 0, :], axis=-1)  # (T-1,)
        speed = np.linalg.norm(kin["velocity"][:, 0, :], axis=-1)     # (T-1,)
        accel = np.linalg.norm(kin["acceleration"][:, 0, :], axis=-1) # (T-2,)

        out[name] = {
            "disp": disp,
            "speed": speed,
            "accel": accel,
        }

    return out


def summarize_region_kinematics(
    region_kin: Dict[str, Dict[str, np.ndarray]],
    prefix: str = "",
) -> Dict[str, float]:
    """
    Collapse region-level kinematics into scalar summary stats.

    For each region and each quantity (disp/speed/accel) compute mean, std, and RMS.

    Returns
    -------
    feats : dict
        Keys like:
          f"{prefix}{region}_speed_mean",
          f"{prefix}{region}_speed_std",
          f"{prefix}{region}_speed_rms", ...
    """
    feats: Dict[str, float] = {}
    for region, stats in region_kin.items():
        for qname, arr in stats.items():
            a = np.asarray(arr, dtype=float)
            finite = np.isfinite(a)
            if not finite.any():
                mean = std = rms = np.nan
            else:
                aa = a[finite]
                mean = float(aa.mean())
                std = float(aa.std(ddof=1)) if len(aa) > 1 else 0.0
                rms = float(np.sqrt(np.mean(aa ** 2)))
            base = f"{prefix}{region}_{qname}"
            feats[f"{base}_mean"] = mean
            feats[f"{base}_std"] = std
            feats[f"{base}_rms"] = rms
    return feats


# ---------------------------------------------------------------------
# Symmetry features (left/right pairs)
# ---------------------------------------------------------------------

def symmetry_metrics(
    coords: np.ndarray,
    pairs: List[Tuple[int, int]],
    fps: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute simple symmetry metrics for left/right keypoint pairs.

    For each pair (i, j) we compute:
      - mean distance between them
      - std of that distance
      - correlation of their vertical coordinates (if dims >= 2)
      - optional speed correlation if fps is provided

    Parameters
    ----------
    coords : np.ndarray, shape (T, n_points, dims)
    pairs : list of (int, int)
        Keypoint index pairs, e.g. [(left_wrist, right_wrist), ...].
    fps : float, optional
        If provided, compute speed and speed correlation.

    Returns
    -------
    feats : dict
        Keys like "sym_pair0_dist_mean", "sym_pair0_y_corr", etc.
    """
    coords = to_TKD(coords)
    T, K, D = coords.shape
    feats: Dict[str, float] = {}

    for k, (i, j) in enumerate(pairs):
        if i >= K or j >= K:
            continue

        li = coords[:, i, :]  # (T, D)
        rj = coords[:, j, :]

        # Euclidean distance over time
        dist = np.linalg.norm(li - rj, axis=-1)  # (T,)
        finite = np.isfinite(dist)
        if finite.any():
            dd = dist[finite]
            feats[f"sym_pair{k}_dist_mean"] = float(dd.mean())
            feats[f"sym_pair{k}_dist_std"] = float(dd.std(ddof=1)) if len(dd) > 1 else 0.0
        else:
            feats[f"sym_pair{k}_dist_mean"] = np.nan
            feats[f"sym_pair{k}_dist_std"] = np.nan

        # Vertical coord correlation (if D >= 2)
        if D >= 2:
            y_l = li[:, 1]
            y_r = rj[:, 1]
            mask = np.isfinite(y_l) & np.isfinite(y_r)
            if mask.sum() > 2:
                yl = y_l[mask] - y_l[mask].mean()
                yr = y_r[mask] - y_r[mask].mean()
                denom = np.sqrt((yl**2).sum() * (yr**2).sum())
                corr = float((yl @ yr) / denom) if denom > 0 else np.nan
            else:
                corr = np.nan
            feats[f"sym_pair{k}_y_corr"] = corr

        # Optional speed correlation
        if fps is not None:
            dt = 1.0 / fps
            vel_l = np.diff(li, axis=0) / dt   # (T-1, D)
            vel_r = np.diff(rj, axis=0) / dt
            speed_l = np.linalg.norm(vel_l, axis=-1)
            speed_r = np.linalg.norm(vel_r, axis=-1)
            m = np.isfinite(speed_l) & np.isfinite(speed_r)
            if m.sum() > 2:
                sl = speed_l[m] - speed_l[m].mean()
                sr = speed_r[m] - speed_r[m].mean()
                denom = np.sqrt((sl**2).sum() * (sr**2).sum())
                corr = float((sl @ sr) / denom) if denom > 0 else np.nan
            else:
                corr = np.nan
            feats[f"sym_pair{k}_speed_corr"] = corr

    return feats


# ---------------------------------------------------------------------
# Windowed pose summary
# ---------------------------------------------------------------------

def windowed_pose_features(
    coords: np.ndarray,
    fps: float,
    regions: Dict[str, List[int]],
    symmetry_pairs: Optional[List[Tuple[int, int]]] = None,
    window_size_sec: float = 10.0,
    step_size_sec: float = 5.0,
) -> pd.DataFrame:
    """
    Compute pose features per overlapping window.

    For each window:
      - region-level kinematic summaries (mean/std/RMS of disp/speed/accel)
      - symmetry metrics for left/right pairs (optional)

    Parameters
    ----------
    coords : np.ndarray, shape (T, n_points, dims) or (T, n_points*dims)
    fps : float
        Sampling rate in Hz.
    regions : dict
        {region_name: [keypoint_indices]}.
    symmetry_pairs : list of (i, j), optional
        Left/right keypoint pairs.
    window_size_sec : float
        Window length in seconds.
    step_size_sec : float
        Step between window starts in seconds.

    Returns
    -------
    df : pd.DataFrame
        One row per window, columns = feature names + ['t_start', 't_end'].
    """
    coords_TKD = to_TKD(coords)
    T = coords_TKD.shape[0]

    win = int(round(window_size_sec * fps))
    step = int(round(step_size_sec * fps))

    rows = []
    t_starts = []
    t_ends = []

    for start in range(0, T - win + 1, step):
        end = start + win
        seg = coords_TKD[start:end]

        # region kinematics
        reg_kin = compute_region_kinematics(seg, regions=regions, fps=fps)
        feat_reg = summarize_region_kinematics(reg_kin, prefix="")

        # symmetry
        feat_sym = {}
        if symmetry_pairs is not None and len(symmetry_pairs) > 0:
            feat_sym = symmetry_metrics(seg, symmetry_pairs, fps=fps)

        feats = {**feat_reg, **feat_sym}
        feats["t_start"] = start / fps
        feats["t_end"] = end / fps

        rows.append(feats)
        t_starts.append(start)
        t_ends.append(end)

    df = pd.DataFrame(rows)
    return df
