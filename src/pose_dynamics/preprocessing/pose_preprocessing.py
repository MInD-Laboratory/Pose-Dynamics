"""
Pose preprocessing utilities.

Bridges raw pose DataFrames and numeric pose arrays by:
- handling XYZ column layouts
- converting between DataFrames and (T, n_points, 3) arrays
- running generic preprocessing pipelines (interp, filter, centering)
- providing Procrustes-based alignment helpers
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd

from . import signal_cleaning
from . import geometry


# -----------------------------------------------------------------------------
# Column layout & conversions
# -----------------------------------------------------------------------------

import re
from collections import defaultdict

def order_xyz_triplets(columns):
    """
    Return a flat, ordered list of XYZ columns.

    Supports:
      - 'kp_00_x', 'kp_00_y', 'kp_00_z' style
      - 'x0', 'y0', 'z0' style (Mirror Game)
    """
    cols = list(columns)
    triplets = []

    # ------------------------------------------------------------------
    # Pattern 1: kp_XX_{x,y,z}
    # ------------------------------------------------------------------
    kp_pattern = re.compile(r"^kp_(\d+)_([xyz])$")
    kp_groups = defaultdict(dict)

    for c in cols:
        m = kp_pattern.match(c)
        if m:
            idx, coord = m.groups()
            kp_groups[idx][coord] = c

    for idx in sorted(kp_groups, key=lambda i: int(i)):
        g = kp_groups[idx]
        if all(k in g for k in ("x", "y", "z")):
            triplets.extend([g["x"], g["y"], g["z"]])

    # ------------------------------------------------------------------
    # Pattern 2: xN, yN, zN  (Mirror Game)
    # ------------------------------------------------------------------
    xyz_pattern = re.compile(r"^([xyz])(\d+)$")
    xyz_groups = defaultdict(dict)

    for c in cols:
        m = xyz_pattern.match(c)
        if m:
            coord, idx = m.groups()
            xyz_groups[idx][coord] = c

    for idx in sorted(xyz_groups, key=lambda i: int(i)):
        g = xyz_groups[idx]
        if all(k in g for k in ("x", "y", "z")):
            triplets.extend([g["x"], g["y"], g["z"]])

    return triplets


def df_to_pose_array(
    df: pd.DataFrame,
    xyz_cols: Sequence[str],
    dim: int = 3,
) -> np.ndarray:
    """
    Convert DataFrame columns to a (T, n_points, dim) pose array.

    Parameters
    ----------
    df : DataFrame
        Input data.
    xyz_cols : sequence of str
        Columns in the order [x0, y0, z0, x1, y1, z1, ...].
    dim : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    arr : np.ndarray
        Pose array of shape (T, n_points, dim).
    """
    X = df.loc[:, xyz_cols].to_numpy(dtype=float)
    T, D = X.shape
    if D % dim != 0:
        raise ValueError(f"Columns ({D}) not divisible by dim={dim}")
    n_points = D // dim
    return X.reshape(T, n_points, dim)


def pose_array_to_df(
    arr: np.ndarray,
    xyz_cols: Sequence[str],
    index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """
    Convert (T, n_points, dim) pose array back to a DataFrame.
    """
    T, n_points, dim = arr.shape
    if len(xyz_cols) != n_points * dim:
        raise ValueError("xyz_cols length does not match array shape")
    flat = arr.reshape(T, n_points * dim)
    return pd.DataFrame(flat, columns=list(xyz_cols), index=index)


# -----------------------------------------------------------------------------
# High-level preprocessing pipeline
# -----------------------------------------------------------------------------

def compute_interpolation_limit(m: int, tau: int) -> int:
    """
    Maximum safe interpolation run length for RQA embedding:
    L_max = (m - 1) * tau
    """
    if m < 2:
        raise ValueError("Embedding dimension m must be >= 2")
    return (m - 1) * tau


def preprocess_pose_dataframe(
    df: pd.DataFrame,
    xyz_cols: Sequence[str],
    fps: float,
    center_ref_idx: Optional[int] = None,
    # interpolation control
    interpolate_max_run: Optional[int] = 60,
    embedding_m: Optional[int] = None,
    embedding_tau: Optional[int] = None,
    # filtering control
    butterworth_cutoff: Optional[float] = 10.0,
    butterworth_order: int = 4,
) -> np.ndarray:
    """
    Generic preprocessing pipeline for a single pose trial.

    Steps:
        1) (optional) interpolate short NaN runs per column
           - if embedding_m and embedding_tau are provided, the maximum
             interpolated run is capped at L_max = (m - 1) * tau
        2) (optional) Butterworth low-pass filter per column
        3) convert to (T, n_points, 3)
        4) center by reference keypoint or centroid
    """
    work = df.copy()

    # --- decide interpolation limit ---
    if embedding_m is not None and embedding_tau is not None:
        L_max = compute_interpolation_limit(embedding_m, embedding_tau)
        # if user also passed interpolate_max_run, respect the stricter one
        if interpolate_max_run is None:
            effective_max_run = L_max
        else:
            effective_max_run = min(interpolate_max_run, L_max)
    else:
        effective_max_run = interpolate_max_run

    # 1) interpolate NaN runs
    if effective_max_run is not None:
        work = signal_cleaning.interpolate_dataframe_nan_runs(
            work, max_run=effective_max_run
        )

    # 2) Butterworth low-pass
    if butterworth_cutoff is not None:
        work = signal_cleaning.butterworth_filter_dataframe(
            work,
            fs=fps,
            cutoff_hz=butterworth_cutoff,
            order=butterworth_order,
            btype="low",
        )

    # 3) to (T, n_points, 3)
    coords = df_to_pose_array(work, xyz_cols, dim=3)

    # 4) center
    coords_centered, center = geometry.center_points(
        coords, ref_idx=center_ref_idx, axis_points=1
    )

    return coords_centered


# -----------------------------------------------------------------------------
# Procrustes-based alignment
# -----------------------------------------------------------------------------

def align_keypoints_3d(
    df: pd.DataFrame,
    expected_cols: Sequence[str],
    ref_idx: Optional[int] = None,
    template: Optional[np.ndarray] = None,
    use_procrustes: bool = False,
    allow_rotation: bool = True,
    allow_scale: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Convert a pose DataFrame to a centered / aligned flattened array.

    Parameters
    ----------
    df : DataFrame
        Pose data for one trial.
    expected_cols : sequence of str
        XYZ columns in the correct order.
    ref_idx : int or None
        Reference keypoint for centering. If None, use centroid.
    template : np.ndarray or None
        Template sequence for Procrustes alignment.
        Shape (T, n_points, 3) or (n_points, 3).
    use_procrustes : bool
        If False, do translation-only centering and ignore template.
        If True, align each frame to the template using Procrustes.
    allow_rotation : bool
        Toggle for rotation in Procrustes.
    allow_scale : bool
        Toggle for scale in Procrustes.

    Returns
    -------
    arr_flat : np.ndarray
        Flattened aligned array of shape (T, n_points*3).
    meta : dict
        Metadata including n_frames, n_points, ref_idx, and (optionally)
        Procrustes fit statistics.
    """
    # Extract coords (T, n_points, 3)
    coords = df_to_pose_array(df, expected_cols, dim=3)
    T, n_points, dim = coords.shape

    # Stage 1: center
    coords_centered, center = geometry.center_points(
        coords, ref_idx=ref_idx, axis_points=1
    )

    meta: Dict = {
        "n_frames": T,
        "n_points": n_points,
        "dim": dim,
        "ref_idx": ref_idx,
        "center_type": "keypoint" if ref_idx is not None else "centroid",
    }

    if not use_procrustes:
        arr_flat = coords_centered.reshape(T, n_points * dim)
        return arr_flat, meta

    # Stage 2: Procrustes alignment to template
    if template is None:
        raise ValueError("template must be provided when use_procrustes=True")

    aligned, info = geometry.procrustes_align_sequence(
        coords_centered,
        template=template,
        allow_translation=True,
        allow_rotation=allow_rotation,
        allow_scale=allow_scale,
    )

    meta["procrustes"] = {
        "allow_rotation": allow_rotation,
        "allow_scale": allow_scale,
        "mean_fit": info["mean_fit"],
    }

    arr_flat = aligned.reshape(T, n_points * dim)
    return arr_flat, meta


# -----------------------------------------------------------------------------
# Keypoint selection & quality control
# -----------------------------------------------------------------------------

def extract_keypoint_subset(
    aligned_array: np.ndarray,
    n_keypoints_total: Optional[int] = None,
    selected_indices: Optional[Sequence[int]] = None,
    keypoint_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Extract subset of keypoints from aligned array with quality report.

    Parameters
    ----------
    aligned_array : np.ndarray
        (T, n_points*3) flattened keypoint array.
    n_keypoints_total : int, optional
        Total number of keypoints (inferred if None).
    selected_indices : sequence of int, optional
        Keypoint indices to keep. Takes precedence over keypoint_indices.
    keypoint_indices : sequence of int, optional
        Legacy parameter name for selected_indices.

    Returns
    -------
    extracted_array : np.ndarray
        (T, n_selected*3) array with selected keypoints only.
    quality_report : dict
        Extraction statistics and NaN summary.
    """
    if selected_indices is not None:
        indices = list(selected_indices)
    elif keypoint_indices is not None:
        indices = list(keypoint_indices)
    else:
        raise ValueError("Must provide either 'selected_indices' or 'keypoint_indices'")

    n_frames = aligned_array.shape[0]

    if n_keypoints_total is None:
        n_keypoints_total = aligned_array.shape[1] // 3

    coords_3d = aligned_array.reshape(n_frames, n_keypoints_total, 3)
    coords_selected = coords_3d[:, indices, :]
    extracted_array = coords_selected.reshape(n_frames, -1)

    n_selected = len(indices)
    n_nans = int(np.isnan(extracted_array).sum())
    total_values = int(extracted_array.size)

    quality_report = {
        "status": "ok" if n_nans == 0 else "warning",
        "n_frames": n_frames,
        "n_keypoints_total": n_keypoints_total,
        "n_keypoints_selected": n_selected,
        "selected_indices": indices,
        "n_nan_values": n_nans,
        "total_values": total_values,
        "pct_nan": 100.0 * n_nans / total_values if total_values > 0 else 0.0,
        "shape_input": (n_frames, n_keypoints_total * 3),
        "shape_output": extracted_array.shape,
    }

    return extracted_array, quality_report


def check_data_quality(
    aligned_array: np.ndarray,
    trial_name: str,
    fps: float = 30.0,
    verbose: bool = True,
) -> Dict:
    """
    Run generic quality checks on aligned pose data.

    Checks:
        - NaNs
        - extreme values (>5 m from origin)
        - nearly constant keypoints
        - sudden jumps (large speeds)
        - duplicate frames
        - movement irregularity (CV of total displacement)
    """
    n_frames, n_dims = aligned_array.shape
    n_points = n_dims // 3

    coords_3d = aligned_array.reshape(n_frames, n_points, 3)

    report: Dict = {
        "trial_name": trial_name,
        "n_frames": n_frames,
        "duration_sec": n_frames / fps,
        "issues": [],
        "warnings": [],
        "passed": True,
    }

    # NaNs
    n_nans = int(np.isnan(aligned_array).sum())
    if n_nans > 0:
        pct_nan = 100 * n_nans / aligned_array.size
        report["issues"].append(f"{n_nans} NaN values ({pct_nan:.2f}%)")
        report["passed"] = False

    # Extreme values
    extreme_threshold = 5.0
    n_extreme = int(np.sum(np.abs(coords_3d) > extreme_threshold))
    if n_extreme > 0:
        pct_extreme = 100 * n_extreme / coords_3d.size
        if pct_extreme > 1.0:
            report["issues"].append(
                f"{n_extreme} extreme values >±{extreme_threshold}m ({pct_extreme:.2f}%)"
            )
            report["passed"] = False
        else:
            report["warnings"].append(
                f"{n_extreme} extreme values detected ({pct_extreme:.3f}%)"
            )

    # Frozen keypoints
    for pt_idx in range(n_points):
        pt_coords = coords_3d[:, pt_idx, :]
        pt_std = np.std(pt_coords, axis=0)
        if np.any(pt_std < 0.001):
            report["warnings"].append(
                f"Keypoint {pt_idx} nearly constant (std={pt_std.max():.6f})"
            )

    # Sudden jumps
    velocity = np.diff(coords_3d, axis=0)
    speed = np.linalg.norm(velocity, axis=2)  # (T-1, n_points)

    for pt_idx in range(n_points):
        pt_speed = speed[:, pt_idx]
        median_speed = np.median(pt_speed)
        max_speed = np.max(pt_speed)
        jump_threshold = 5 * median_speed
        n_jumps = int(np.sum(pt_speed > jump_threshold))
        if n_jumps > 0:
            report["warnings"].append(
                f"Keypoint {pt_idx}: {n_jumps} sudden jumps "
                f"(max={max_speed:.3f} m/frame, threshold={jump_threshold:.3f})"
            )

    # Duplicate frames
    frame_diffs = np.diff(coords_3d, axis=0)
    duplicate_frames = np.all(frame_diffs == 0, axis=(1, 2))
    n_duplicates = int(np.sum(duplicate_frames))
    if n_duplicates > n_frames * 0.05:
        pct_dup = 100 * n_duplicates / n_frames
        report["warnings"].append(
            f"{n_duplicates} duplicate frames ({pct_dup:.1f}%)"
        )

    # Movement irregularity
    total_disp = np.sum(speed, axis=1)
    cv = np.std(total_disp) / (np.mean(total_disp) + 1e-8)
    if cv > 2.0:
        report["warnings"].append(f"High movement irregularity (CV={cv:.2f})")

    if verbose:
        status = "✓ PASSED" if report["passed"] else "✗ FAILED"
        print(f"\n{status}: {trial_name}")
        print(
            f"  Duration: {report['duration_sec']:.1f}s "
            f"({report['n_frames']} frames)"
        )
        if report["issues"]:
            print("  Issues:")
            for issue in report["issues"]:
                print(f"    ✗ {issue}")
        if report["warnings"]:
            print("  Warnings:")
            for w in report["warnings"]:
                print(f"    ⚠ {w}")
        if not report["issues"] and not report["warnings"]:
            print("  No issues detected")

    return report
