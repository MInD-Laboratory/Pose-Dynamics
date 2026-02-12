"""ROI (Region of Interest) feature extraction.

Computes centroid-based movement features for anatomically-defined regions.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _compute_centroid_series(
    df_win: pd.DataFrame,
    time_col: str,
    keypoints: List[str],
    dims: List[str] = ["x", "y"],
) -> Dict[str, np.ndarray]:
    """
    Compute centroid trajectory for a set of keypoints.
    
    Returns dict with keys like 'centroid_x', 'centroid_y' containing
    time series of the mean position across specified keypoints.
    """
    out: Dict[str, np.ndarray] = {}
    if not keypoints or df_win.empty:
        return out
    
    # Filter to only the keypoints in this ROI
    df_roi = df_win[df_win["keypoint"].isin(keypoints)]
    if df_roi.empty:
        return out
    
    # Pivot to wide format: index=time, columns=(dim, keypoint)
    pivot = df_roi.pivot_table(
        index=time_col,
        columns="keypoint",
        values=dims,
        aggfunc="mean"
    )
    
    if pivot.empty:
        return out
    
    # Compute centroid for each dimension
    for dim in dims:
        if dim not in pivot.columns.levels[0]:
            continue
        
        # Get all keypoint columns for this dimension
        kp_cols = [kp for kp in keypoints if (dim, kp) in pivot.columns]
        if not kp_cols:
            continue
        
        # Stack the keypoint data and compute mean across keypoints
        kp_data = np.stack([pivot[(dim, kp)].to_numpy() for kp in kp_cols], axis=1)
        centroid = np.nanmean(kp_data, axis=1)
        
        out[f"centroid_{dim}"] = centroid
    
    return out


def _compute_velocity_magnitude(
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    dt: float,
) -> np.ndarray | None:
    """
    Compute velocity magnitude from centroid x, y trajectories.
    
    Returns 1D array of velocity magnitudes (one per frame).
    """
    if centroid_x.size == 0 or centroid_y.size == 0:
        return None
    if not np.isfinite(dt) or dt <= 0:
        return None
    
    # Compute velocity as first derivative
    vx = np.diff(centroid_x) / dt
    vy = np.diff(centroid_y) / dt
    
    # Magnitude
    vel_mag = np.sqrt(vx**2 + vy**2)
    
    return vel_mag


def roi_feature_series(
    df_win: pd.DataFrame,
    time_col: str,
    roi_name: str,
    keypoints: List[str],
    derivatives: List[str],
    dt: float,
) -> Dict[str, np.ndarray]:
    """
    Extract feature time series for a single ROI.
    
    Args:
        df_win: Window data with columns [time_col, keypoint, x, y, ...]
        time_col: Name of time column
        roi_name: Name of the ROI (e.g., 'centre_face')
        keypoints: List of keypoint names belonging to this ROI
        derivatives: Which derivatives to compute ['velocity', 'acceleration']
        dt: Time step between frames
        
    Returns:
        Dict mapping feature names to time series arrays.
        Keys follow pattern: '{roi_name}_vel_mag', '{roi_name}_vel_x', etc.
    """
    out: Dict[str, np.ndarray] = {}
    
    # Compute centroid trajectories
    centroid = _compute_centroid_series(df_win, time_col, keypoints)
    if not centroid:
        return out
    
    # Extract x, y centroids
    cx = centroid.get("centroid_x")
    cy = centroid.get("centroid_y")
    
    if cx is None or cy is None:
        return out
    
    # Compute velocity if requested
    if "velocity" in derivatives:
        vel_mag = _compute_velocity_magnitude(cx, cy, dt)
        if vel_mag is not None:
            out[f"{roi_name}_vel_mag"] = vel_mag
            
            # Also store component velocities
            if np.isfinite(dt) and dt > 0:
                vx = np.diff(cx) / dt
                vy = np.diff(cy) / dt
                out[f"{roi_name}_vel_x"] = vx
                out[f"{roi_name}_vel_y"] = vy
    
    # Compute acceleration if requested
    if "acceleration" in derivatives:
        if np.isfinite(dt) and dt > 0:
            vx = np.diff(cx) / dt
            vy = np.diff(cy) / dt
            
            if vx.size > 0 and vy.size > 0:
                ax = np.diff(vx) / dt
                ay = np.diff(vy) / dt
                acc_mag = np.sqrt(ax**2 + ay**2)
                out[f"{roi_name}_acc_mag"] = acc_mag
                out[f"{roi_name}_acc_x"] = ax
                out[f"{roi_name}_acc_y"] = ay
    
    return out