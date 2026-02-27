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


def compute_roi_timeseries_for_trial(
    df_trial: pd.DataFrame,
    time_col: str,
    regions: List[Dict],
    derivatives: List[str] = ["velocity"],
) -> pd.DataFrame:
    """
    Compute ROI centroid velocity magnitude time series for an entire trial.
    
    This produces the raw time series (not summary stats) that can be saved
    for downstream use in RQA/CRQA.
    
    Args:
        df_trial: Trial data with columns [time_col, keypoint, x, y, ...]
        time_col: Name of time column ('time' or 'frame')
        regions: List of ROI definitions, each with 'name' and 'keypoints'
        derivatives: Which derivatives to compute ['velocity', 'acceleration']
        
    Returns:
        DataFrame with columns: [time_col, {roi_name}_vel_mag, ...]
        One row per time point (N-1 for velocity due to diff).
    """
    if df_trial.empty:
        return pd.DataFrame()
    
    # Get unique sorted times
    times = np.sort(df_trial[time_col].dropna().unique())
    if len(times) < 2:
        return pd.DataFrame()
    
    # Estimate dt
    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0:
        return pd.DataFrame()
    
    # Pivot once for efficiency
    dims = ["x", "y"]
    df_pivot = df_trial.pivot_table(
        index=time_col,
        columns="keypoint",
        values=dims,
        aggfunc="mean"
    )
    
    if df_pivot.empty:
        return pd.DataFrame()
    
    # Output times are N-1 due to velocity diff (use midpoints or end points)
    # Using end time points (after the diff)
    out_times = df_pivot.index.to_numpy()[1:]
    
    result_data = {time_col: out_times}
    
    for region in regions:
        roi_name = region["name"] if isinstance(region, dict) else region.name
        keypoints = region["keypoints"] if isinstance(region, dict) else region.keypoints
        
        # Compute centroid for this ROI
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        
        for kp in keypoints:
            if ("x", kp) in df_pivot.columns and ("y", kp) in df_pivot.columns:
                xs.append(df_pivot[("x", kp)].to_numpy())
                ys.append(df_pivot[("y", kp)].to_numpy())
        
        if not xs or not ys:
            continue
        
        # Centroid = mean across keypoints
        centroid_x = np.nanmean(np.vstack(xs), axis=0)
        centroid_y = np.nanmean(np.vstack(ys), axis=0)
        
        if "velocity" in derivatives:
            vx = np.diff(centroid_x) / dt
            vy = np.diff(centroid_y) / dt
            vel_mag = np.sqrt(vx**2 + vy**2)
            result_data[f"{roi_name}_vel_mag"] = vel_mag
            result_data[f"{roi_name}_vel_x"] = vx
            result_data[f"{roi_name}_vel_y"] = vy
        
        if "acceleration" in derivatives:
            vx = np.diff(centroid_x) / dt
            vy = np.diff(centroid_y) / dt
            if vx.size > 0:
                ax = np.diff(vx) / dt
                ay = np.diff(vy) / dt
                acc_mag = np.sqrt(ax**2 + ay**2)
                # Acceleration has N-2 points; pad with NaN to align
                result_data[f"{roi_name}_acc_mag"] = np.concatenate([[np.nan], acc_mag])
                result_data[f"{roi_name}_acc_x"] = np.concatenate([[np.nan], ax])
                result_data[f"{roi_name}_acc_y"] = np.concatenate([[np.nan], ay])
    
    return pd.DataFrame(result_data)