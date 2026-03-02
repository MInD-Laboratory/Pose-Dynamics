"""
MOSAIC Features Module

Custom feature extraction and velocity computation specific to MOSAIC dataset.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

# ============================================================================
# VELOCITY COMPUTATION
# ============================================================================

def compute_velocity(df: pd.DataFrame, fps: float = 60) -> pd.DataFrame:
    """
    Compute velocity (dx/dt, dy/dt) for each keypoint column.
    
    Uses finite differences to approximate first derivative.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with keypoint position columns
    fps : float
        Sampling rate in Hz (default: 60)
        
    Returns
    -------
    pd.DataFrame
        Velocity DataFrame with columns renamed (_vx, _vy)
        
    Notes
    -----
    First frame will have zero velocity (forward difference).
    Column names are transformed: "_x_offset" → "_vx", "_y_offset" → "_vy"
    """
    df_vel = df.diff().fillna(0) * fps
    df_vel.columns = [
        col.replace("_x_offset", "_vx").replace("_y_offset", "_vy") 
        for col in df.columns
    ]
    return df_vel


# ============================================================================
# CUSTOM FEATURES (BLINK AND LIP DISTANCE)
# ============================================================================

def add_custom_features(
    df_raw: pd.DataFrame,
    df_extracted: pd.DataFrame,
    calculated_metrics: List[str] = None
) -> pd.DataFrame:
    """
    Compute custom facial features (blink distance, lip distance).
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw pose data (not currently used, kept for compatibility)
    df_extracted : pd.DataFrame
        Extracted keypoint data with _x_offset, _y_offset columns
    calculated_metrics : list of str
        Features to compute. Options: ['blink_dist', 'lip_dist']
        Default: ['blink_dist', 'lip_dist']
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added feature columns
        
    Notes
    -----
    - blink_dist: Average vertical distance between upper/lower eyelids (both eyes)
    - lip_dist: Vertical distance between upper and lower lip
    
    Missing keypoints will result in NaN values for that feature.
    """
    if calculated_metrics is None:
        calculated_metrics = ['blink_dist', 'lip_dist']
        
    df = df_extracted.copy()
    
    # Blink distance (average of both eyes)
    if 'blink_dist' in calculated_metrics:
        blink_dist = _compute_blink_distance(df)
        df['blink_dist'] = blink_dist
    
    # Lip distance
    if 'lip_dist' in calculated_metrics:
        lip_dist = _compute_lip_distance(df)
        df['lip_dist'] = lip_dist
    
    return df


def _compute_blink_distance(df: pd.DataFrame) -> pd.Series:
    """
    Compute average blink distance (vertical eyelid separation).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with eye keypoint columns
        
    Returns
    -------
    pd.Series
        Blink distance per frame
    """
    # Try to find upper and lower eyelid keypoints
    # Common OpenPose naming: LEye_Upper, LEye_Lower, REye_Upper, REye_Lower
    # or variations like LEyeTop, LEyeBottom
    
    distances = []
    
    for side in ['L', 'R']:
        # Try different naming patterns
        upper_patterns = [f'{side}Eye_Upper', f'{side}EyeTop', f'{side}Eye_Top']
        lower_patterns = [f'{side}Eye_Lower', f'{side}EyeBottom', f'{side}Eye_Bottom']
        
        upper_y = None
        lower_y = None
        
        # Find upper eyelid
        for pattern in upper_patterns:
            col_name = f'{pattern}_y_offset'
            if col_name in df.columns:
                upper_y = df[col_name]
                break
        
        # Find lower eyelid  
        for pattern in lower_patterns:
            col_name = f'{pattern}_y_offset'
            if col_name in df.columns:
                lower_y = df[col_name]
                break
        
        # Compute distance if both found
        if upper_y is not None and lower_y is not None:
            distances.append((lower_y - upper_y).abs())
    
    if not distances:
        # Fallback: return NaN series
        return pd.Series(np.nan, index=df.index)
    
    # Average across eyes
    return pd.concat(distances, axis=1).mean(axis=1)


def _compute_lip_distance(df: pd.DataFrame) -> pd.Series:
    """
    Compute lip distance (vertical mouth opening).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with lip keypoint columns
        
    Returns
    -------
    pd.Series
        Lip distance per frame
    """
    # Try to find upper and lower lip keypoints
    upper_patterns = ['UpperLip', 'Lip_Upper', 'LipTop', 'Lip_Top']
    lower_patterns = ['LowerLip', 'Lip_Lower', 'LipBottom', 'Lip_Bottom']
    
    upper_y = None
    lower_y = None
    
    # Find upper lip
    for pattern in upper_patterns:
        col_name = f'{pattern}_y_offset'
        if col_name in df.columns:
            upper_y = df[col_name]
            break
    
    # Find lower lip
    for pattern in lower_patterns:
        col_name = f'{pattern}_y_offset'
        if col_name in df.columns:
            lower_y = df[col_name]
            break
    
    # Compute distance
    if upper_y is not None and lower_y is not None:
        return (lower_y - upper_y).abs()
    
    # Fallback: return NaN series
    return pd.Series(np.nan, index=df.index)


# ============================================================================
# LINEAR METRICS (VELOCITY, ACCELERATION)
# ============================================================================

def compute_linear_metrics(series: pd.Series, fps: float = 60) -> Dict[str, float]:
    """
    Compute linear metrics from a 1D time series (e.g., PC trajectory).
    
    Includes position RMS, signed velocity/acceleration, and magnitude 
    (absolute value) of velocity/acceleration.
    
    Parameters
    ----------
    series : pd.Series or np.ndarray
        1D time series
    fps : float
        Sampling rate in Hz (default: 60)
        
    Returns
    -------
    dict
        Dictionary with metric names and values:
        - RMS: Root mean square of position
        - MeanVel: Mean velocity (signed)
        - StdVel: Standard deviation of velocity
        - MeanAcc: Mean acceleration (signed)
        - StdAcc: Standard deviation of acceleration
        - MeanVelMag: Mean velocity magnitude (speed)
        - StdVelMag: Standard deviation of velocity magnitude
        - MeanAccelMag: Mean acceleration magnitude
        - StdAccelMag: Standard deviation of acceleration magnitude
    """
    dt = 1.0 / fps

    # Convert to numpy if needed
    if isinstance(series, pd.Series):
        series = series.values

    # First and second derivatives
    vel = np.gradient(series, dt)
    acc = np.gradient(vel, dt)

    # Magnitudes (absolute value per sample)
    vel_mag = np.abs(vel)
    acc_mag = np.abs(acc)

    return {
        # Position (PC trajectory or feature value)
        "RMS": float(np.sqrt(np.mean(series**2))),

        # Velocity (signed)
        "MeanVel": float(np.mean(vel)),
        "StdVel": float(np.std(vel)),

        # Acceleration (signed)
        "MeanAcc": float(np.mean(acc)),
        "StdAcc": float(np.std(acc)),

        # Velocity magnitude (speed, direction-invariant)
        "MeanVelMag": float(np.mean(vel_mag)),
        "StdVelMag": float(np.std(vel_mag)),

        # Acceleration magnitude (intensity of acceleration)
        "MeanAccelMag": float(np.mean(acc_mag)),
        "StdAccelMag": float(np.std(acc_mag)),
    }
