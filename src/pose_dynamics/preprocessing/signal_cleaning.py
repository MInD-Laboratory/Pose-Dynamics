"""
Generic signal cleaning utilities.

Pure time-series operations that do NOT know anything about keypoints,
pose, filenames, or experimental conditions.
"""

from __future__ import annotations
from typing import Tuple, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import signal


# ---------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------

def resample_array(
    x: np.ndarray,
    orig_rate: float,
    target_rate: float,
    axis: int = 0,
) -> np.ndarray:
    """
    Resample a uniformly sampled array from orig_rate -> target_rate.

    Uses scipy.signal.resample_poly for good anti-aliasing.

    Parameters
    ----------
    x : np.ndarray
        Input array. `axis` is treated as time.
    orig_rate : float
        Original sampling rate in Hz.
    target_rate : float
        Desired sampling rate in Hz.
    axis : int
        Time axis (default 0).

    Returns
    -------
    np.ndarray
        Resampled array.
    """
    if orig_rate <= 0 or target_rate <= 0:
        raise ValueError("orig_rate and target_rate must be positive")

    # rational approximation of the rate ratio
    up, down = signal.resample_poly.__wrapped__.__defaults__[:2]  # type: ignore
    # Fallback: compute directly if that hack ever breaks
    ratio = target_rate / orig_rate
    up = int(round(ratio * 1000))
    down = 1000

    return signal.resample_poly(x, up=up, down=down, axis=axis)


def resample_dataframe(
    df: pd.DataFrame,
    orig_rate: float,
    target_rate: float,
) -> pd.DataFrame:
    """
    Resample all numeric columns of a DataFrame.

    Assumes rows are uniformly sampled in time.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    orig_rate : float
        Original sampling rate in Hz.
    target_rate : float
        Desired sampling rate in Hz.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with the same columns.
    """
    x = df.to_numpy()
    x_resampled = resample_array(x, orig_rate=orig_rate, target_rate=target_rate, axis=0)
    n_new = x_resampled.shape[0]

    # simple synthetic time index (seconds) if you ever care
    time = np.arange(n_new) / target_rate
    out = pd.DataFrame(x_resampled, columns=df.columns)
    out.index = time
    return out


def align_pair(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    time_col: Optional[str] = None,
    mode: str = "truncate"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two time series DataFrames to a common temporal support.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Input DataFrames to be aligned.
    time_col : str, optional
        Name of a time / frame column to align on. If provided and present
        in both DataFrames, alignment is done by intersecting time stamps.
        If None, alignment is done by truncating to the same length.
    mode : {"truncate", "inner"}
        - "truncate": (no time_col) cut both to min(len(df1), len(df2)).
        - "inner": (with time_col) keep only overlapping time stamps.

    Returns
    -------
    df1_aligned, df2_aligned : pd.DataFrame
        Aligned DataFrames with matching number of rows and (if time_col is
        used) the same time index/order.

    Notes
    -----
    - This is intentionally simple and dataset-agnostic.
    - For more complex alignment (e.g. cross-correlation), write a
      dataset-specific wrapper in the project pipeline.
    """
    df1 = df1.copy()
    df2 = df2.copy()

    # Case 1: align by explicit time column if provided
    if time_col is not None and time_col in df1.columns and time_col in df2.columns:
        # Set index to time and intersect
        s1 = df1.set_index(time_col)
        s2 = df2.set_index(time_col)

        common_idx = s1.index.intersection(s2.index)
        if mode == "inner":
            idx = common_idx
        else:
            # For now, inner and truncate are effectively the same when using time stamps.
            idx = common_idx

        idx = idx.sort_values()

        s1_aligned = s1.loc[idx]
        s2_aligned = s2.loc[idx]

        # Restore time_col as a regular column
        df1_aligned = s1_aligned.reset_index()
        df2_aligned = s2_aligned.reset_index()

        return df1_aligned, df2_aligned

    # Case 2: no usable time column → truncate to min length
    n = min(len(df1), len(df2))
    df1_aligned = df1.iloc[:n].reset_index(drop=True)
    df2_aligned = df2.iloc[:n].reset_index(drop=True)
    return df1_aligned, df2_aligned


# ---------------------------------------------------------------------
# Pose-specific preprocessing utilities
# ---------------------------------------------------------------------

def normalize_by_resolution(
    df: pd.DataFrame,
    width: int = 720,
    height: int = 720,
) -> pd.DataFrame:
    """
    Normalize pose coordinates by video resolution.
    
    Divides x coordinates by width and y coordinates by height to get
    coordinates in [0, 1] range.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with x/y coordinate columns (e.g., 'Nose_x_offset', 'Nose_y_offset')
    width : int
        Video width in pixels
    height : int
        Video height in pixels
        
    Returns
    -------
    pd.DataFrame
        Normalized DataFrame
    """
    normalized = df.copy()
    
    for col in df.columns:
        if col.endswith('_x_offset') or col.endswith('_x'):
            normalized[col] = df[col] / width
        elif col.endswith('_y_offset') or col.endswith('_y'):
            normalized[col] = df[col] / height
    
    return normalized


def mask_low_confidence(
    df: pd.DataFrame,
    threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Mask keypoints with confidence below threshold by setting to NaN.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with coordinate and confidence columns
    threshold : float
        Minimum confidence value (0-1)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with low-confidence coordinates set to NaN
    """
    masked = df.copy()
    
    # Find all confidence columns
    conf_cols = [col for col in df.columns if col.endswith('_conf') or col.endswith('_confidence')]
    
    for conf_col in conf_cols:
        # Extract keypoint name
        if conf_col.endswith('_conf'):
            base = conf_col[:-5]  # Remove '_conf'
        else:
            base = conf_col[:-11]  # Remove '_confidence'
        
        # Find corresponding x/y columns
        x_col = f"{base}_x_offset" if f"{base}_x_offset" in df.columns else f"{base}_x"
        y_col = f"{base}_y_offset" if f"{base}_y_offset" in df.columns else f"{base}_y"
        
        # Mask low confidence points
        low_conf_mask = df[conf_col] < threshold
        if x_col in masked.columns:
            masked.loc[low_conf_mask, x_col] = np.nan
        if y_col in masked.columns:
            masked.loc[low_conf_mask, y_col] = np.nan
    
    return masked


def interpolate_nans(
    df: pd.DataFrame,
    max_gap: int = 60,
) -> pd.DataFrame:
    """
    Interpolate NaN values in DataFrame columns.
    
    Wrapper around interpolate_dataframe_nan_runs for clearer naming.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    max_gap : int
        Maximum consecutive NaNs to interpolate
        
    Returns
    -------
    pd.DataFrame
        DataFrame with interpolated values
    """
    return interpolate_dataframe_nan_runs(df, max_run=max_gap)


def filter_data_safe_preserve_nans(
    df: pd.DataFrame,
    fps: float,
    cutoff_hz: float,
    order: int = 4,
) -> pd.DataFrame:
    """
    Apply Butterworth filter while preserving NaN locations.
    
    NaN values are preserved in their original positions after filtering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    fps : float
        Sampling rate in Hz
    cutoff_hz : float
        Cutoff frequency for low-pass filter
    order : int
        Filter order
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with NaNs preserved
    """
    filtered = df.copy()
    
    for col in df.columns:
        # Skip non-numeric columns
        if not np.issubdtype(df[col].dtype, np.number):
            continue
            
        # Store NaN mask
        nan_mask = df[col].isna()
        
        # Skip if all NaN
        if nan_mask.all():
            continue
        
        # Skip if all valid (no filtering needed for NaN preservation)
        if not nan_mask.any():
            filtered[col] = butterworth_filter_dataframe(
                df[[col]], fs=fps, cutoff_hz=cutoff_hz, order=order
            )[col]
            continue
        
        # For columns with some NaNs: forward/back fill temporarily for filtering
        temp = df[col].ffill().bfill()
        
        # Filter the filled data
        temp_df = pd.DataFrame({col: temp})
        filtered_temp = butterworth_filter_dataframe(
            temp_df, fs=fps, cutoff_hz=cutoff_hz, order=order
        )[col]
        
        # Restore NaNs to original positions
        filtered.loc[nan_mask, col] = np.nan
        filtered.loc[~nan_mask, col] = filtered_temp[~nan_mask]
    
    return filtered


# ---------------------------------------------------------------------
# Interpolation (NaN gap filling)
# ---------------------------------------------------------------------

def interpolate_run_limited(
    x: pd.Series,
    max_run: int = 60,
) -> pd.Series:
    """
    Linearly interpolate NaN runs up to max_run samples long.

    Uses pandas' built-in interpolate with limit to safely handle edges
    without manual slicing.

    Parameters
    ----------
    x : pd.Series
        Input series with NaNs.
    max_run : int
        Maximum consecutive NaNs to interpolate.

    Returns
    -------
    pd.Series
        Interpolated series.
    """
    s = x.copy()
    if max_run is None or max_run <= 0:
        return s
    return s.interpolate(method="linear", limit=max_run, limit_direction="both")


def interpolate_dataframe_nan_runs(
    df: pd.DataFrame,
    max_run: int = 60,
) -> pd.DataFrame:
    """
    Apply interpolate_run_limited to each column of a DataFrame.
    """
    out = df.copy()
    for c in out.columns:
        out[c] = interpolate_run_limited(out[c], max_run=max_run)
    return out


# ---------------------------------------------------------------------
# Detrending
# ---------------------------------------------------------------------

def detrend_array(
    x: np.ndarray,
    type: str = "linear",
    axis: int = 0,
) -> np.ndarray:
    """
    Remove mean or linear trend along time axis.
    """
    return signal.detrend(x, type=type, axis=axis)


def detrend_dataframe(
    df: pd.DataFrame,
    type: str = "linear",
) -> pd.DataFrame:
    """
    Detrend all numeric columns of a DataFrame.
    """
    x = df.to_numpy()
    x_dt = detrend_array(x, type=type, axis=0)
    return pd.DataFrame(x_dt, columns=df.columns, index=df.index)


# ---------------------------------------------------------------------
# Normalization (z-score / unit interval)
# ---------------------------------------------------------------------

def normalize_array(
    x: np.ndarray,
    mode: str = "zscore",
    axis: int = 0,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, dict]:
    """
    Normalize array along a given axis.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    mode : {'zscore', 'minmax', None}
        - 'zscore' : (x - mean) / std
        - 'minmax' : (x - min) / (max - min)
        - None     : no change
    axis : int
        Axis along which to compute stats.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    x_norm : np.ndarray
        Normalized data.
    stats : dict
        Dict with 'mean', 'std', 'min', 'max' as applicable.
    """
    if mode is None:
        return x.copy(), {}

    x = np.asarray(x, dtype=float)

    if mode == "zscore":
        mean = np.nanmean(x, axis=axis, keepdims=True)
        std = np.nanstd(x, axis=axis, keepdims=True)
        x_norm = (x - mean) / (std + eps)
        return x_norm, {"mean": mean, "std": std}

    if mode == "minmax":
        min_ = np.nanmin(x, axis=axis, keepdims=True)
        max_ = np.nanmax(x, axis=axis, keepdims=True)
        x_norm = (x - min_) / (max_ - min_ + eps)
        return x_norm, {"min": min_, "max": max_}

    raise ValueError(f"Unknown normalization mode '{mode}'")


def normalize_dataframe(
    df: pd.DataFrame,
    mode: str = "zscore",
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize all numeric columns of a DataFrame.

    Returns normalized df and stats dict for potential inverse transform.
    """
    x = df.to_numpy(dtype=float)
    x_norm, stats = normalize_array(x, mode=mode, axis=0)
    out = pd.DataFrame(x_norm, columns=df.columns, index=df.index)
    return out, stats


# ---------------------------------------------------------------------
# Butterworth filtering
# ---------------------------------------------------------------------

def butterworth_filter_array(
    x: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
    axis: int = 0,
    btype: str = "low",
) -> np.ndarray:
    """
    Zero-phase Butterworth filter along time axis.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    fs : float
        Sampling rate (Hz).
    cutoff_hz : float
        Cutoff frequency (Hz).
    order : int
        Filter order.
    axis : int
        Time axis.
    btype : str
        'low', 'high', 'bandpass', etc.

    Returns
    -------
    np.ndarray
        Filtered data.
    """
    nyq = fs / 2.0
    if isinstance(cutoff_hz, (list, tuple, np.ndarray)):
        wn = np.asarray(cutoff_hz, dtype=float) / nyq
    else:
        wn = cutoff_hz / nyq

    b, a = signal.butter(order, wn, btype=btype, analog=False)
    return signal.filtfilt(b, a, x, axis=axis)


def butterworth_filter_dataframe(
    df: pd.DataFrame,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
    btype: str = "low",
) -> pd.DataFrame:
    """
    Apply Butterworth filter to all numeric columns of a DataFrame.
    """
    x = df.to_numpy(dtype=float)
    x_f = butterworth_filter_array(x, fs=fs, cutoff_hz=cutoff_hz,
                                   order=order, axis=0, btype=btype)
    return pd.DataFrame(x_f, columns=df.columns, index=df.index)


# ---------------------------------------------------------------------
# Windowing with overlap
# ---------------------------------------------------------------------

def sliding_windows(
    x: np.ndarray,
    window_size: int,
    step: int,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping sliding windows along time axis.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    window_size : int
        Number of samples per window.
    step : int
        Step between windows (e.g., window_size//2 for 50% overlap).
    axis : int
        Time axis.

    Returns
    -------
    windows : np.ndarray
        Array of shape (n_windows, window_size, ...) where time is second dim.
    starts : np.ndarray
        Start indices for each window along the original axis.
    """
    x = np.asarray(x)
    n = x.shape[axis]
    if window_size > n:
        raise ValueError("window_size cannot be larger than signal length")

    starts = np.arange(0, n - window_size + 1, step)
    windows = []
    for s in starts:
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(s, s + window_size)
        windows.append(x[tuple(sl)])

    windows = np.stack(windows, axis=0)
    return windows, starts
