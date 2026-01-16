"""
MOSAIC Pipeline Module

Dataset-specific utilities for loading, preprocessing, and organizing
MOSAIC dyadic pose data.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Import from shared preprocessing utilities
from pose_dynamics.preprocessing.signal_cleaning import (
    normalize_by_resolution,
    mask_low_confidence, 
    interpolate_nans,
    filter_data_safe_preserve_nans
)

# Import order_xy_pairs from local alignment module
from .alignment import order_xy_pairs

# ============================================================================
# KEYPOINT EXTRACTION
# ============================================================================

# Predefined keypoint sets (from config)
PREDEFINED_SETS = {
    'face': ["Eye", "Pupil", "Chin", "Jaw", "Cheek", "Nostril", "Lip", "Temple", "Nose"], 
    'center_face': ["Eye", "Pupil", "Chin", "Nostril", "Lip", "Nose"],
    'hand': ["wristBase", "Tip"],
    'arm': ["Shoulder", "Elbow", "Wrist"],
    'body': ["Shoulder", "MidHip", "Neck"],
    'temple': ["Temple"],   # strict match
    'nose': ["Nose"]        # strict match
}

def extract_keypoints(file_path_or_data, sets=["hand", "face", "body"]):
    """
    Extract keypoints (x_offset, y_offset, confidence) from MOSAIC dataset.

    Parameters
    ----------
    file_path_or_data : str or pd.DataFrame
        Path to CSV file or DataFrame with pose data
    sets : list of str
        Keypoint sets to extract. Options: "hand", "face", "body", 
        "center_face", "arm", "temple", "nose"

    Returns
    -------
    pd.DataFrame
        DataFrame containing selected keypoint columns
        
    Notes
    -----
    MOSAIC uses OpenPose format with centered coordinates (_x_offset, _y_offset)
    and confidence scores (_confidence).
    """
    # Load data if needed
    if isinstance(file_path_or_data, str):
        data = pd.read_csv(file_path_or_data)
    elif isinstance(file_path_or_data, pd.DataFrame):
        data = file_path_or_data
    else:
        raise ValueError("Input must be a file path or a pandas DataFrame.")

    # Collect labels from requested sets
    labels = []
    for s in sets:
        labels.extend(PREDEFINED_SETS.get(s, [s]))

    # Separate strict and substring matches for efficiency
    strict_labels = set(lbl for lbl in labels if lbl in ["Nose", "Temple"])
    substring_labels = set(lbl for lbl in labels if lbl not in strict_labels)

    # Find matching columns (optimized: avoid nested loops)
    xyconf_cols = []
    valid_suffixes = ("_x_offset", "_y_offset", "_confidence")
    
    for col in data.columns:
        # Quick suffix check
        if not col.endswith(valid_suffixes):
            continue
        
        # Check strict matches (faster)
        if any(col.startswith(lbl + "_") for lbl in strict_labels):
            xyconf_cols.append(col)
        # Check substring matches
        elif any(lbl in col for lbl in substring_labels):
            xyconf_cols.append(col)

    return data[xyconf_cols].copy()

# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

def preprocess_mosaic_trial(
    df: pd.DataFrame,
    expected_cols: List[str],
    conf_threshold: float = 0.3,
    max_interp_gap: int = 60,
    filter_cutoff: float = 10.0,
    filter_order: int = 4,
    fps: float = 60.0,
    target_fps: Optional[float] = None,
    video_width: int = 720,
    video_height: int = 720
) -> pd.DataFrame:
    """
    Preprocess MOSAIC pose data for a single trial.
    Parameters
    ----------
    df : pd.DataFrame
        Raw pose data DataFrame
    expected_cols : list of str
        List of expected columns in the final output
    conf_threshold : float
        Confidence threshold for masking low-confidence keypoints
    max_interp_gap : int
        Maximum gap (in frames) for interpolation
    filter_cutoff : float
        Cutoff frequency (Hz) for low-pass filtering
    filter_order : int
        Order of the Butterworth filter
    fps : float
        Original frames per second of the data
    target_fps : float or None
        Desired frames per second after downsampling (None to keep original)
    video_width : int
        Width of the video frame (for normalization)
    video_height : int
        Height of the video frame (for normalization)
    """

    if target_fps is not None and target_fps > fps:
        raise ValueError(f"target_fps ({target_fps}) cannot exceed fps ({fps})")

    selected = normalize_by_resolution(df, width=video_width, height=video_height)
    selected = mask_low_confidence(selected, threshold=conf_threshold)
    selected = interpolate_nans(selected, max_gap=max_interp_gap)

    selected = filter_data_safe_preserve_nans(
        selected,
        fps=fps,
        cutoff_hz=filter_cutoff,
        order=filter_order,
    )

    # Downsample if requested
    if target_fps is not None and target_fps < fps:
        downsample_factor = int(np.round(fps / target_fps))
        selected = selected.iloc[::downsample_factor].reset_index(drop=True)

    selected.columns = [col.strip() for col in selected.columns]

    # Ensure expected XY columns exist
    for col in expected_cols:
        if col not in selected.columns:
            selected[col] = np.nan

    return selected[expected_cols]

# ============================================================================
# WINDOW UTILITIES
# ============================================================================

def get_window_indices(data_length: int, window_size: int, overlap: float) -> List[Tuple[int, int]]:
    """
    Generate sliding window indices.
    
    Parameters
    ----------
    data_length : int
        Total number of frames
    window_size : int
        Window size in frames
    overlap : float
        Overlap fraction (0-1)
        
    Returns
    -------
    list of tuples
        List of (start, end) indices for each window
    """
    step = int(window_size * (1 - overlap))
    return [(start, start + window_size) 
            for start in range(0, data_length - window_size + 1, step)]