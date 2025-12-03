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

    # Find matching columns
    xyconf_cols = []
    for col in data.columns:
        if not (col.endswith("_x_offset") or col.endswith("_y_offset") or col.endswith("_confidence")):
            continue
        for lbl in labels:
            if lbl in ["Nose", "Temple"]:   # strict match
                if col.startswith(lbl + "_"):
                    xyconf_cols.append(col)
                    break
            else:   # substring match
                if lbl in col:
                    xyconf_cols.append(col)
                    break

    return data[xyconf_cols].copy()


# ============================================================================
# SESSION LOADING
# ============================================================================

def load_mosaic_session(data_path: str, session_number: int, 
                       trial_number: int, role: str) -> Optional[pd.DataFrame]:
    """
    Load a single MOSAIC pose file.
    
    Parameters
    ----------
    data_path : str
        Base directory containing session folders
    session_number : int
        Session number (1-49)
    trial_number : int
        Trial number (1-6)
    role : str
        Participant role: "left" or "right"
        
    Returns
    -------
    pd.DataFrame or None
        Pose data, or None if file not found
    """
    file_path = (
        f"{data_path}/Session{session_number:03}/trial_{trial_number}/"
        f"pose_S{session_number:03}_T{trial_number}_{role.lower()}.csv"
    )
    
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def load_trial_info(data_path: str, session_number: int) -> Optional[pd.DataFrame]:
    """
    Load trial info CSV for a session.
    
    Parameters
    ----------
    data_path : str
        Base directory containing session folders
    session_number : int
        Session number (1-49)
        
    Returns
    -------
    pd.DataFrame or None
        Trial info with audio condition metadata
    """
    trial_info_file = f'{data_path}/Session{session_number:03}/S{session_number:03}_TrialInfo.csv'
    try:
        return pd.read_csv(trial_info_file)
    except FileNotFoundError:
        print(f"File not found: {trial_info_file}")
        return None


def extract_audio_conditions(trial_info_df: pd.DataFrame) -> List[str]:
    """
    Extract audio condition labels from trial info.
    
    Parameters
    ----------
    trial_info_df : pd.DataFrame
        Trial info DataFrame with 'NoiseFile_Audio' column
        
    Returns
    -------
    list of str
        Audio condition labels for each trial
    """
    audio_conditions = trial_info_df['NoiseFile_Audio'].str.split("_", expand=True)[0].values
    return [audio.split("\\")[-1] for audio in audio_conditions]


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

def preprocess_mosaic_trial(
    df: pd.DataFrame,
    expected_cols: List[str],
    conf_threshold: float = 0.4,
    max_interp_gap: int = 60,
    filter_cutoff: float = 10.0,
    filter_order: int = 4,
    fps: float = 60.0,
    video_width: int = 720,
    video_height: int = 720
) -> pd.DataFrame:
    """
    Apply standard MOSAIC preprocessing pipeline to a single trial.
    
    Steps:
    1. Extract relevant keypoints
    2. Normalize by resolution
    3. Mask low confidence
    4. Interpolate short gaps
    5. Apply low-pass filter
    6. Ensure column consistency
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw pose data
    expected_cols : list of str
        Expected column names for consistency
    conf_threshold : float
        Minimum confidence threshold (default: 0.4)
    max_interp_gap : int
        Maximum frames to interpolate (default: 60)
    filter_cutoff : float
        Butterworth filter cutoff in Hz (default: 10.0)
    filter_order : int
        Filter order (default: 4)
    fps : float
        Sampling rate in Hz (default: 60.0)
    video_width : int
        Video width in pixels (default: 720)
    video_height : int
        Video height in pixels (default: 720)
        
    Returns
    -------
    pd.DataFrame
        Preprocessed pose data with consistent columns
    """
    # Normalize coordinates
    selected = normalize_by_resolution(df, width=video_width, height=video_height)
    
    # Mask low confidence
    selected = mask_low_confidence(selected, threshold=conf_threshold)
    
    # Interpolate gaps
    selected = interpolate_nans(selected, max_gap=max_interp_gap)
    
    # Filter
    selected = filter_data_safe_preserve_nans(
        selected,
        fps=fps,
        cutoff_hz=filter_cutoff,
        order=filter_order,
    )
    
    # Clean column names
    selected.columns = [col.strip() for col in selected.columns]
    
    # Ensure all expected columns exist
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


# ============================================================================
# BATCH LOADING
# ============================================================================

def load_all_sessions(
    data_path: str,
    session_range: Tuple[int, int] = (1, 50),
    keypoint_sets: List[str] = None,
    skip_sessions: List[int] = None,
    conf_threshold: float = 0.4,
    max_interp_gap: int = 60,
    verbose: bool = True
) -> Tuple[List[Tuple], List[str]]:
    """
    Load and preprocess all MOSAIC sessions.
    
    Parameters
    ----------
    data_path : str
        Base directory containing session folders
    session_range : tuple of int
        (start, end) session numbers (default: 1-49)
    keypoint_sets : list of str
        Keypoint sets to extract (default: ["center_face", "body", "arm"])
    skip_sessions : list of int
        Session numbers to skip (default: None)
    conf_threshold : float
        Confidence threshold (default: 0.4)
    max_interp_gap : int
        Maximum interpolation gap in frames (default: 60)
    verbose : bool
        Print progress (default: True)
        
    Returns
    -------
    trials : list of tuples
        List of (preprocessed_df, metadata_dict) tuples
    expected_cols : list of str
        Standardized column names
    """
    if keypoint_sets is None:
        keypoint_sets = ["center_face", "body", "arm"]
    if skip_sessions is None:
        skip_sessions = []
        
    trials = []
    expected_cols = None
    
    start_session, end_session = session_range
    
    for session_number in range(start_session, end_session):
        if session_number in skip_sessions:
            if verbose:
                print(f"[SKIP] Session {session_number:03}")
            continue
            
        # Load trial info
        trial_info = load_trial_info(data_path, session_number)
        if trial_info is None:
            continue
            
        audio_conditions = extract_audio_conditions(trial_info)
        
        for trial_number in range(1, 7):  # 6 trials per session
            for role in ["Left", "Right"]:
                # Load pose data
                data = load_mosaic_session(
                    data_path, session_number, trial_number, role
                )
                if data is None:
                    continue
                    
                # Extract keypoints
                selected = extract_keypoints(data, sets=keypoint_sets)
                
                # Get expected columns from first valid trial
                if expected_cols is None and not selected.empty:
                    expected_cols = order_xy_pairs(selected.columns)
                    
                # Preprocess
                try:
                    preprocessed = preprocess_mosaic_trial(
                        selected,
                        expected_cols,
                        conf_threshold=conf_threshold,
                        max_interp_gap=max_interp_gap
                    )
                    
                    metadata = {
                        'Session': session_number,
                        'Trial': trial_number,
                        'Role': role,
                        'Condition': audio_conditions[trial_number - 1]
                    }
                    
                    trials.append((preprocessed, metadata))
                    
                except Exception as e:
                    if verbose:
                        print(f"Error processing S{session_number:03} "
                              f"T{trial_number} {role}: {e}")
                    continue
    
    if verbose:
        print(f"\n[DONE] Loaded {len(trials)} trials")
        
    return trials, expected_cols
