"""Data preprocessing and manipulation utilities for pose analysis.

This module contains functions for processing, filtering, and transforming
landmark data from pose estimation outputs.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd
from . import signal_utils

from pose_dynamics.config import get_cfg

# ---------- Filename and metadata parsing ------------------------------------
def parse_participant_trial(filename: str) -> Tuple[str, int]:
    """Parse participant ID and trial number from pose filename.

    Args:
        filename: Pose filename (e.g., '3101_02_pose.csv')

    Returns:
        Tuple of (participant_id, trial_number)

    Raises:
        ValueError: If filename doesn't match expected pattern
    """
    base = Path(filename).name
    # Match pattern: PPPP_TT where PPPP is participant ID and TT is trial number
    m = re.match(r"^(\d{4})_(\d{2}).*\.csv$", base)
    if not m:
        raise ValueError(f"Filename does not match expected pattern 'PPPP_TT_*.csv': {base}")
    return m.group(1), int(m.group(2))


def load_participant_info(participant_info_path: str) -> pd.DataFrame:
    """Load participant information including condition order.

    Args:
        participant_info_path: Path to participant_info.csv file

    Returns:
        DataFrame with participant IDs and session conditions
    """
    df = pd.read_csv(participant_info_path)
    # Clean participant ID to ensure it's a string
    df['Participant ID'] = df['Participant ID'].astype(str)
    return df


def create_condition_mapping(participant_info: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    """Create a mapping from participant ID and trial number to condition.

    Args:
        participant_info: DataFrame from load_participant_info()

    Returns:
        Nested dict: {participant_id: {trial_number: condition}}
        Example: mapping['3101'][1] returns 'L'
    """
    condition_map = {}

    for _, row in participant_info.iterrows():
        participant_id = str(row['Participant ID'])

        # Create trial mapping for this participant
        trial_map = {}

        # Parse each session (Session01, Session02, Session03)
        for session_num in [1, 2, 3]:
            session_col = f'session0{session_num}'
            if session_col in row and not pd.isna(row[session_col]) and row[session_col] != '-':
                session_value = str(row[session_col])

                # Extract condition letter (L, M, or H) - first character
                if session_value:
                    condition = session_value[0]  # L, M, or H
                    # Map trial number (session number) to condition
                    trial_map[session_num] = condition

        condition_map[participant_id] = trial_map

    return condition_map


def get_condition_for_file(filename: str, condition_map: Dict[str, Dict[int, str]]) -> str:
    """Get the condition (L/M/H) for a given pose file.

    Args:
        filename: Pose filename
        condition_map: Mapping from create_condition_mapping()

    Returns:
        Condition letter ('L', 'M', or 'H')

    Raises:
        ValueError: If no condition mapping found
    """
    participant_id, trial_num = parse_participant_trial(filename)

    if participant_id not in condition_map:
        raise ValueError(f"No condition mapping for participant {participant_id}")

    if trial_num not in condition_map[participant_id]:
        raise ValueError(f"No condition mapping for participant {participant_id} trial {trial_num}")

    return condition_map[participant_id][trial_num]


# ---------- Column detection and selection -----------------------------------
def detect_conf_prefix_case_insensitive(columns: List[str]) -> str:
    """Detect the confidence column prefix from available columns.

    Checks for common confidence prefixes: 'prob', 'c', 'confidence'

    Args:
        columns: List of column names to search

    Returns:
        The detected confidence prefix

    Raises:
        ValueError: If no confidence columns found
    """
    cols_low = [c.lower() for c in columns]
    for prefix in ("prob", "c", "confidence"):
        if any(col.startswith(prefix) for col in cols_low):
            return prefix
    raise ValueError("Confidence prefix not found (expected 'prob*', 'c*', or 'confidence*').")


def find_real_colname(prefix: str, i: int, columns: List[str]) -> Optional[str]:
    """Find the actual column name for a given prefix and index.

    Handles case-insensitive matching and partial matches.

    Args:
        prefix: Column prefix (e.g., 'x', 'y', 'prob')
        i: Landmark index number
        columns: List of available column names

    Returns:
        Actual column name if found, None otherwise
    """
    target = f"{prefix}{i}".lower()
    # First try exact match
    for col in columns:
        if col.lower() == target:
            return col
    # Then try prefix match
    for col in columns:
        if col.lower().startswith(target):
            return col
    return None


def lm_triplet_colnames(i: int, conf_prefix: str, columns: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Get the (x, y, confidence) column names for a landmark.

    Args:
        i: Landmark index
        conf_prefix: Confidence column prefix
        columns: Available column names

    Returns:
        Tuple of (x_col, y_col, conf_col) names, or None for missing columns
    """
    return (
        find_real_colname("x", i, columns),
        find_real_colname("y", i, columns),
        find_real_colname(conf_prefix, i, columns),
    )


# ---------- Landmark selection and filtering ---------------------------------
def relevant_indices() -> List[int]:
    """Get the list of relevant landmark indices from configuration.

    Combines all landmark groups defined in config:
    - Procrustes reference points
    - Head rotation points
    - Mouth points
    - Center face points
    - Blink detection points (both eyes)
    - Pupil points

    Returns:
        Sorted list of unique landmark indices
    """
    s = set()
    CFG = get_cfg()
    s.update(CFG.PROCRUSTES_REF)
    s.update(CFG.HEAD_ROT)
    s.update(CFG.MOUTH)
    s.update(CFG.CENTER_FACE)
    s.update(CFG.BLINK_L_TOP)
    s.update(CFG.BLINK_L_BOT)
    s.update(CFG.BLINK_R_TOP)
    s.update(CFG.BLINK_R_BOT)
    s.update([69, 70])  # pupils
    return sorted(s)


def filter_df_to_relevant(df: pd.DataFrame, conf_prefix: str, indices: List[int]) -> pd.DataFrame:
    """Filter DataFrame to keep only relevant landmark columns.

    Args:
        df: Input DataFrame with all landmarks
        conf_prefix: Confidence column prefix
        indices: List of landmark indices to keep

    Returns:
        DataFrame with only the (x, y, conf) columns for specified landmarks

    Raises:
        ValueError: If no relevant columns found
    """
    kept: List[str] = []
    cols = list(df.columns)

    for i in indices:
        x, y, c = lm_triplet_colnames(i, conf_prefix, cols)
        if x and y and c:
            kept.extend([x, y, c])

    if not kept:
        raise ValueError("No relevant triplets found.")

    return df.loc[:, kept].copy()


# ---------- Confidence-based masking -----------------------------------------
def confidence_mask(df_reduced: pd.DataFrame, conf_prefix: str, indices: List[int], thr: float) -> Tuple[pd.DataFrame, Dict]:
    """Apply confidence masking to landmark data.

    Sets coordinates to NaN where confidence is below threshold.

    Args:
        df_reduced: DataFrame with landmark data
        conf_prefix: Confidence column prefix
        indices: Landmark indices to process
        thr: Confidence threshold (0-1)

    Returns:
        Tuple of:
        - DataFrame with low-confidence values masked
        - Dictionary with masking statistics
    """
    dfm = df_reduced.copy()
    cols = list(dfm.columns)
    n_frames = len(dfm)
    per_lm = {}
    total_considered = 0
    total_masked = 0

    for i in indices:
        x, y, c = lm_triplet_colnames(i, conf_prefix, cols)
        if not (x and y and c):
            continue

        # Get confidence values and identify low-confidence frames
        conf = pd.to_numeric(dfm[c], errors="coerce")
        low = conf < thr
        low = low.fillna(False)

        # Count values before masking
        pre_x = dfm[x].notna()
        pre_y = dfm[y].notna()
        considered = int((pre_x | pre_y).sum()) * 2  # x and y coordinates
        masked = int(((pre_x | pre_y) & low).sum()) * 2

        # Apply masking
        if low.any():
            dfm.loc[low, [x, y, c]] = np.nan

        # Store statistics for this landmark
        per_lm[i] = {
            "frames_total": int(n_frames),
            "frames_low_conf": int(low.sum()),
            "coords_considered": considered,
            "coords_masked": masked,
            "pct_frames_low_conf": (int(low.sum()) / n_frames * 100.0) if n_frames else 0.0
        }

        total_considered += considered
        total_masked += masked

    # Overall statistics
    overall = {
        "frames": int(n_frames),
        "n_landmarks_considered": len(per_lm),
        "total_coord_values": int(total_considered),
        "total_coords_masked": int(total_masked),
        "pct_coords_masked": (total_masked / total_considered * 100.0) if total_considered else 0.0
    }

    return dfm, {"per_landmark": per_lm, "overall": overall}


# --- Convenience wrappers matching notebook API ---

def mask_low_confidence(df: pd.DataFrame, thresh: float = 0.3) -> pd.DataFrame:
    """Wrapper around confidence_mask()."""
    conf_prefix = detect_conf_prefix_case_insensitive(df.columns)
    indices = relevant_indices()
    masked, stats = confidence_mask(df, conf_prefix, indices, thr=thresh)
    print(f"Masked {stats['overall']['pct_coords_masked']:.2f}% of coords below conf<{thresh}")
    return masked


def interpolate_missing(df: pd.DataFrame, max_run: int = 60) -> pd.DataFrame:
    """Interpolate short NaN runs (≤max_run) per column."""
    out = df.copy()
    for c in out.columns:
        out[c] = signal_utils.interpolate_run_limited(out[c], max_run=max_run)
    return out


def butterworth_filter(df: pd.DataFrame, cutoff: float = 10.0, order: int = 4, fs: float = 60.0) -> pd.DataFrame:
    """Apply low-pass Butterworth per column."""
    out = df.copy()
    for c in out.columns:
        out[c] = signal_utils.butterworth_segment_filter(out[c], order=order, cutoff_hz=cutoff, fs=fs)
    return out



# for 3d pose data processing functions
# ZED skeleton connections
SKELETON_CONNECTIONS = {
    "face": [    
        ("9", "7"),   # Right ear to right eye  
        ("7", "5"),   # Right eye to nose  
        ("5", "6"),   # Nose to left eye
        ("6", "8"),   # Left eye to left ear
        ("5", "4"),   # Nose to mouth
        ("4", "3"),   # Mouth to neck
    ],
    "body": [
        ("3", "10"),    # Neck to left shoulder
        ("3", "2"),     # Neck to mid chest
        ("2","1"),      # Mid chest to stomach
        ("1", "0"),     # Stomach to pelvis
        ("10", "12"),   # Left shoulder to left bicep
        ("12", "14"),   # Left bicep to left elbow
        ("14", "16"),   # Left elbow to left wrist
        ("16", "30"),   # Left wrist to left finger 1
        ("16", "32"),   # Left wrist to left finger 2
        ("16", "34"),   # Left wrist to left finger 3
        ("16", "36"),   # Left wrist to left finger 4
        ("3", "11"),    # Neck to right shoulder
        ("11", "13"),   # Right shoulder to right bicep
        ("13", "15"),   # Right bicep to right elbow
        ("15", "17"),   # Right elbow to right wrist
        ("17", "31"),   # Right wrist to right finger 1
        ("17", "33"),   # Right wrist to right finger 2
        ("17", "35"),   # Right wrist to right finger 3
        ("17", "37"),   # Right wrist to right finger 4
    ],
    "legs": [
        ("0", "19"),    # Pelvis to right hip
        ("19", "21"),   # Right hip to right knee
        ("21", "23"),   # Right knee to right ankle
        ("23", "29"),   # Right ankle to right heel
        ("23", "27"),   # Right ankle to right outer foot
        ("23", "25"),   # Right ankle to right inner foot
        ("0", "18"),    # Pelvis to left hip
        ("18", "20"),   # Left hip to left knee
        ("20", "22"),   # Left knee to left ankle
        ("22", "28"),   # Left ankle to left heel
        ("22", "26"),   # Left ankle to left outer foot
        ("22", "24"),   # Left ankle to left inner foot
    ]
}

def build_edge_list(connections, include=("face","body","legs")):
    edges = []
    for k in include:
        for i, j in connections.get(k, []):
            edges.append((int(i), int(j)))
    return edges

edges = build_edge_list(SKELETON_CONNECTIONS, include=("body","legs","face"))

def order_xyz_triplets(columns):
    idxs = sorted({int(c[1:]) for c in columns if (c[0] in ("x","y","z") and c[1:].isdigit())})
    ordered = []
    for i in idxs:
        t = [f"x{i}", f"y{i}", f"z{i}"]
        if all(col in columns for col in t):
            ordered.extend(t)
    return ordered

def compute_procrustes_transform_3d(template, trial_mean, allow_rotation=True, allow_scale=False):
    T0 = template.mean(axis=0)
    M0 = trial_mean.mean(axis=0)
    Tc = template - T0
    Mc = trial_mean - M0
    nT, nM = np.linalg.norm(Tc), np.linalg.norm(Mc)
    if nT < 1e-12 or nM < 1e-12:
        return np.eye(3), 1.0, (T0 - M0)

    Tc_n, Mc_n = Tc / nT, Mc / nM
    if allow_rotation:
        H = Mc_n.T @ Tc_n
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = U @ Vt
    else:
        R = np.eye(3)

    s = 1.0
    if allow_scale:
        num = np.trace((Mc @ R).T @ Tc)
        den = (Mc**2).sum()
        s = float(num / max(den, 1e-12))

    t = T0 - (s * M0 @ R)
    return R, s, t

def align_keypoints_3d(df, expected_cols=None, ref_idx=None,
                       template=None, use_procrustes=False,
                       allow_rotation=True, allow_scale=False):
    cols = expected_cols if expected_cols else order_xyz_triplets(df.columns)
    X = df[cols].to_numpy(dtype=np.float32)
    n_points, T = len(cols)//3, X.shape[0]
    coords = X.reshape(T, n_points, 3)

    if ref_idx is not None:
        coords -= coords[:, ref_idx:ref_idx+1, :]
    else:
        coords -= coords.mean(axis=1, keepdims=True)

    if not use_procrustes:
        return coords.reshape(T, n_points*3), (n_points, 3)

    trial_mean = coords.mean(axis=0)
    R, s, t = compute_procrustes_transform_3d(template, trial_mean,
                                              allow_rotation, allow_scale)
    aligned = (coords @ R) * s + t
    return aligned.reshape(T, n_points*3), (n_points, 3)

def canonicalise_mean_pose(pose_xyz):
    P = pose_xyz.copy()
    P -= P[0]  # pelvis center
    x_body = P[11] - P[10]    # L→R shoulder vector
    y_body = P[3] - P[0]      # pelvis to neck (up)
    
    # Make skeleton face forward (positive Y direction in plot coordinates)
    # First establish right direction (X)
    x = x_body / (np.linalg.norm(x_body)+1e-12)
    
    # Then up direction (Z), orthogonal to right
    y = y_body - np.dot(x, y_body)*x
    y = y / (np.linalg.norm(y)+1e-12)
    
    # Forward direction (Y) - cross product of right and up
    z = np.cross(x, y)
    
    # Ensure we have a right-handed coordinate system
    y = np.cross(z, x)
    if np.dot(np.cross(x,y),z) < 0: 
        z = -z
    
    # Make skeleton face towards positive Y (front of plot)
    if z[1] < 0:  # if facing backwards
        z = -z
        y = np.cross(z, x)
    
    R = np.stack([x,y,z], axis=1)
    return P @ R

def build_template_with_canonicalisation(trials_flat, n_points):
    means = []
    for arr in trials_flat:
        m = arr.reshape(-1, n_points, 3).mean(axis=0)
        means.append(canonicalise_mean_pose(m))
    template = np.mean(np.stack(means,0),0)
    aligned = []
    for m in means:
        R,s,t = compute_procrustes_transform_3d(template, m)
        aligned.append((m@R)*s + t)
    return np.mean(np.stack(aligned,0),0)

def canonicalise_trial(seq, global_template):
    """
    Canonicalise all frames in a trial to the global template orientation.
    seq: (T, n_points, 3)
    global_template: (n_points, 3) canonical reference skeleton
    """
    # Get mean pose of this trial
    trial_mean = seq.mean(axis=0)

    # Compute rigid transform aligning trial_mean to global template
    R, _, t = compute_procrustes_transform_3d(global_template, trial_mean,
                                              allow_rotation=True, allow_scale=False)

    # Apply same transform to all frames in this trial
    aligned = (seq @ R) + t
    return aligned

def align_yaw_only(X, template, ref_idx, neck_idx):
    """
    Align a 3D skeleton sequence X (n_points x 3) to template:
    - translate pelvis (ref_idx) to origin
    - constrain rotation to yaw (Y-axis) only
    """
    # --- centre pelvis ---
    Xc = X - X[ref_idx]
    Tc = template - template[ref_idx]

    # --- forward direction (pelvis -> neck) projected to XZ plane ---
    vX = (Xc[neck_idx] - Xc[ref_idx]) * np.array([1, 0, 1])
    vT = (Tc[neck_idx] - Tc[ref_idx]) * np.array([1, 0, 1])

    # normalise
    vX /= np.linalg.norm(vX) + 1e-8
    vT /= np.linalg.norm(vT) + 1e-8

    # yaw angle between them
    angle = np.arctan2(vT[0]*vX[2] - vT[2]*vX[0],
                       vT[0]*vX[0] + vT[2]*vX[2])

    R = np.array([[ np.cos(angle), 0, np.sin(angle)],
                  [ 0,             1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])

    return (Xc @ R.T)