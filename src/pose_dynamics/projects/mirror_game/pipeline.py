"""
Mirror Game Dataset Module

Dataset-specific configurations, file patterns, and data loading utilities
for the Mirror Game pose dynamics analysis.
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pose_dynamics.preprocessing.signal_cleaning import align_pair
from pose_dynamics.preprocessing.pose_preprocessing import order_xyz_triplets


import pandas as pd
import numpy as np

# ============================================================================
# CONSTANTS AND CONFIGURATIONS
# ============================================================================

# Sampling rate for this dataset
TARGET_RATE = 30.0  # Hz

# File naming pattern for Mirror Game CSV files
# Format: P###_T#_P#_pose_3d.csv (e.g., P001_T10_P1_pose_3d.csv)
MG_FILENAME_PATTERN = re.compile(
    r"^(P\d{3})_(T\d+)_((?:P1|P2))_pose_3d\.csv$", 
    re.IGNORECASE
)

# Keypoint mapping from indices to anatomical labels
# Based on the pose estimation model used (38 keypoints)
KEYPOINT_MAPPING = {
    # Core body (0-4)
    0: 'pelvis',
    1: 'spine_1',
    2: 'spine_2',
    3: 'spine_3',
    4: 'neck',
    
    # Head (5-9)
    5: 'nose',
    6: 'left_eye',
    7: 'right_eye',
    8: 'left_ear',
    9: 'right_ear',
    
    # Shoulders/Clavicles (10-13)
    10: 'left_clavicle',
    11: 'right_clavicle',
    12: 'left_shoulder',
    13: 'right_shoulder',
    
    # Arms (14-17)
    14: 'left_elbow',
    15: 'right_elbow',
    16: 'left_wrist',
    17: 'right_wrist',
    
    # Hips/Legs (18-23)
    18: 'left_hip',
    19: 'right_hip',
    20: 'left_knee',
    21: 'right_knee',
    22: 'left_ankle',
    23: 'right_ankle',
    
    # Feet (24-29)
    24: 'left_big_toe',
    25: 'right_big_toe',
    26: 'left_small_toe',
    27: 'right_small_toe',
    28: 'left_heel',
    29: 'right_heel',
    
    # Left hand fingers (30-33)
    30: 'left_hand_thumb_4',
    31: 'right_hand_thumb_4',
    32: 'left_hand_index_1',
    33: 'right_hand_index_1',
    34: 'left_hand_middle_4',
    35: 'right_hand_middle_4',
    36: 'left_hand_pinky_1',
    37: 'right_hand_pinky_1',
}

SELECTED_KEYPOINTS = list(range(38))  # All keypoints

SKELETON_EDGES = [
    # Spine
    (0, 1), (1, 2), (2, 3), (3, 4),  # Pelvis → Spine → Neck
    
    # Head
    (4, 5),  # Neck → Nose
    (5, 6), (5, 7),  # Nose → Eyes
    (6, 8), (7, 9),  # Eyes → Ears
    
    # Shoulders/Clavicles
    (4, 10), (4, 11),  # Neck → Clavicles
    (10, 12), (11, 13),  # Clavicles → Shoulders
    
    # Left arm
    (12, 14), (14, 16),  # Shoulder → Elbow → Wrist
    
    # Right arm
    (13, 15), (15, 17),  # Shoulder → Elbow → Wrist
    
    # Hips
    (0, 18), (0, 19),  # Pelvis → Hips
    
    # Left leg
    (18, 20), (20, 22),  # Hip → Knee → Ankle
    (22, 24), (22, 26),  # Ankle → Toes
    (22, 28),  # Ankle → Heel
    
    # Right leg
    (19, 21), (21, 23),  # Hip → Knee → Ankle
    (23, 25), (23, 27),  # Ankle → Toes
    (23, 29),  # Ankle → Heel
    
    # Left hand fingers (from wrist)
    (16, 30), (16, 32), (16, 34), (16, 36),
    
    # Right hand fingers (from wrist)
    (17, 31), (17, 33), (17, 35), (17, 37),
]

BODY_REGIONS = {
    'head': {
        'indices': [5, 6, 7, 8, 9],
        'label': 'Head',
        'n_keypoints': 5
    },
    'arms': {
        'indices': [12, 13, 14, 15, 16, 17],
        'label': 'Arms', 
        'n_keypoints': 6
    },
    'legs': {
        'indices': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        'label': 'Legs',
        'n_keypoints': 12
    },
    'torso': {
        'indices': [0, 1, 2, 3, 4],
        'label': 'Torso',
        'n_keypoints': 5
    }
}

# Experimental conditions
CONDITION_ORDER = ["b2b", "uni", "f2f"]
CONDITION_LABELS = {
    "b2b": "Back-to-back (no visual feedback)",
    "uni": "Unidirectional (one-way visual feedback)", 
    "f2f": "Face-to-face (mutual visual feedback)"
}

# RQA Metrics for analysis
CRQA_METRICS = [
    'perc_recur', 'perc_determ', 'laminarity',
    'entropy', 'trapping_time', 'divergence'
]

METRIC_LABELS = {
    "perc_recur": "% Recurrence",
    "perc_determ": "% Determinism",
    "laminarity": "Laminarity",
    "entropy": "Entropy",
    "trapping_time": "Trapping time",
    "divergence": "Divergence",
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def build_file_index(data_dir: Path) -> Tuple[Dict, List[str]]:
    """
    Build an index of Mirror Game CSV files organized by (pair, trial, party).
    
    Parameters
    ----------
    data_dir : Path
        Directory containing Mirror Game CSV files
        
    Returns
    -------
    index : dict
        Nested dict: {(pair_id, trial_id): {'P1': Path, 'P2': Path}}
    unparsed : list
        List of filenames that didn't match the expected pattern
        
    Example
    -------
    >>> index, bad = build_file_index(Path("data/Mirror"))
    >>> print(f"Indexed {len(index)} trials, {len(bad)} unparsed")
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    index = defaultdict(dict)
    unparsed = []
    
    for filepath in csv_files:
        match = MG_FILENAME_PATTERN.match(filepath.name)
        if not match:
            unparsed.append(filepath.name)
            continue
            
        pair_id = match.group(1).upper()
        trial_id = match.group(2).upper()
        party = match.group(3).upper()
        
        index[(pair_id, trial_id)][party] = filepath
    
    return dict(index), unparsed

def resample_ms(df, target_rate=30, dt_col="dt_ms"):
    """Resample using dt_ms increments instead of timestamp_ns."""
    # cumulative sum of ms → seconds
    df["time_s"] = df[dt_col].cumsum() / 1000.0
    df.loc[0, "time_s"] = 0.0  # force first row to 0
    
    # set as index
    df = df.set_index("time_s")
    
    # uniform timeline at target rate
    start, end = df.index.min(), df.index.max()
    new_index = np.arange(start, end, 1/target_rate)
    
    # interpolate onto uniform grid
    df = df.reindex(df.index.union(new_index)).interpolate("linear").loc[new_index]
    
    return df.reset_index().rename(columns={"index": "time_s"})


def load_trial_groups(
    file_index: Dict,
    target_rate: float = TARGET_RATE,
    verbose: bool = True
) -> Dict:
    """
    Load and temporally align all trial pairs from file index.
    
    For each trial with both P1 and P2 files:
    1. Resample to target rate (default 30 Hz)
    2. Find temporal overlap
    3. Truncate to equal length
    4. Extract XYZ coordinate columns
    
    Parameters
    ----------
    file_index : dict
        Output from build_file_index()
    target_rate : float
        Target sampling rate in Hz
    verbose : bool
        Print progress information
        
    Returns
    -------
    trial_groups : dict
        {(pair_id, trial_id): {
            'P1': (filename, dataframe, xyz_columns),
            'P2': (filename, dataframe, xyz_columns)
        }}
    """    
    trial_groups = {}
    n_incomplete = 0
    
    for (pair_id, trial_id), parties in file_index.items():
        # Skip if missing P1 or P2
        if 'P1' not in parties or 'P2' not in parties:
            n_incomplete += 1
            continue
        
        # Load raw data
        f1, f2 = parties['P1'], parties['P2']
        df1_raw = pd.read_csv(f1)
        df2_raw = pd.read_csv(f2)
        
        # Resample to target rate
        df1 = resample_ms(df1_raw, target_rate=target_rate)
        df2 = resample_ms(df2_raw, target_rate=target_rate)
        
        # Temporal alignment
        df1_aligned, df2_aligned = align_pair(df1, df2)
        
        # Get XYZ column orderings
        cols1 = order_xyz_triplets(df1_aligned.columns)
        cols2 = order_xyz_triplets(df2_aligned.columns)
        
        if not cols1 or not cols2:
            n_incomplete += 1
            continue
        
        trial_groups[(pair_id, trial_id)] = {
            'P1': (f1.name, df1_aligned, cols1),
            'P2': (f2.name, df2_aligned, cols2),
        }
    
    if verbose:
        print(f"Loaded {len(trial_groups)} complete trial pairs")
        if n_incomplete > 0:
            print(f"Skipped {n_incomplete} incomplete trials")
    
    return trial_groups


def load_mirror_game_conditions_long(conditions_csv: str) -> pd.DataFrame:
    """
    Load Mirror Game conditions and convert from wide → long.

    Input CSV columns:
      - Pair
      - block1_lead
      - block1_1 ... block1_6
      - block2_1 ... block2_6

    Output columns:
      - Pair       (int)
      - Trial      (int, 1–12)
      - Condition  (str: 'b2b', 'uni', 'f2f', ...)
      - Leader     (str: 'P1' or 'P2', with leader switching between blocks)
    """
    conditions_df = pd.read_csv(conditions_csv)
    conditions_df["Pair"] = conditions_df["Pair"].astype(int)

    # Melt all block*_N columns to long
    value_cols = [
        col for col in conditions_df.columns
        if col.startswith("block") and col != "block1_lead"
    ]

    cond_long = conditions_df.melt(
        id_vars=["Pair", "block1_lead"],
        value_vars=value_cols,
        var_name="block_trial",
        value_name="Condition",
    )

    # Extract block number (1 or 2) and trial index within block (1–6)
    cond_long[["block", "Trial_in_block"]] = cond_long["block_trial"].str.extract(
        r"block(\d)_(\d)"
    ).astype(int)

    # Absolute trial number: 1–12
    cond_long["Trial"] = (cond_long["block"] - 1) * 6 + cond_long["Trial_in_block"]

    # Leader switching rule: block 1 = block1_lead, block 2 = opposite
    def assign_leader(row):
        if row["block"] == 1:
            return row["block1_lead"]
        else:
            return "P1" if row["block1_lead"] == "P2" else "P2"

    cond_long["Leader"] = cond_long.apply(assign_leader, axis=1)

    # Keep clean columns
    cond_long = cond_long[["Pair", "Trial", "Condition", "Leader"]]

    return cond_long


def add_experimental_conditions(
    df: pd.DataFrame,
    conditions_csv: str = "Mirror_Game_Conditions.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge experimental conditions (visual feedback, leader/follower roles)
    with analysis results.

    Expects df with:
      - 'pair_trial' (e.g. 'P001_T10')
      - 'party'      ('P1' or 'P2')
    """
    df = df.copy()

    # Parse Pair (int) and Trial (int) from 'pair_trial' like 'P001_T10'
    df["Pair"] = df["pair_trial"].str.extract(r"P(\d+)")[0].astype(int)
    df["Trial"] = df["pair_trial"].str.extract(r"T(\d+)")[0].astype(int)

    # Load long-format conditions with correct leader switching
    cond_long = load_mirror_game_conditions_long(conditions_csv)

    # Merge on Pair + Trial
    df_merged = df.merge(
        cond_long[["Pair", "Trial", "Condition", "Leader"]],
        on=["Pair", "Trial"],
        how="left",
    )

    # Assign roles based on party vs Leader
    df_merged["Role"] = df_merged.apply(
        lambda row: "Leader" if row["party"] == row["Leader"] else "Follower",
        axis=1,
    )

    if verbose:
        print("\nMerge summary:")
        print(f"  Rows before: {len(df)}")
        print(f"  Rows after:  {len(df_merged)}")
        print(f"  Conditions:  {df_merged['Condition'].unique()}")
        print(f"  Roles:       {df_merged['Role'].unique()}")

    return df_merged


def load_and_align_trials_from_dir(
    data_dir: Path,
    target_rate: float = TARGET_RATE,
    verbose: bool = True,
):
    """
    High-level entry point for the Mirror Game preprocessing notebook.

    Steps:
      1. Build file index from raw CSVs.
      2. Load, resample, and temporally align trial pairs.
      3. Return trial_groups, pair_trial list, and list of bad filenames.
    """
    file_index, bad = build_file_index(data_dir)
    if verbose:
        print(f"Found {len(file_index)} trial entries, {len(bad)} unmatched files")

    trial_groups = load_trial_groups(file_index, target_rate=target_rate, verbose=verbose)
    pair_trials = create_pair_trial_list(trial_groups)

    return trial_groups, pair_trials, bad


# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

def validate_trial_lengths(
    trial_data: List[np.ndarray],
    window_frames: int,
    fps: float = TARGET_RATE,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Validate that trial lengths are compatible with windowing parameters.
    
    Parameters
    ----------
    trial_data : list of arrays
        List of trial arrays to check
    window_frames : int
        Number of frames per window
    fps : float
        Sampling rate in Hz
    verbose : bool
        Print validation summary
        
    Returns
    -------
    stats : dict
        Dictionary with min/max trial lengths in frames and seconds
    """
    lengths = [arr.shape[0] for arr in trial_data]
    min_len = min(lengths)
    max_len = max(lengths)
    
    stats = {
        'min_frames': min_len,
        'max_frames': max_len,
        'min_seconds': min_len / fps,
        'max_seconds': max_len / fps,
        'window_frames': window_frames,
        'window_seconds': window_frames / fps
    }
    
    if verbose:
        print("=" * 80)
        print("TRIAL LENGTH VALIDATION")
        print("=" * 80)
        print(f"Trial length (frames): min={min_len}, max={max_len}")
        print(f"Trial length (sec):    min={min_len/fps:.1f}, max={max_len/fps:.1f}")
        print(f"Window frames:         {window_frames} ({window_frames/fps:.1f}s)")
        
        if min_len < window_frames:
            print(f"\n⚠️  WARNING: Some trials shorter than window size!")
        else:
            print(f"\n✓ All trials longer than window size")
    
    return stats


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_pair_trial_list(trial_groups: Dict) -> List[Tuple[str, str]]:
    """
    Create a list of (pair_trial, party) tuples matching the order of trials.
    
    Parameters
    ----------
    trial_groups : dict
        Output from load_trial_groups()
        
    Returns
    -------
    pair_trials : list of tuples
        [(pair_trial, party), ...] in consistent order with trial processing
    """
    pair_trials = []
    for (pair_id, trial_id), parties in trial_groups.items():
        for party in ('P1', 'P2'):
            pair_trial = f"{pair_id}_{trial_id}"
            pair_trials.append((pair_trial, party))
    return pair_trials


def get_keypoint_labels(indices: List[int]) -> List[str]:
    """
    Get anatomical labels for a list of keypoint indices.

    Parameters
    ----------
    indices : list of int
        Keypoint indices

    Returns
    -------
    labels : list of str
        Corresponding anatomical labels
    """
    return [KEYPOINT_MAPPING[idx] for idx in indices]


# ============================================================================
# 3D POSE PROCESSING - ZED SKELETON
# ============================================================================

# ZED skeleton connections for 3D pose visualization
ZED_SKELETON_CONNECTIONS = {
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
    """
    Build edge list from skeleton connections dictionary.

    Parameters
    ----------
    connections : dict
        Dictionary of skeleton connections by body part
    include : tuple
        Which body parts to include (default: all)

    Returns
    -------
    edges : list of tuples
        List of (i, j) vertex pairs representing skeleton edges
    """
    edges = []
    for k in include:
        for i, j in connections.get(k, []):
            edges.append((int(i), int(j)))
    return edges


# Precomputed edges for ZED skeleton (all body parts)
ZED_EDGES = build_edge_list(ZED_SKELETON_CONNECTIONS, include=("body","legs","face"))



# -----------------------------
# Basic tensor helper
# -----------------------------
def df_to_pose_tensor(df, xyz_cols, n_keypoints=38, dim=3):
    """df[xyz_cols] -> (T, n_keypoints, 3)"""
    arr = df[xyz_cols].to_numpy(dtype=float)
    T, F = arr.shape
    assert F == n_keypoints * dim, f"Expected {n_keypoints*dim} features, got {F}"
    return arr.reshape(T, n_keypoints, dim)

# -----------------------------
# Alignment utilities
# -----------------------------

PELVIS_IDX = 0
LEFT_SHOULDER_IDX = 10
RIGHT_SHOULDER_IDX = 11
NECK_IDX = 3

def canonicalise_mean_pose(pose_xyz):
    """
    Build a body-fixed coordinate frame from pelvis + shoulders + neck,
    and express the mean pose in that frame.
    """
    P = pose_xyz.copy()

    # pelvis to origin
    P -= P[PELVIS_IDX]

    # body axes
    x_body = P[RIGHT_SHOULDER_IDX] - P[LEFT_SHOULDER_IDX]  # left->right
    y_body = P[NECK_IDX] - P[PELVIS_IDX]                   # pelvis->neck (up)

    x = x_body / (np.linalg.norm(x_body) + 1e-12)

    y = y_body - np.dot(x, y_body) * x
    y = y / (np.linalg.norm(y) + 1e-12)

    z = np.cross(x, y)  # forward
    y = np.cross(z, x)  # re-orthogonalise

    # enforce right-handed
    if np.dot(np.cross(x, y), z) < 0:
        z = -z

    # make sure "forward" is roughly positive Y in plot coords
    if z[1] < 0:
        z = -z
        y = np.cross(z, x)

    R = np.stack([x, y, z], axis=1)  # shape (3,3)
    return P @ R  # (n_points, 3)


def compute_procrustes_transform_3d(template, trial_mean,
                                    allow_rotation=True,
                                    allow_scale=False):
    """
    Rigid / similarity Procrustes: align trial_mean to template.
    """
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
            Vt[-1, :] *= -1
            R = U @ Vt
    else:
        R = np.eye(3)

    s = 1.0
    if allow_scale:
        num = np.trace((Mc @ R).T @ Tc)
        den = (Mc ** 2).sum()
        s = float(num / max(den, 1e-12))

    t = T0 - (s * M0 @ R)
    return R, s, t


def build_template_with_canonicalisation(seqs, n_points=38):
    """
    seqs: list of (T, n_points, 3) arrays (already pelvis/centroid centered)
    returns: global template pose (n_points, 3) in canonical body frame
    """
    means = []
    for seq in seqs:
        m = seq.mean(axis=0)             # (n_points, 3)
        means.append(canonicalise_mean_pose(m))

    # rough template
    template0 = np.mean(np.stack(means, axis=0), axis=0)

    # refine with Procrustes alignment of means to template
    aligned_means = []
    for m in means:
        R, s, t = compute_procrustes_transform_3d(template0, m,
                                                  allow_rotation=True,
                                                  allow_scale=False)
        aligned_means.append((m @ R) * s + t)

    template = np.mean(np.stack(aligned_means, axis=0), axis=0)
    return template  # (n_points, 3)


def canonicalise_trial(seq, global_template):
    """
    Align an entire sequence to the global template with one rigid transform.
    seq: (T, n_points, 3)
    global_template: (n_points, 3)
    """
    trial_mean = seq.mean(axis=0)
    R, s, t = compute_procrustes_transform_3d(
        global_template,
        trial_mean,
        allow_rotation=True,
        allow_scale=False,
    )
    aligned = (seq @ R) * s + t
    return aligned

    """
    Align a 3D skeleton sequence to template using only yaw rotation.

    Constrains alignment to rotation around vertical (Y) axis only:
    - Translates pelvis (ref_idx) to origin
    - Rotates around Y-axis to match template orientation

    Parameters
    ----------
    X : ndarray (n_points, 3)
        Skeleton keypoints
    template : ndarray (n_points, 3)
        Template skeleton
    ref_idx : int
        Pelvis keypoint index for centering
    neck_idx : int
        Neck keypoint index for computing forward direction

    Returns
    -------
    aligned : ndarray (n_points, 3)
        Yaw-aligned skeleton
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