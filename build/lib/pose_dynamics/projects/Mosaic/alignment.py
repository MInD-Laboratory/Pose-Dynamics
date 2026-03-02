"""
MOSAIC Alignment Module

Symmetric template building, Procrustes alignment, and limb length constraints
specific to MOSAIC dyadic pose analysis.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from scipy.linalg import orthogonal_procrustes

# ============================================================================
# COLUMN ORDERING
# ============================================================================

def order_xy_pairs(columns: List[str]) -> List[str]:
    """
    Order columns so left/right pairs are grouped consistently.
    Uses skeleton-like pairing logic instead of pure alphabetical.
    
    Parameters
    ----------
    columns : list of str
        Column names (e.g., ['LEye_x_offset', 'LEye_y_offset', ...])
        
    Returns
    -------
    list of str
        Ordered column names with paired keypoints grouped
    """
    ordered = []
    used = set()

    # Extract all base labels (without x/y suffix)
    base_labels = sorted({
        c.rsplit("_", 2)[0] 
        for c in columns 
        if c.endswith(("_x_offset", "_y_offset"))
    })

    for base in base_labels:
        if base in used:
            continue

        # Handle left/right pairs
        if base.startswith("L") or base.startswith("R"):
            side = base[0]
            opposite_side = "R" if side == "L" else "L"
            opposite = opposite_side + base[1:]
            
            # Add both left and right together
            for label in [base, opposite]:
                if label not in used:
                    ordered.extend([f"{label}_x_offset", f"{label}_y_offset"])
                    used.add(label)
        else:
            # Non-paired keypoint
            ordered.extend([f"{base}_x_offset", f"{base}_y_offset"])
            used.add(base)

    return ordered


# ============================================================================
# SYMMETRIC TEMPLATE BUILDING
# ============================================================================

def build_symmetric_template(
    X_raw: pd.DataFrame, 
    expected_cols: List[str], 
    mode: str = "none"
) -> np.ndarray:
    """
    Build a global template with optional symmetrization.
    
    Symmetrization helps create balanced templates for dyadic analysis
    by averaging left/right corresponding keypoints.
    
    Parameters
    ----------
    X_raw : pd.DataFrame
        Concatenated raw keypoint data across participants/sessions
    expected_cols : list of str
        List of column names (x/y offsets)
    mode : str
        Symmetrization mode:
        - "none": return raw mean template (default)
        - "nose": center the nose horizontally
        - "torso": symmetrize shoulders/hips only
        - "full": symmetrize all left/right pairs
    
    Returns
    -------
    np.ndarray
        Template as (n_points, 2) array
        
    Raises
    ------
    ValueError
        If mode is not recognized
    """
    n_points = len(expected_cols) // 2
    
    # Compute mean position for each keypoint
    template = np.array([
        [X_raw[col_x].mean(), X_raw[col_y].mean()]
        for col_x, col_y in zip(expected_cols[::2], expected_cols[1::2])
    ])

    if mode == "none":
        return template

    # Compute midline x-coordinate
    mid_x = (template[:, 0].max() + template[:, 0].min()) / 2

    if mode == "nose":
        # Center nose horizontally
        try:
            nose_idx = next(i for i, col in enumerate(expected_cols[::2]) 
                          if "Nose" in col)
        except StopIteration:
            raise ValueError("Nose keypoint not found for 'nose' mode")
            
        x_shift = mid_x - template[nose_idx, 0]
        template[:, 0] += x_shift
        return template

    if mode == "torso":
        # Symmetrize shoulders and hips
        torso_labels = ["Shoulder", "Hip"]
        for label in torso_labels:
            try:
                l_idx = next(i for i, col in enumerate(expected_cols[::2]) 
                           if f"L{label}" in col)
                r_idx = next(i for i, col in enumerate(expected_cols[::2]) 
                           if f"R{label}" in col)
            except StopIteration:
                continue
                
            # Average left/right positions
            mean_y = (template[l_idx, 1] + template[r_idx, 1]) / 2
            dist_x = abs(template[r_idx, 0] - template[l_idx, 0]) / 2
            
            # Place symmetrically around midline
            template[l_idx] = [mid_x - dist_x, mean_y]
            template[r_idx] = [mid_x + dist_x, mean_y]
            
        return template

    if mode == "full":
        # Symmetrize all left/right pairs
        for i, name in enumerate(expected_cols[::2]):
            if not (name.startswith("L") or name.startswith("R")):
                continue
                
            side = name[0]
            opposite_side = "R" if side == "L" else "L"
            base_name = name[1:]  # Remove L/R prefix
            opposite_name = opposite_side + base_name
            
            try:
                j = next(k for k, col in enumerate(expected_cols[::2]) 
                        if col == opposite_name)
            except StopIteration:
                continue
                
            # Average positions
            mean_y = (template[i, 1] + template[j, 1]) / 2
            dist_x = abs(template[j, 0] - template[i, 0]) / 2
            
            # Place symmetrically
            if side == "L":
                template[i] = [mid_x - dist_x, mean_y]
                template[j] = [mid_x + dist_x, mean_y]
            else:
                template[i] = [mid_x + dist_x, mean_y]
                template[j] = [mid_x - dist_x, mean_y]
                
        return template

    raise ValueError(f"Unknown symmetrization mode: {mode}")


# ============================================================================
# LIMB LENGTH CONSTRAINTS
# ============================================================================

def compute_reference_limb_lengths(
    global_template: np.ndarray, 
    keypoint_names: List[str]
) -> Dict[Tuple[int, int], float]:
    """
    Compute fixed limb lengths from the global template pose.
    
    Used to enforce anatomical constraints during alignment by
    preserving expected limb segment lengths.
    
    Parameters
    ----------
    global_template : np.ndarray
        Global template pose (n_points, 2)
    keypoint_names : list of str
        Column names for keypoints
        
    Returns
    -------
    dict
        Mapping from (point_i, point_j) to reference length
        
    Notes
    -----
    Currently computes lengths for upper limb segments:
    - Shoulder to Elbow
    - Elbow to Wrist
    """
    ref_lengths = {}
    
    for side in ["L", "R"]:
        try:
            # Find indices for shoulder, elbow, wrist
            i_shoulder = next(i for i, col in enumerate(keypoint_names[::2]) 
                            if f"{side}Shoulder" in col)
            i_elbow = next(i for i, col in enumerate(keypoint_names[::2]) 
                          if f"{side}Elbow" in col)
            i_wrist = next(i for i, col in enumerate(keypoint_names[::2]) 
                          if f"{side}Wrist" in col)
        except StopIteration:
            continue

        # Compute segment lengths from template
        ref_lengths[(i_shoulder, i_elbow)] = np.linalg.norm(
            global_template[i_elbow] - global_template[i_shoulder]
        )
        ref_lengths[(i_elbow, i_wrist)] = np.linalg.norm(
            global_template[i_wrist] - global_template[i_elbow]
        )
        
    return ref_lengths


def batch_apply_fixed_lengths(
    poses: np.ndarray, 
    ref_lengths: Dict[Tuple[int, int], float]
) -> np.ndarray:
    """
    Apply fixed limb lengths to pose sequences.
    
    Vectorized limb normalization that preserves joint angles while
    enforcing reference segment lengths.
    
    Parameters
    ----------
    poses : np.ndarray
        Pose array of shape (n_frames, n_points, 2)
    ref_lengths : dict
        Reference lengths from compute_reference_limb_lengths()
        
    Returns
    -------
    np.ndarray
        Pose array with corrected limb lengths
    """
    poses = poses.copy()
    
    for (i1, i2), target_len in ref_lengths.items():
        # Vector from point i1 to point i2
        v = poses[:, i2] - poses[:, i1]  # shape (n_frames, 2)
        
        # Current length
        current_len = np.linalg.norm(v, axis=1, keepdims=True)
        
        # Scale to target length while preserving direction
        scale = target_len / (current_len + 1e-12)
        poses[:, i2] = poses[:, i1] + v * scale
        
    return poses


# ============================================================================
# PROCRUSTES ALIGNMENT
# ============================================================================

def compute_procrustes_transform(
    template: np.ndarray, 
    trial_mean: np.ndarray, 
    allow_rotation: bool = True,
    allow_scale: bool = False
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute Procrustes transform (rotation, scale, translation).
    
    Parameters
    ----------
    template : np.ndarray
        Template pose (n_points, 2)
    trial_mean : np.ndarray
        Trial mean pose (n_points, 2)
    allow_rotation : bool
        Include rotation in transform (default: True)
    allow_scale : bool
        Include scale in transform (default: False)
        
    Returns
    -------
    R : np.ndarray
        Rotation matrix (2, 2)
    scale : float
        Scaling factor (1.0 if allow_scale=False)
    t : np.ndarray
        Translation vector (2,)
    """
    # Center both poses
    template_c = template - template.mean(axis=0)
    trial_c = trial_mean - trial_mean.mean(axis=0)

    # Normalize by Frobenius norm
    norm_template = np.linalg.norm(template_c)
    norm_trial = np.linalg.norm(trial_c)
    template_c /= norm_template
    trial_c /= norm_trial

    # Compute rotation
    if allow_rotation:
        R, _ = orthogonal_procrustes(trial_c, template_c)
    else:
        R = np.eye(2)

    # Compute scale and translation
    if allow_scale:
        scale = norm_template / norm_trial
    else:
        scale = 1.0
    t = template.mean(axis=0) - scale * trial_mean.mean(axis=0) @ R
    
    return R, scale, t


def align_keypoints(
    df: pd.DataFrame, 
    keypoint_names: List[str], 
    reference: str = "Torso",
    template: Optional[np.ndarray] = None, 
    use_procrustes: bool = True,
    allow_rotation: bool = True,
    allow_scale: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Align keypoints for a trial (or window).
    
    Performs reference centering followed by optional Procrustes alignment
    to a global template.
    
    Parameters
    ----------
    df : pd.DataFrame
        Pose data with keypoint columns
    keypoint_names : list of str
        Column names for keypoints
    reference : str
        Reference point for centering: "Torso" or "Nose" (default: "Torso")
    template : np.ndarray or None
        Global template pose (n_points, 2). Required if use_procrustes=True
    use_procrustes : bool
        Apply Procrustes alignment (default: True)
    allow_rotation : bool
        Include rotation in Procrustes (default: True)
    allow_scale : bool
        Include scale in Procrustes (default: False)
        
    Returns
    -------
    aligned_frames : np.ndarray
        Aligned poses (n_frames, n_points*2)
    stats : dict
        Alignment statistics
        
    Raises
    ------
    ValueError
        If data contains NaN/inf values or if template is required but missing
    """
    n_points = len(keypoint_names) // 2
    coords_all = df.values.reshape(len(df), n_points, 2)

    if np.isnan(coords_all).any() or np.isinf(coords_all).any():
        raise ValueError("Input contains NaN or inf values")

    # Step 1: Reference centering (subtract trial/window mean of reference)
    # CRITICAL: Use trial/window mean (not per-frame) to preserve motion variance
    if reference == "Torso":
        # Use midpoint of shoulders and hips
        ref_indices = []
        for label in ["Shoulder", "Hip"]:
            for side in ["L", "R"]:
                try:
                    idx = next(i for i, col in enumerate(keypoint_names[::2]) 
                             if f"{side}{label}" in col)
                    ref_indices.append(idx)
                except StopIteration:
                    pass
        
        if not ref_indices:
            raise ValueError("Torso keypoints not found")
            
        # Mean across entire trial/window (single value per dimension)
        ref_x = coords_all[:, ref_indices, 0].mean()
        ref_y = coords_all[:, ref_indices, 1].mean()
        
    else:  # "Nose"
        try:
            nose_idx = next(i for i, col in enumerate(keypoint_names[::2]) 
                          if "Nose" in col)
        except StopIteration:
            raise ValueError("Nose keypoint not found")
            
        # Mean across entire trial/window (single value per dimension)
        ref_x = coords_all[:, nose_idx, 0].mean()
        ref_y = coords_all[:, nose_idx, 1].mean()

    # Center on reference (subtract same value from all frames)
    coords_all[:, :, 0] -= ref_x
    coords_all[:, :, 1] -= ref_y

    if not use_procrustes:
        return coords_all.reshape(len(df), -1), {}

    if template is None:
        raise ValueError("Template required for Procrustes alignment")

    # Step 2: Procrustes alignment using mean pose of this stack
    trial_mean = coords_all.mean(axis=0)  # (n_points, 2)
    R, scale, t = compute_procrustes_transform(template, trial_mean, allow_rotation, allow_scale)
    
    # Apply transform to all frames
    aligned_frames = np.array([scale * c @ R + t for c in coords_all])

    stats = {
        'rotation_matrix': R,
        'scale': scale,
        'translation': t
    }

    return aligned_frames.reshape(len(df), -1), stats


def rebuild_aligned_dataframe(
    aligned_X: np.ndarray, 
    expected_cols: List[str]
) -> pd.DataFrame:
    """
    Rebuild a DataFrame from aligned pose array.
    
    Parameters
    ----------
    aligned_X : np.ndarray
        Aligned poses of shape (n_frames, n_points*2)
    expected_cols : list of str
        Expected column names [kp1_x, kp1_y, kp2_x, kp2_y, ...]
        
    Returns
    -------
    pd.DataFrame
        DataFrame with aligned keypoint columns in consistent order
    """
    n_points = len(expected_cols) // 2
    poses = aligned_X.reshape(-1, n_points, 2)

    data = {}
    for idx, base_label in enumerate(expected_cols[::2]):
        data[base_label] = poses[:, idx, 0]
        data[expected_cols[2*idx + 1]] = poses[:, idx, 1]
        
    return pd.DataFrame(data)[expected_cols]
