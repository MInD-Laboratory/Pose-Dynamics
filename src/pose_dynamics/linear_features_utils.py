"""
Linear features extraction utilities for pose-based movement analysis.

Functions for extracting kinematic features (displacement, velocity, acceleration)
from pose data, windowing, and aggregating across time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def extract_kinematics_per_frame(coords_3d, fps=30):
    """
    Extract displacement, velocity, and acceleration per keypoint per frame.
    
    Parameters:
        coords_3d: (T, n_keypoints, 3) array of 3D coordinates
        fps: sampling rate in Hz
        
    Returns:
        kinematics: dict with 'displacement', 'velocity', 'acceleration' arrays
    """
    dt = 1.0 / fps
    
    # Frame-to-frame displacement (Euclidean distance)
    delta_pos = np.diff(coords_3d, axis=0)  # (T-1, n_keypoints, 3)
    displacement = np.linalg.norm(delta_pos, axis=2)  # (T-1, n_keypoints)
    
    # Velocity (displacement per second)
    velocity = displacement / dt  # (T-1, n_keypoints) in m/s
    
    # Acceleration (change in velocity per second)
    delta_velocity = np.diff(velocity, axis=0)  # (T-2, n_keypoints)
    acceleration = delta_velocity / dt  # (T-2, n_keypoints) in m/s²
    
    return {
        'displacement': displacement,
        'velocity': velocity,
        'acceleration': acceleration
    }


def create_windows(data, window_frames, step_frames):
    """
    Divide time series into overlapping windows.
    
    Parameters:
        data: (T, n_keypoints) array
        window_frames: window size in frames
        step_frames: step size in frames (window_frames * 0.5 for 50% overlap)
        
    Returns:
        windows: list of arrays, each (window_frames, n_keypoints)
        window_info: list of dicts with start/end frame indices
    """
    n_frames = data.shape[0]
    windows = []
    window_info = []
    
    start = 0
    while start + window_frames <= n_frames:
        end = start + window_frames
        windows.append(data[start:end])
        window_info.append({'start_frame': start, 'end_frame': end})
        start += step_frames
    
    return windows, window_info


def aggregate_window_features(window_data, keypoint_indices):
    """
    Aggregate kinematic features for a subset of keypoints over one window.
    
    Parameters:
        window_data: (window_frames, n_keypoints) array for one kinematic feature
        keypoint_indices: list of keypoint indices to include
        
    Returns:
        features: dict with 'mean' and 'rms' aggregations
    """
    # Extract subset of keypoints
    subset = window_data[:, keypoint_indices]
    
    # Aggregate across both time and keypoints
    features = {
        'mean': np.mean(subset),
        'rms': np.sqrt(np.mean(subset**2))
    }
    
    return features


def process_trial_linear_features(coords_3d, keypoint_groups, window_frames, 
                                  step_frames, fps=30):
    """
    Extract windowed linear features for one trial.
    
    Parameters:
        coords_3d: (T, n_keypoints, 3) array of coordinates
        keypoint_groups: dict mapping group names to keypoint indices
        window_frames: window size in frames
        step_frames: step size in frames
        fps: sampling rate in Hz
        
    Returns:
        features_df: DataFrame with features per window per body region
    """
    # Extract kinematics
    kinematics = extract_kinematics_per_frame(coords_3d, fps=fps)
    
    # Create windows for each kinematic feature
    # Use displacement (longest common length)
    disp_windows, window_info = create_windows(
        kinematics['displacement'], window_frames, step_frames
    )
    vel_windows, _ = create_windows(
        kinematics['velocity'], window_frames, step_frames
    )
    
    # Acceleration is 1 frame shorter, adjust
    acc_data = kinematics['acceleration']
    if len(acc_data) >= window_frames:
        acc_windows, _ = create_windows(acc_data, window_frames, step_frames)
    else:
        acc_windows = []
    
    # Extract features per window per body region
    rows = []
    for w_idx in range(len(disp_windows)):
        for region_name, region_info in keypoint_groups.items():
            kpt_indices = region_info['indices']
            
            # Displacement features
            disp_feats = aggregate_window_features(disp_windows[w_idx], kpt_indices)
            
            # Velocity features
            vel_feats = aggregate_window_features(vel_windows[w_idx], kpt_indices)
            
            # Acceleration features (if available)
            if w_idx < len(acc_windows):
                acc_feats = aggregate_window_features(acc_windows[w_idx], kpt_indices)
            else:
                acc_feats = {'mean': np.nan, 'rms': np.nan}
            
            row = {
                'window_index': w_idx,
                'start_frame': window_info[w_idx]['start_frame'],
                'end_frame': window_info[w_idx]['end_frame'],
                'start_time_sec': window_info[w_idx]['start_frame'] / fps,
                'end_time_sec': window_info[w_idx]['end_frame'] / fps,
                'body_region': region_name,
                'displacement_mean': disp_feats['mean'],
                'displacement_rms': disp_feats['rms'],
                'velocity_mean': vel_feats['mean'],
                'velocity_rms': vel_feats['rms'],
                'acceleration_mean': acc_feats['mean'],
                'acceleration_rms': acc_feats['rms'],
            }
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def process_all_trials_linear_features(filtered_trials, pair_trials, keypoint_groups,
                                       window_frames, step_frames, fps=30,
                                       verbose=True):
    """
    Extract linear features for all trials.
    
    Parameters:
        filtered_trials: list of (T, n_keypoints*3) filtered coordinate arrays
        pair_trials: list of (pair_trial, party) tuples parallel to filtered_trials
        keypoint_groups: dict of body region definitions
        window_frames: window size in frames
        step_frames: step size in frames
        fps: sampling rate in Hz
        verbose: print progress
        
    Returns:
        all_features_df: DataFrame with features for all trials
    """
    all_features = []
    
    for trial_idx, (filtered_flat, (pair_trial, party)) in enumerate(
        zip(filtered_trials, pair_trials)
    ):
        if verbose:
            print(f"Processing {pair_trial}_{party} ({trial_idx+1}/{len(filtered_trials)})")
        
        # Reshape to 3D
        n_frames = filtered_flat.shape[0]
        n_keypoints = filtered_flat.shape[1] // 3
        coords_3d = filtered_flat.reshape(n_frames, n_keypoints, 3)
        
        # Extract features for this trial
        trial_features = process_trial_linear_features(
            coords_3d, keypoint_groups, window_frames, step_frames, fps
        )
        
        # Add trial identifiers
        trial_features['pair_trial'] = pair_trial
        trial_features['party'] = party
        trial_features['trial_index'] = trial_idx
        
        all_features.append(trial_features)
    
    # Concatenate all trials
    all_features_df = pd.concat(all_features, ignore_index=True)
    
    if verbose:
        print(f"\nExtracted features from {len(filtered_trials)} trials")
        print(f"Total windows: {len(all_features_df)}")
        print(f"Features per window: {len(all_features_df.columns) - 7}")  # Exclude identifiers
    
    return all_features_df


def merge_with_conditions(features_df, conditions_csv, verbose=True):
    """
    Merge linear features with experimental conditions and roles.

    Parameters:
        features_df: DataFrame from process_all_trials_linear_features()
        conditions_csv: path to conditions CSV file
                       Supports two formats:
                       1. Trial1_Condition, Trial2_Condition, ..., block1_lead, block2_lead
                       2. block1_1, ..., block1_6, block2_1, ..., block2_6, block1_lead
        verbose: print merge statistics

    Returns:
        merged_df: DataFrame with Condition, Leader, and Role columns added

    Notes:
        - Trial numbering: block1_1 through block1_6 are Trials 1-6,
          block2_1 through block2_6 are Trials 7-12
        - If block2_lead is not present, assumes leader switches between blocks
          (if P1 leads block1, then P2 leads block2)
    """
    # Work on a copy to avoid modifying the input
    features_df = features_df.copy()

    # Parse pair and trial from pair_trial string
    pt = features_df['pair_trial'].str.extract(r'P(\d+)_T(\d+)')
    pt.columns = ['Pair', 'Trial']
    features_df['Pair'] = pt['Pair'].astype(int)
    features_df['Trial'] = pt['Trial'].astype(int)
    
    # Load conditions file
    conditions = pd.read_csv(conditions_csv)

    # Melt conditions from wide to long format
    # Support two formats:
    # 1. Trial1_Condition, Trial2_Condition, ...
    # 2. block1_1, block1_2, ..., block2_1, block2_2, ...
    trial_cols = [c for c in conditions.columns if 'Trial' in c and 'Condition' in c]

    if not trial_cols:
        # Use block format: block1_1, block1_2, etc.
        # Exclude leader columns
        trial_cols = [c for c in conditions.columns
                      if c.startswith('block') and 'lead' not in c]

    cond_long = conditions.melt(
        id_vars=['Pair'],
        value_vars=trial_cols,
        var_name='trial_col',
        value_name='Condition'
    )

    # Extract trial number
    # Handle both "Trial3_Condition" and "block1_3" formats
    if 'Trial' in cond_long['trial_col'].iloc[0]:
        cond_long['Trial'] = cond_long['trial_col'].str.extract(r'Trial(\d+)').astype(int)
    else:
        # Extract trial from block format
        # block1_3 -> block=1, trial_in_block=3 -> Trial = (1-1)*6 + 3 = 3
        # block2_1 -> block=2, trial_in_block=1 -> Trial = (2-1)*6 + 1 = 7
        trial_nums = cond_long['trial_col'].str.extract(r'block(\d+)_(\d+)')
        trial_nums.columns = ['block', 'trial_in_block']
        cond_long['Trial'] = ((trial_nums['block'].astype(int) - 1) * 6 +
                               trial_nums['trial_in_block'].astype(int))

    cond_long = cond_long[['Pair', 'Trial', 'Condition']].dropna()
    
    # Merge leader information
    # Determine leader based on block
    leader_info = []
    for _, row in conditions.iterrows():
        pair = row['Pair']
        for trial in range(1, 13):  # 12 trials (6 per block)
            if trial <= 6:  # Block 1
                block1_leader = row.get('block1_lead', 'P1')
                leader = block1_leader
            else:  # Block 2
                # If block2_lead exists, use it; otherwise swap from block1
                if 'block2_lead' in row:
                    leader = row.get('block2_lead')
                else:
                    # Assume leader switches in block 2
                    block1_leader = row.get('block1_lead', 'P1')
                    leader = 'P2' if block1_leader == 'P1' else 'P1'
            leader_info.append({'Pair': pair, 'Trial': trial, 'Leader': leader})
    
    leader_df = pd.DataFrame(leader_info)
    cond_long = cond_long.merge(leader_df, on=['Pair', 'Trial'], how='left')
    
    # Merge with features
    merged = features_df.merge(cond_long, on=['Pair', 'Trial'], how='left')
    
    # Assign role
    merged['Role'] = np.where(
        merged['party'] == merged['Leader'],
        'Leader',
        'Follower'
    )
    
    if verbose:
        print(f"\nMerge summary:")
        print(f"  Rows before: {len(features_df)}")
        print(f"  Rows after:  {len(merged)}")
        print(f"  Conditions:  {merged['Condition'].unique()}")
        print(f"  Roles:       {merged['Role'].unique()}")
        n_missing = merged['Condition'].isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {n_missing} rows with missing condition")
    
    return merged


def collapse_windows_for_stats(merged_df, verbose=True):
    """
    Collapse windows within (Pair, Trial, Condition, Role, body_region).
    Takes mean across windows to prepare for statistical analysis.
    
    Parameters:
        merged_df: DataFrame with all windowed features
        verbose: print collapse statistics
        
    Returns:
        collapsed_df: DataFrame with one row per (Pair, Trial, Condition, Role, body_region)
    """
    groupby_cols = ['Pair', 'Trial', 'Condition', 'Role', 'body_region']
    
    feature_cols = [
        'displacement_mean', 'displacement_rms',
        'velocity_mean', 'velocity_rms',
        'acceleration_mean', 'acceleration_rms'
    ]
    
    collapsed = merged_df.groupby(groupby_cols)[feature_cols].mean().reset_index()
    
    if verbose:
        print(f"\nCollapsed windows:")
        print(f"  Original rows:  {len(merged_df)}")
        print(f"  Collapsed rows: {len(collapsed)}")
        print(f"  Average windows per group: {len(merged_df) / len(collapsed):.1f}")
    
    return collapsed