"""
Multivariate RQA utilities for keypoint analysis.
Functions for running mdRQA on pose keypoint data with various options.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict


def extract_keypoint_data(
    aligned_array: np.ndarray,
    mode: str = 'all',
    keypoint_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Extract keypoint data from aligned array based on mode.

    Parameters:
        aligned_array: (T, n_points*3) array of aligned keypoints
        mode: Extraction mode - 'all', 'xy_only', or 'subset'
        keypoint_indices: List of keypoint indices if mode='subset'

    Returns:
        (T, dimensions) array of extracted keypoint data

    Modes:
        - 'all': Use all keypoints with all 3 dimensions (x,y,z)
        - 'xy_only': Use all keypoints but only x,y coordinates (ignore z)
        - 'subset': Use only specified keypoint indices with all 3 dimensions
    """
    T, total_dims = aligned_array.shape
    n_points = total_dims // 3

    # Reshape to (T, n_points, 3)
    reshaped = aligned_array.reshape(T, n_points, 3)

    if mode == 'all':
        # Use all keypoints, all dimensions
        return aligned_array

    elif mode == 'xy_only':
        # Use only x,y coordinates
        xy_data = reshaped[:, :, :2]  # (T, n_points, 2)
        return xy_data.reshape(T, -1)

    elif mode == 'subset':
        if keypoint_indices is None:
            raise ValueError("keypoint_indices must be provided for 'subset' mode")
        # Extract specific keypoints
        subset = reshaped[:, keypoint_indices, :]  # (T, n_selected, 3)
        return subset.reshape(T, -1)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'all', 'xy_only', or 'subset'")


def run_mdrqa_on_data(
    data1: np.ndarray,
    data2: Optional[np.ndarray] = None,
    params: Dict[str, Any] = None,
    mode: str = "cross",
    filename: str = "mdRQA"
) -> Tuple[Dict[str, float], int]:
    """
    Perform multivariate RQA on keypoint data.

    Parameters:
        data1: (T, dimensions) array for participant 1
        data2: (T, dimensions) array for participant 2 (required for cross/joint modes)
        params: Dictionary of RQA parameters
        mode: Type of analysis - 'auto', 'cross', or 'joint'
            - 'auto': Auto-RQA on data1 only
            - 'cross': Cross-RQA between data1 and data2
            - 'joint': Joint RQA on concatenated [data1, data2]
        filename: Identifier for this analysis (for logging/debugging)

    Returns:
        Tuple of (RQA statistics dict, error_code)
        error_code is 0 for success, non-zero for failure
    """
    from .rqa.multivariateRQA import multivariateRQA, multivariateCrossRQA

    if params is None:
        params = {}

    try:
        if mode == "auto":
            # Auto-RQA on data1 only
            td, rs, mats, err_code = multivariateRQA(data1, params, mode="auto")

        elif mode == "cross":
            # Cross-RQA between data1 and data2
            if data2 is None:
                raise ValueError("data2 is required for cross mode")
            td, rs, mats, err_code = multivariateCrossRQA(data1, data2, params)

        elif mode == "joint":
            # Joint RQA on concatenated data
            if data2 is None:
                raise ValueError("data2 is required for joint mode")
            # Concatenate along dimension axis
            joint_data = np.hstack([data1, data2])
            td, rs, mats, err_code = multivariateRQA(joint_data, params, mode="auto")

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'auto', 'cross', or 'joint'")

        return rs, err_code

    except Exception as e:
        print(f"Error in mdRQA for {filename} (mode={mode}): {e}")
        import traceback
        traceback.print_exc()
        return {}, -1


def compute_movement_stats(data: np.ndarray) -> Dict[str, float]:
    """
    Compute basic movement statistics for keypoint data.

    Parameters:
        data: (T, dimensions) array

    Returns:
        Dictionary with movement statistics
    """
    # Compute frame-to-frame displacement (Euclidean norm across all dimensions)
    diff = np.diff(data, axis=0)  # (T-1, dims)
    displacement = np.linalg.norm(diff, axis=1)  # (T-1,)

    return {
        'mean_movement': float(np.mean(displacement)),
        'std_movement': float(np.std(displacement)),
        'max_movement': float(np.max(displacement)),
        'total_movement': float(np.sum(displacement))
    }


def run_mdrqa_analysis(
    all_aligned: List[np.ndarray],
    pair_trials: List[Tuple[str, str]],
    params: Dict[str, Any],
    keypoint_mode: str = 'all',
    keypoint_indices: Optional[List[int]] = None,
    rqa_mode: str = 'cross',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run multivariate RQA analysis on all dyadic pairs.

    Parameters:
        all_aligned: List of (T, n_points*3) arrays, one per participant/trial
        pair_trials: List of (pair_trial_id, party) tuples parallel to all_aligned
        params: RQA parameters dictionary
        keypoint_mode: Keypoint extraction mode ('all', 'xy_only', or 'subset')
        keypoint_indices: List of keypoint indices if keypoint_mode='subset'
        rqa_mode: Type of RQA analysis ('auto', 'cross', or 'joint')
            - 'auto': Run auto-RQA on P1 and P2 separately (2 analyses per dyad)
            - 'cross': Run cross-RQA between P1 and P2 (1 analysis per dyad)
            - 'joint': Run joint RQA on concatenated P1+P2 data (1 analysis per dyad)
        verbose: Whether to print progress

    Returns:
        DataFrame with mdRQA results for each dyad
    """
    # Build mapping from pair_trial to indices
    pair_to_indices = defaultdict(dict)
    for idx, (pair_trial, party) in enumerate(pair_trials):
        pair_to_indices[pair_trial][party] = idx

    results = []

    for pair_trial, pmap in pair_to_indices.items():
        if 'P1' not in pmap or 'P2' not in pmap:
            continue

        i1, i2 = pmap['P1'], pmap['P2']
        aligned1 = all_aligned[i1]  # (T, n_points*3)
        aligned2 = all_aligned[i2]  # (T, n_points*3)

        # Extract keypoint data based on mode
        try:
            data1 = extract_keypoint_data(aligned1, mode=keypoint_mode, keypoint_indices=keypoint_indices)
            data2 = extract_keypoint_data(aligned2, mode=keypoint_mode, keypoint_indices=keypoint_indices)
        except Exception as e:
            if verbose:
                print(f"Error extracting keypoints for {pair_trial}: {e}")
            continue

        if rqa_mode == 'auto':
            # Run auto-RQA on P1 and P2 separately
            for party, data in [('P1', data1), ('P2', data2)]:
                rs, err = run_mdrqa_on_data(
                    data, None, params, mode='auto',
                    filename=f"{pair_trial}_{party}_mdRQA"
                )

                if err != 0:
                    results.append({
                        'pair_trial': pair_trial,
                        'party': party,
                        'error': err
                    })
                    if verbose:
                        print(f"mdRQA error for {pair_trial} {party}: code {err}")
                    continue

                # Store results
                row = {
                    'pair_trial': pair_trial,
                    'party': party,
                    'rqa_mode': 'auto',
                    'keypoint_mode': keypoint_mode,
                    'n_dims': data.shape[1]
                }
                row.update({k: float(v) for k, v in rs.items()})

                # Add movement statistics
                stats = compute_movement_stats(data)
                row.update({f'{party}_{k}': v for k, v in stats.items()})

                results.append(row)

        else:
            # Run cross or joint RQA
            rs, err = run_mdrqa_on_data(
                data1, data2, params, mode=rqa_mode,
                filename=f"{pair_trial}_mdRQA"
            )

            if err != 0:
                results.append({
                    'pair_trial': pair_trial,
                    'rqa_mode': rqa_mode,
                    'error': err
                })
                if verbose:
                    print(f"mdRQA error for {pair_trial}: code {err}")
                continue

            # Store results
            row = {
                'pair_trial': pair_trial,
                'rqa_mode': rqa_mode,
                'keypoint_mode': keypoint_mode,
                'n_dims': data1.shape[1] if rqa_mode == 'cross' else data1.shape[1] + data2.shape[1]
            }
            row.update({k: float(v) for k, v in rs.items()})

            # Add movement statistics
            stats1 = compute_movement_stats(data1)
            stats2 = compute_movement_stats(data2)
            row.update({f'P1_{k}': v for k, v in stats1.items()})
            row.update({f'P2_{k}': v for k, v in stats2.items()})

            results.append(row)

    return pd.DataFrame(results)


def merge_with_conditions(
    df_mdrqa: pd.DataFrame,
    conditions_csv: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Merge mdRQA results with experimental conditions.

    Parameters:
        df_mdrqa: DataFrame with mdRQA results (must have 'pair_trial' column)
        conditions_csv: Path to conditions CSV file
        verbose: Whether to print progress

    Returns:
        Merged DataFrame with condition information
    """
    if df_mdrqa.empty:
        return df_mdrqa

    # Extract Pair/Trial from 'pair_trial' like 'P001_T3'
    pt = df_mdrqa["pair_trial"].str.extract(r"P(\d+)_T(\d+)")
    pt.columns = ["Pair", "Trial"]
    df_mdrqa["Pair"] = pt["Pair"].astype(int)
    df_mdrqa["Trial"] = pt["Trial"].astype(int)

    # Load conditions
    conditions_df = pd.read_csv(conditions_csv)

    # Conditions: wide -> long
    cond_long = conditions_df.melt(
        id_vars=["Pair", "block1_lead"],
        value_vars=[c for c in conditions_df.columns if c.startswith("block")],
        var_name="block_trial",
        value_name="Condition"
    )

    bt = cond_long["block_trial"].str.extract(r"block(\d)_(\d)")
    bt.columns = ["block", "Trial_in_block"]
    cond_long["block"] = bt["block"].astype(int)
    cond_long["Trial_in_block"] = bt["Trial_in_block"].astype(int)
    cond_long["Trial"] = (cond_long["block"] - 1) * 6 + cond_long["Trial_in_block"]

    def assign_leader(row):
        if row["block"] == 1:
            return row["block1_lead"]
        return "P1" if row["block1_lead"] == "P2" else "P2"

    cond_long["Leader"] = cond_long.apply(assign_leader, axis=1)
    cond_long = cond_long[["Pair", "Trial", "Condition", "Leader"]]

    # Merge
    merged = df_mdrqa.merge(cond_long, on=["Pair", "Trial"], how="left")

    if verbose:
        print(f"Merged {len(merged)} rows with conditions")

    return merged
