"""
RQA utilities for Principal Component analysis.
Functions for running RQA/CRQA on PC time series.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict


def run_crqa_on_pc(
    x: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    filename: str = "CrossRQA"
) -> Tuple[Dict[str, float], int]:
    """
    Perform cross-RQA on two 1D PC time series.

    Parameters:
        x: (T,) array for participant 1
        y: (T,) array for participant 2
        params: Dictionary of RQA parameters
        filename: Identifier for this analysis

    Returns:
        Tuple of (RQA statistics dict, error_code)
    """
    from .rqa.utils import norm_utils, rqa_utils_cpp

    # Normalize
    x_n = norm_utils.normalize_data(x, params['norm'])
    y_n = norm_utils.normalize_data(y, params['norm'])

    # Distance matrix
    ds = rqa_utils_cpp.rqa_dist(x_n, y_n, dim=params['eDim'], lag=params['tLag'])

    # Stats
    td, rs, mats, err_code = rqa_utils_cpp.rqa_stats(
        ds["d"],
        rescale=params['rescaleNorm'],
        rad=params['radius'],
        diag_ignore=0,
        minl=params['minl'],
        rqa_mode="cross"
    )

    return rs, err_code


def run_pc_rqa_analysis(
    pc_scores: List[np.ndarray],
    pair_trials: List[Tuple[str, str]],
    params: Dict[str, Any],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run cross-RQA analysis on all PC pairs for all dyads.

    Parameters:
        pc_scores: List of (T, n_components) arrays, one per participant/trial
        pair_trials: List of (pair_trial_id, party) tuples parallel to pc_scores
        params: RQA parameters dictionary
        verbose: Whether to print progress

    Returns:
        DataFrame with CRQA results for each dyad and PC
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
        pcs1, pcs2 = pc_scores[i1], pc_scores[i2]

        # Run CRQA for each PC
        for pc in range(pcs1.shape[1]):
            x, y = pcs1[:, pc], pcs2[:, pc]

            rs, err = run_crqa_on_pc(x, y, params, filename=f"{pair_trial}_PC{pc+1}")

            if err != 0:
                results.append({'pair_trial': pair_trial, 'pc': pc+1, 'error': err})
                if verbose:
                    print(f"CRQA error for {pair_trial} PC{pc+1}: code {err}")
                continue

            # Store results
            row = {'pair_trial': pair_trial, 'pc': pc+1}
            row.update({k: float(v) for k, v in rs.items()})

            # Add basic statistics
            row['P1_SD'] = float(np.std(x))
            row['P2_SD'] = float(np.std(y))
            row['P1_mean_velocity'] = float(np.mean(np.abs(np.diff(x))))
            row['P2_mean_velocity'] = float(np.mean(np.abs(np.diff(y))))
            row['P1_range'] = float(np.ptp(x))
            row['P2_range'] = float(np.ptp(y))

            results.append(row)

    return pd.DataFrame(results)


def merge_pc_rqa_with_conditions(
    df_crqa: pd.DataFrame,
    conditions_csv: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Merge PC-based CRQA results with experimental conditions.

    Parameters:
        df_crqa: DataFrame with CRQA results (must have 'pair_trial' column)
        conditions_csv: Path to conditions CSV file
        verbose: Whether to print progress

    Returns:
        Merged DataFrame with condition information
    """
    if df_crqa.empty:
        return df_crqa

    # Harmonize column names (pc vs PC)
    if "PC" not in df_crqa and "pc" in df_crqa:
        df_crqa["PC"] = df_crqa["pc"].astype(int)

    # Extract Pair/Trial from 'pair_trial' like 'P001_T3'
    if "pair_trial" not in df_crqa.columns:
        raise ValueError("Expected 'pair_trial' in df_crqa (e.g., 'P001_T3').")

    pt = df_crqa["pair_trial"].str.extract(r"P(\d+)_T(\d+)")
    pt.columns = ["Pair", "Trial"]
    df_crqa["Pair"] = pt["Pair"].astype(int)
    df_crqa["Trial"] = pt["Trial"].astype(int)

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
    merged = df_crqa.merge(cond_long, on=["Pair", "Trial"], how="left")

    if verbose:
        print(f"Merged {len(merged)} rows with conditions")

    return merged
