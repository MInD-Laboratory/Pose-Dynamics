from __future__ import annotations

from typing import Tuple

import pandas as pd

from pose_dynamics.preprocess.alignment.procrustes import align_procrustes
from pose_dynamics.preprocess.confidence import apply_confidence_mask
from pose_dynamics.preprocess.filtering import apply_detrend_filter
from pose_dynamics.preprocess.missing import interpolate_missing
from pose_dynamics.preprocess.normalize import apply_normalization
from pose_dynamics.preprocess.schema import PreprocessConfig
from pose_dynamics.preprocess.selection import apply_selection
from pose_dynamics.preprocess.spatial import apply_spatial
from pose_dynamics.preprocess.timebase import apply_timebase
from pose_dynamics.preprocess.windowing import (
    build_windows,
    ensure_time_column,
    score_windows_missingness,
)


def run_pipeline(
    df: pd.DataFrame, recording: dict, cfg: PreprocessConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, list]:
    """
    Minimal preprocessing pipeline for selection + windowing.

    Returns
    -------
    df_clean : pd.DataFrame
            Selected, dimension-adjusted pose data.
    windows : pd.DataFrame
            Windows table with missingness and drop flags.
    qc : dict
            Basic QC summary.
    """
    df_clean = apply_selection(df, cfg)
    df_clean = apply_timebase(df_clean, cfg, recording)
    df_clean = apply_confidence_mask(df_clean, cfg)
    df_clean = interpolate_missing(df_clean, cfg)
    df_clean = apply_spatial(df_clean, cfg)
    df_clean, transforms = align_procrustes(df_clean, cfg)

    windows = pd.DataFrame()
    if cfg.normalization.enabled and cfg.normalization.scope == "windowed":
        df_for_windows = df_clean
        if cfg.windowing.enabled and cfg.windowing.units == "seconds":
            if "time" not in df_for_windows.columns:
                df_for_windows = ensure_time_column(df_for_windows, cfg, recording)
        windows = build_windows(df_for_windows, cfg, recording)
        df_clean = apply_normalization(df_for_windows, cfg, windows=windows)
    else:
        df_clean = apply_normalization(df_clean, cfg, windows=None)

    df_clean, filter_meta = apply_detrend_filter(df_clean, cfg)

    df_for_windowing = df_clean
    if cfg.windowing.enabled and cfg.windowing.units == "seconds":
        if "time" not in df_for_windowing.columns:
            df_for_windowing = ensure_time_column(df_for_windowing, cfg, recording)

    if windows.empty:
        windows = build_windows(df_for_windowing, cfg, recording)
    windows_scored = score_windows_missingness(df_for_windowing, windows, cfg)

    qc = {
        "n_windows": int(len(windows_scored)),
        "n_dropped": int(windows_scored["dropped"].sum())
        if not windows_scored.empty
        else 0,
        "n_trials": int(df_clean["trial_id"].nunique())
        if "trial_id" in df_clean.columns
        else 0,
        "filtering": filter_meta,
    }

    return df_clean, windows_scored, qc, transforms
