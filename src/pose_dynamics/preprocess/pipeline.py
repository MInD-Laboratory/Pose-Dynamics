from __future__ import annotations

from typing import Tuple

import numpy as np
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


def _add_interocular_screen(df: pd.DataFrame) -> pd.DataFrame:
    """Compute inter-ocular distance on screen-normalized coordinates before alignment.

    Matches legacy behavior where distances were scaled using pre-Procrustes, screen-space
    coordinates (landmarks 37 and 46). The resulting series is merged back on time so it can
    be used later for feature scaling without being affected by alignment transforms.
    """

    if not {"trial_id", "keypoint", "x", "y"}.issubset(df.columns):
        return df

    time_col = "time" if "time" in df.columns else "frame"
    if time_col not in df.columns:
        return df

    # Landmarks use OpenPose indexing; match regardless of dtype (string or numeric)
    key_as_str = df["keypoint"].astype(str)
    left_eye = df[key_as_str == "37"]
    right_eye = df[key_as_str == "46"]
    if left_eye.empty or right_eye.empty:
        return df

    left = (
        left_eye.groupby(["trial_id", time_col])[["x", "y"]]
        .mean()
        .rename(columns={"x": "x_left", "y": "y_left"})
    )
    right = (
        right_eye.groupby(["trial_id", time_col])[["x", "y"]]
        .mean()
        .rename(columns={"x": "x_right", "y": "y_right"})
    )

    interocular = left.join(right, how="inner")
    if interocular.empty:
        return df

    dx = interocular["x_left"] - interocular["x_right"]
    dy = interocular["y_left"] - interocular["y_right"]
    interocular["interocular_screen"] = np.sqrt(dx**2 + dy**2)

    io_df = interocular.reset_index()[["trial_id", time_col, "interocular_screen"]]
    df_out = df.merge(io_df, on=["trial_id", time_col], how="left")

    # Keep a copy of the pre-alignment, screen-normalized coordinates so that
    # facial apertures can be measured in the same space as the interocular distance.
    # These columns are preserved through later alignment transforms.
    if "x_screen" not in df_out.columns and "y_screen" not in df_out.columns:
        df_out = df_out.copy()
        df_out["x_screen"] = df_out["x"]
        df_out["y_screen"] = df_out["y"]

    return df_out


def run_pipeline(
    df: pd.DataFrame, recording: dict, cfg: PreprocessConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, list, pd.DataFrame | None]:
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
    df_clean = _add_interocular_screen(df_clean)
    df_clean, transforms, transforms_df = align_procrustes(df_clean, cfg)

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

    return df_clean, windows_scored, qc, transforms, transforms_df
