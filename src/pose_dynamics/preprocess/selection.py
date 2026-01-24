from __future__ import annotations

import warnings

import pandas as pd

from pose_dynamics.preprocess.schema import ConfigError, PreprocessConfig


def apply_selection(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    """
    Apply selection rules to canonical long-form pose data.

    Behavior:
    - Filters keypoints by include/exclude rules (unless keep_unselected).
    - Drops z if dims=xy.
    - If dims=xyz but z is missing:
            - require_xyz -> error
            - else -> fallback to xy and emit a warning.
    """
    if "keypoint" not in df.columns:
        raise ConfigError("pose data missing required 'keypoint' column.")
    if "x" not in df.columns or "y" not in df.columns:
        raise ConfigError("pose data missing required 'x'/'y' columns.")

    df_out = df.copy()

    all_kps = sorted(df_out["keypoint"].dropna().unique().tolist())
    if cfg.selection.keypoints == "all":
        selected = set(all_kps)
    else:
        selected = set(cfg.selection.keypoints)

    # Apply exclusions
    selected -= set(cfg.selection.exclude_keypoints)

    if not selected:
        raise ConfigError("selection removed all keypoints; nothing left to process.")

    if not cfg.selection.keep_unselected:
        df_out = df_out[df_out["keypoint"].isin(selected)].copy()

    # Dimension handling
    has_z = "z" in df_out.columns
    if cfg.selection.dims == "xy":
        if has_z:
            df_out = df_out.drop(columns=["z"])
    else:  # xyz
        if not has_z:
            if cfg.selection.require_xyz:
                raise ConfigError("selection.dims='xyz' but 'z' column not found.")
            warnings.warn(
                "selection.dims='xyz' requested but 'z' missing; falling back to 'xy'.",
                RuntimeWarning,
            )

    return df_out
