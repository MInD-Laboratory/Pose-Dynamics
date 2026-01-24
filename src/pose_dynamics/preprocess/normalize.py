from __future__ import annotations

import numpy as np
import pandas as pd

from pose_dynamics.preprocess.schema import ConfigError, PreprocessConfig


def _norm_group(df: pd.DataFrame, method: str) -> pd.DataFrame:
    df_out = df.copy()
    dims = [c for c in ["x", "y", "z"] if c in df_out.columns]
    if not dims:
        return df_out

    if method == "zscore":
        for d in dims:
            mean = df_out[d].mean(skipna=True)
            std = df_out[d].std(skipna=True)
            if std == 0 or np.isnan(std):
                std = 1.0
            df_out[d] = (df_out[d] - mean) / std
    elif method == "minmax":
        for d in dims:
            minv = df_out[d].min(skipna=True)
            maxv = df_out[d].max(skipna=True)
            rng = maxv - minv
            if rng == 0 or np.isnan(rng):
                df_out[d] = 0.0
            else:
                df_out[d] = (df_out[d] - minv) / rng
    else:
        raise ConfigError(f"normalization.method '{method}' not supported.")

    return df_out


def apply_normalization(
    df: pd.DataFrame, cfg: PreprocessConfig, windows: pd.DataFrame | None = None
) -> pd.DataFrame:
    if not cfg.normalization.enabled or cfg.normalization.method == "none":
        return df

    method = cfg.normalization.method
    if cfg.normalization.scope == "global_trial":
        parts = []
        for (trial_id, keypoint), g in df.groupby(["trial_id", "keypoint"], sort=False):
            g_norm = _norm_group(g, method)
            parts.append(g_norm)
        return pd.concat(parts, ignore_index=True) if parts else df

    if cfg.normalization.scope != "windowed":
        raise ConfigError("normalization.scope must be 'global_trial' or 'windowed'.")

    if windows is None or windows.empty:
        raise ConfigError("windowed normalization requires non-empty windows.")

    time_col = (
        "time"
        if ("time" in df.columns and (windows["units"] == "seconds").any())
        else "frame"
    )
    if time_col not in df.columns:
        raise ConfigError("windowed normalization requires a time or frame column.")

    df_out_parts = []
    for _, w in windows.iterrows():
        trial_id = w["trial_id"]
        s = float(w["start"])
        e = float(w["end"])

        df_trial = df[df["trial_id"] == trial_id]
        if w["units"] == "seconds":
            mask = (df_trial[time_col] >= s) & (df_trial[time_col] < e)
        else:
            mask = (df_trial[time_col] >= s) & (df_trial[time_col] < e)

        df_win = df_trial.loc[mask]
        if df_win.empty:
            continue

        for keypoint, g in df_win.groupby("keypoint", sort=False):
            df_out_parts.append(_norm_group(g, method))

    if not df_out_parts:
        return df

    df_out = pd.concat(df_out_parts, ignore_index=True)
    # Merge normalized rows back into original df to keep non-windowed rows intact
    key_cols = ["trial_id", "keypoint"]
    if "time" in df.columns:
        key_cols.append("time")
    elif "frame" in df.columns:
        key_cols.append("frame")

    df_keep = df.copy()
    df_keep = df_keep.set_index(key_cols)
    df_out = df_out.set_index(key_cols)

    for col in df_out.columns:
        if col in df_keep.columns:
            df_keep.loc[df_out.index, col] = df_out[col]

    df_keep = df_keep.reset_index()
    return df_keep
