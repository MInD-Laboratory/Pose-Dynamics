from __future__ import annotations

import numpy as np
import pandas as pd

from pose_dynamics.preprocess.schema import ConfigError, PreprocessConfig


def _median_dt(values: pd.Series) -> float:
    uniq = pd.Series(values.dropna().unique()).sort_values()
    if uniq.size <= 1:
        return float("nan")
    dt = float(uniq.diff().dropna().median())
    return dt


def _max_gap_samples(df_trial: pd.DataFrame, cfg: PreprocessConfig) -> int:
    limit = cfg.missing.interpolation.limit
    if limit.type == "frames":
        return int(limit.max_gap_frames or 0)

    if limit.type == "embedding":
        if limit.embedding is None:
            raise ConfigError("missing.interpolation.limit.embedding required.")
        gap = (limit.embedding.m - 1) * limit.embedding.tau
        if limit.embedding.units == "frames":
            return int(gap)
        # units seconds
        if "time" not in df_trial.columns:
            raise ConfigError("embedding limit in seconds requires 'time' column.")
        dt = _median_dt(df_trial["time"])
        if not np.isfinite(dt) or dt <= 0:
            raise ConfigError("cannot compute dt for embedding seconds limit.")
        return int(np.floor(float(gap) / dt))

    # seconds
    if "time" not in df_trial.columns:
        raise ConfigError("seconds-based interpolation requires 'time' column.")
    dt = _median_dt(df_trial["time"])
    if not np.isfinite(dt) or dt <= 0:
        raise ConfigError("cannot compute dt for seconds-based interpolation.")
    max_gap_s = float(limit.max_gap_s or 0)
    return int(np.floor(max_gap_s / dt))


def interpolate_missing(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    if not cfg.missing.interpolation.enabled:
        return df

    data_cols = [c for c in ["x", "y", "z"] if c in df.columns]
    if not data_cols:
        raise ConfigError("no signal columns found for interpolation.")

    df_out_parts = []
    for trial_id, df_trial in df.groupby("trial_id", sort=False):
        max_gap = _max_gap_samples(df_trial, cfg)
        if max_gap <= 0:
            df_out_parts.append(df_trial)
            continue

        for keypoint, df_kp in df_trial.groupby("keypoint", sort=False):
            df_kp = df_kp.sort_values("time" if "time" in df_kp.columns else "frame")
            df_kp = df_kp.copy()
            for col in data_cols:
                df_kp[col] = df_kp[col].interpolate(
                    method="linear",
                    limit=max_gap,
                    limit_area="inside",
                )
            df_out_parts.append(df_kp)

    if not df_out_parts:
        return df

    return pd.concat(df_out_parts, ignore_index=True)
