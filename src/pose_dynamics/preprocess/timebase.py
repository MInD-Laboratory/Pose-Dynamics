from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pose_dynamics.preprocess.schema import ConfigError, PreprocessConfig


def ensure_time_column(
    df: pd.DataFrame, cfg: PreprocessConfig, recording: dict
) -> pd.DataFrame:
    if "time" in df.columns:
        return df
    if "frame" not in df.columns:
        raise ConfigError("pose data missing both 'time' and 'frame' columns.")

    if not cfg.timebase.enforce_time:
        raise ConfigError("timebase.enforce_time=false but no 'time' column exists.")

    fps_by_trial = {
        t.get("trial_id"): t.get("fps_used") for t in recording.get("trials", [])
    }
    fps_series = df["trial_id"].map(fps_by_trial)
    if fps_series.isna().any():
        raise ConfigError("missing fps metadata for some trials; cannot derive time.")
    if (fps_series <= 0).any():
        raise ConfigError("fps metadata must be > 0 to derive time from frame.")

    df_out = df.copy()
    df_out["time"] = df_out["frame"].astype(float) / fps_series.astype(float)
    return df_out


def resample_timebase(
    df: pd.DataFrame, cfg: PreprocessConfig, recording: dict
) -> pd.DataFrame:
    if not cfg.timebase.resample.enabled:
        return df

    if "time" not in df.columns:
        df = ensure_time_column(df, cfg, recording)

    target_hz = float(cfg.timebase.resample.target_hz)
    if target_hz <= 0:
        raise ConfigError("timebase.resample.target_hz must be > 0.")

    dt = 1.0 / target_hz
    if dt <= 0:
        raise ConfigError("invalid resample target dt.")

    data_cols = [c for c in ["x", "y", "z", "conf"] if c in df.columns]
    if not data_cols:
        raise ConfigError("no signal columns found to resample.")

    fps_by_trial = {
        t.get("trial_id"): t.get("fps_used") for t in recording.get("trials", [])
    }

    parts = []
    for (trial_id, keypoint), g in df.groupby(["trial_id", "keypoint"], sort=False):
        g = g.sort_values("time")
        t_min = float(g["time"].min())
        t_max = float(g["time"].max())
        if np.isnan(t_min) or np.isnan(t_max):
            continue

        grid = np.arange(t_min, t_max + dt * 0.5, dt)
        g_idx = g.set_index("time")
        g_resamp = g_idx.reindex(grid)
        g_resamp[data_cols] = g_resamp[data_cols].interpolate(
            method="linear", limit_area="inside"
        )

        g_resamp["trial_id"] = trial_id
        g_resamp["keypoint"] = keypoint
        g_resamp["source_file"] = (
            g["source_file"].iloc[0] if "source_file" in g.columns else None
        )
        g_resamp = g_resamp.reset_index().rename(columns={"index": "time"})

        if "frame" in df.columns:
            fps = fps_by_trial.get(trial_id, None)
            if fps is None:
                warnings.warn(
                    "frame column present but fps metadata missing; frame will be NaN after resample.",
                    RuntimeWarning,
                )
                g_resamp["frame"] = np.nan
            else:
                g_resamp["frame"] = (
                    (g_resamp["time"] * float(fps)).round().astype("Int64")
                )

        parts.append(
            g_resamp[
                ["trial_id", "source_file", "time", "frame", "keypoint", *data_cols]
            ]
            if "frame" in g_resamp.columns
            else g_resamp[["trial_id", "source_file", "time", "keypoint", *data_cols]]
        )

    if not parts:
        raise ConfigError("resampling produced no data.")

    df_out = pd.concat(parts, ignore_index=True)
    return df_out


def apply_timebase(
    df: pd.DataFrame, cfg: PreprocessConfig, recording: dict
) -> pd.DataFrame:
    if cfg.timebase.enforce_time:
        df = ensure_time_column(df, cfg, recording)
    if cfg.timebase.resample.enabled:
        df = resample_timebase(df, cfg, recording)
    return df
