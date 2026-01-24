from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal import detrend as sp_detrend

from pose_dynamics.preprocess.schema import ConfigError, PreprocessConfig


def _median_dt(values: pd.Series) -> float:
    uniq = pd.Series(values.dropna().unique()).sort_values()
    if uniq.size <= 1:
        return float("nan")
    return float(uniq.diff().dropna().median())


def _apply_linear_detrend(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    df_out = df.copy()
    for d in dims:
        y = df_out[d].to_numpy(dtype=float)
        mask = np.isnan(y)
        y_filled = y.copy()
        y_filled[mask] = 0.0
        y_dt = sp_detrend(y_filled, type="linear")
        y_dt[mask] = np.nan
        df_out[d] = y_dt
    return df_out


def _apply_filter(
    df: pd.DataFrame, dims: list[str], btype: str, cutoff_hz: float, dt: float
) -> pd.DataFrame:
    df_out = df.copy()
    fs = 1.0 / dt
    if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        raise ConfigError("invalid cutoff_hz for filter.")

    b, a = butter(N=4, Wn=cutoff_hz / (fs / 2), btype=btype)

    for d in dims:
        y = df_out[d].to_numpy(dtype=float)
        mask = np.isnan(y)
        if mask.any():
            y = y.copy()
            y[mask] = np.nanmean(y)
        y_f = filtfilt(b, a, y)
        y_f[mask] = np.nan
        df_out[d] = y_f
    return df_out


def apply_detrend_filter(
    df: pd.DataFrame, cfg: PreprocessConfig
) -> tuple[pd.DataFrame, list[dict]]:
    if cfg.detrend_filter.detrend == "none" and not cfg.detrend_filter.lowpass.enabled:
        return df, []

    dims = [c for c in ["x", "y", "z"] if c in df.columns]
    if not dims:
        return df, []

    parts = []
    metadata: list[dict] = []
    for trial_id, df_trial in df.groupby("trial_id", sort=False):
        time_col = "time" if "time" in df_trial.columns else "frame"
        dt = _median_dt(df_trial[time_col])

        if cfg.detrend_filter.detrend == "linear":
            df_trial = _apply_linear_detrend(df_trial, dims)
        elif cfg.detrend_filter.detrend == "highpass":
            if not np.isfinite(dt) or dt <= 0:
                raise ConfigError("highpass detrend requires a valid timebase.")
            cutoff_hz = float(cfg.detrend_filter.lowpass.cutoff_hz)
            df_trial = _apply_filter(df_trial, dims, "highpass", cutoff_hz, dt)

        if cfg.detrend_filter.lowpass.enabled:
            if not np.isfinite(dt) or dt <= 0:
                raise ConfigError("lowpass filter requires a valid timebase.")
            cutoff_hz = float(cfg.detrend_filter.lowpass.cutoff_hz)
            df_trial = _apply_filter(df_trial, dims, "lowpass", cutoff_hz, dt)

        parts.append(df_trial)
        metadata.append(
            {
                "trial_id": trial_id,
                "time_col": time_col,
                "dt_median": float(dt) if np.isfinite(dt) else None,
                "detrend": cfg.detrend_filter.detrend,
                "lowpass_enabled": bool(cfg.detrend_filter.lowpass.enabled),
                "lowpass_cutoff_hz": float(cfg.detrend_filter.lowpass.cutoff_hz),
                "lowpass_order": int(cfg.detrend_filter.lowpass.order),
            }
        )

    if not parts:
        return df, metadata
    return pd.concat(parts, ignore_index=True), metadata
