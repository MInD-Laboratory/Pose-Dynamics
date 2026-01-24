from __future__ import annotations

from typing import Iterable

import pandas as pd

from pose_dynamics.preprocess.schema import ConfigError, PreprocessConfig
from pose_dynamics.preprocess.timebase import ensure_time_column


def _get_qc_keypoints(df: pd.DataFrame, cfg: PreprocessConfig) -> Iterable[str]:
    if cfg.windowing.qc_keypoints == "all":
        return df["keypoint"].dropna().unique().tolist()
    return [k for k in cfg.windowing.qc_keypoints if k in set(df["keypoint"])]


def build_windows(
    df: pd.DataFrame, cfg: PreprocessConfig, recording: dict
) -> pd.DataFrame:
    """Return windows table with (trial_id, window_id, start/end, n_samples)."""
    if not cfg.windowing.enabled:
        return pd.DataFrame(
            columns=["trial_id", "window_id", "start", "end", "n_samples", "units"]
        )

    df_use = df
    if cfg.windowing.units == "seconds":
        df_use = ensure_time_column(df_use, cfg, recording)
        time_col = "time"
        length = cfg.windowing.length_s
        step = cfg.windowing.step_s
    else:
        time_col = "frame"
        if time_col not in df_use.columns:
            raise ConfigError("windowing.units='frames' requires 'frame' column.")
        length = cfg.windowing.length_frames
        step = cfg.windowing.step_frames

    windows = []
    for trial_id, df_trial in df_use.groupby("trial_id", sort=False):
        t_vals = pd.to_numeric(df_trial[time_col], errors="coerce")
        t_min = float(t_vals.min())
        t_max = float(t_vals.max())

        # Extend max by one sample interval so full windows can include the last sample.
        uniq = pd.Series(t_vals.dropna().unique()).sort_values()
        if uniq.size > 1:
            dt = float(uniq.diff().dropna().median())
            if dt > 0:
                t_max = t_max + dt

        if cfg.windowing.units == "seconds":
            t_min += cfg.windowing.trim_edges_s.start
            t_max -= cfg.windowing.trim_edges_s.end
            if t_max <= t_min:
                continue

        starts: list[float] = []
        if cfg.windowing.include_partial:
            s = t_min
            while s < t_max:
                starts.append(float(s))
                s += step
        else:
            end_limit = t_max - length
            s = t_min
            while s <= end_limit + 1e-12:
                starts.append(float(s))
                s += step

        for i, s in enumerate(starts):
            e = s + length
            if cfg.windowing.include_partial:
                e = min(e, t_max)

            if cfg.windowing.include_partial:
                mask = (df_trial[time_col] >= s) & (df_trial[time_col] <= e)
            else:
                mask = (df_trial[time_col] >= s) & (df_trial[time_col] < e)

            n_samples = int(pd.unique(df_trial.loc[mask, time_col]).size)

            windows.append(
                {
                    "trial_id": trial_id,
                    "window_id": f"{trial_id}_w{i:04d}",
                    "start": float(s),
                    "end": float(e),
                    "n_samples": n_samples,
                    "units": cfg.windowing.units,
                }
            )

    return pd.DataFrame(windows)


def score_windows_missingness(
    df: pd.DataFrame, windows: pd.DataFrame, cfg: PreprocessConfig
) -> pd.DataFrame:
    """Compute missing_frac, n_missing, dropped, drop_reason using windowing.drop settings."""
    if windows.empty:
        windows_out = windows.copy()
        for col in ["missing_frac", "n_missing", "dropped", "drop_reason"]:
            windows_out[col] = []
        return windows_out

    if cfg.windowing.units == "seconds":
        time_col = "time" if "time" in df.columns else "frame"
    else:
        time_col = "frame"

    if cfg.windowing.units == "seconds" and time_col != "time":
        raise ConfigError("windowing.units='seconds' requires a 'time' column.")
    if cfg.windowing.units == "frames" and time_col not in df.columns:
        raise ConfigError("windowing.units='frames' requires a 'frame' column.")

    qc_dims = ["x", "y"] + (["z"] if cfg.windowing.qc_dims == "xyz" else [])
    for d in qc_dims:
        if d not in df.columns:
            raise ConfigError(f"windowing.qc_dims requires '{d}' column.")

    qc_kps = set(_get_qc_keypoints(df, cfg))
    df_qc = df[df["keypoint"].isin(qc_kps)].copy()

    out_rows = []
    for _, w in windows.iterrows():
        trial_id = w["trial_id"]
        s = float(w["start"])
        e = float(w["end"])

        df_trial = df_qc[df_qc["trial_id"] == trial_id]
        if cfg.windowing.include_partial:
            mask = (df_trial[time_col] >= s) & (df_trial[time_col] <= e)
        else:
            mask = (df_trial[time_col] >= s) & (df_trial[time_col] < e)

        df_win = df_trial.loc[mask]

        missing_frac = 1.0
        n_missing = 0
        dropped = False
        drop_reason = None

        if df_win.empty:
            dropped = True
            drop_reason = "empty_window"
        else:
            dims_df = df_win[qc_dims]
            if cfg.windowing.drop.missing_rule == "any_dim_nan":
                miss = dims_df.isna().any(axis=1)
            else:
                miss = dims_df.isna().all(axis=1)

            if cfg.windowing.drop.scope == "aggregate":
                n_missing = int(miss.sum())
                missing_frac = float(n_missing / len(miss))
                dropped = missing_frac > cfg.windowing.drop.max_missing_frac
                if cfg.windowing.drop.max_nans is not None:
                    dropped = dropped or (n_missing > cfg.windowing.drop.max_nans)
            else:
                per_kp = (
                    df_win.assign(_missing=miss)
                    .groupby("keypoint", sort=False)
                    .agg(n_missing=("_missing", "sum"), n_total=("_missing", "size"))
                )
                per_kp["missing_frac"] = per_kp["n_missing"] / per_kp["n_total"]

                if per_kp.empty:
                    dropped = True
                    drop_reason = "empty_window"
                else:
                    missing_frac = float(per_kp["missing_frac"].max())
                    n_missing = int(per_kp["n_missing"].max())

                    if cfg.windowing.drop.per_keypoint_policy == "any":
                        exceeds = (
                            per_kp["missing_frac"] > cfg.windowing.drop.max_missing_frac
                        )
                        if cfg.windowing.drop.max_nans is not None:
                            exceeds = exceeds | (
                                per_kp["n_missing"] > cfg.windowing.drop.max_nans
                            )
                        dropped = bool(exceeds.any())
                    else:
                        exceeds = (
                            per_kp["missing_frac"] > cfg.windowing.drop.max_missing_frac
                        )
                        if cfg.windowing.drop.max_nans is not None:
                            exceeds = exceeds | (
                                per_kp["n_missing"] > cfg.windowing.drop.max_nans
                            )
                        dropped = bool(exceeds.all())

            if dropped and drop_reason is None:
                if (
                    cfg.windowing.drop.max_nans is not None
                    and n_missing > cfg.windowing.drop.max_nans
                ):
                    drop_reason = "max_nans"
                elif missing_frac > cfg.windowing.drop.max_missing_frac:
                    drop_reason = "max_missing_frac"

        out = dict(w)
        out.update(
            {
                "missing_frac": float(missing_frac),
                "n_missing": int(n_missing),
                "dropped": bool(dropped) if cfg.windowing.drop.enabled else False,
                "drop_reason": drop_reason if cfg.windowing.drop.enabled else None,
            }
        )
        out_rows.append(out)

    return pd.DataFrame(out_rows)
