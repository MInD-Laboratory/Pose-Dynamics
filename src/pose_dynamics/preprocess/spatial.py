from __future__ import annotations

import pandas as pd

from pose_dynamics.preprocess.schema import ConfigError, PreprocessConfig


def _time_col(df: pd.DataFrame) -> str:
    return "time" if "time" in df.columns else "frame"


def apply_spatial(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    centering = cfg.spatial.centering
    scale = cfg.spatial.scale

    if centering.method == "none" and scale.method == "none":
        return df

    dims = [c for c in ["x", "y", "z"] if c in df.columns]
    if not dims:
        return df

    df_out = df.copy()

    if centering.method != "none":
        tcol = _time_col(df_out)
        if centering.method == "mean_keypoints":
            means = df_out.groupby(["trial_id", tcol])[dims].mean().reset_index()
            df_out = df_out.merge(means, on=["trial_id", tcol], suffixes=("", "_mean"))
            for d in dims:
                df_out[d] = df_out[d] - df_out[f"{d}_mean"]
                df_out.drop(columns=[f"{d}_mean"], inplace=True)
        elif centering.method == "anchor_keypoint":
            anchor = centering.anchor_keypoint
            if anchor is None:
                raise ConfigError("spatial.centering.anchor_keypoint is required.")
            anchor_df = df_out[df_out["keypoint"] == anchor]
            if anchor_df.empty:
                raise ConfigError("anchor_keypoint not found in data.")
            anchor_df = (
                anchor_df[["trial_id", tcol] + dims]
                .groupby(["trial_id", tcol])
                .mean()
                .reset_index()
            )
            df_out = df_out.merge(
                anchor_df, on=["trial_id", tcol], suffixes=("", "_anchor")
            )
            for d in dims:
                df_out[d] = df_out[d] - df_out[f"{d}_anchor"]
                df_out.drop(columns=[f"{d}_anchor"], inplace=True)
        else:
            raise ConfigError("unknown spatial.centering.method")

    if scale.method == "unit_range":
        stats = df_out.groupby("trial_id")[dims].agg(["min", "max"])
        for d in dims:
            df_out[f"_{d}_min"] = df_out["trial_id"].map(stats[(d, "min")])
            df_out[f"_{d}_max"] = df_out["trial_id"].map(stats[(d, "max")])
            denom = df_out[f"_{d}_max"] - df_out[f"_{d}_min"]
            denom = denom.where(denom != 0, 1.0)
            df_out[d] = (df_out[d] - df_out[f"_{d}_min"]) / denom
            df_out.drop(columns=[f"_{d}_min", f"_{d}_max"], inplace=True)
    elif scale.method == "screen":
        width = scale.width_px
        height = scale.height_px
        if width is None or height is None:
            raise ConfigError("spatial.scale.screen requires width_px and height_px.")
        if width <= 0 or height <= 0:
            raise ConfigError("spatial.scale width_px/height_px must be positive.")
        if "x" in dims:
            df_out["x"] = df_out["x"] / width
        if "y" in dims:
            df_out["y"] = df_out["y"] / height
    elif scale.method != "none":
        raise ConfigError("unknown spatial.scale.method")

    return df_out
