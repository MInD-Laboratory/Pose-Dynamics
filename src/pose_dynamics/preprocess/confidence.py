from __future__ import annotations

import warnings

import pandas as pd

from pose_dynamics.preprocess.schema import PreprocessConfig


def apply_confidence_mask(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    if not cfg.confidence.enabled:
        return df

    if "conf" not in df.columns:
        warnings.warn(
            "confidence.enabled=true but no 'conf' column found; skipping masking.",
            RuntimeWarning,
        )
        return df

    df_out = df.copy()
    conf_min = (
        float(cfg.confidence.conf_min) if cfg.confidence.conf_min is not None else None
    )
    if conf_min is None:
        return df_out

    mask = df_out["conf"] < conf_min
    for col in ["x", "y", "z"]:
        if col in df_out.columns:
            df_out.loc[mask, col] = pd.NA

    return df_out
