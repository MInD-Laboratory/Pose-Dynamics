from __future__ import annotations

import pandas as pd

REQUIRED_COLS = ("t", "kp", "x", "y")
OPTIONAL_COLS = ("z", "conf")


def validate_pose_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Pose DataFrame missing required columns: {missing}")

    if not pd.api.types.is_numeric_dtype(df["t"]):
        raise ValueError("Column 't' must be numeric (seconds).")
    if not pd.api.types.is_integer_dtype(df["kp"]):
        # allow numeric that can be cleanly cast
        if pd.api.types.is_numeric_dtype(df["kp"]) and (df["kp"] % 1 == 0).all():
            df["kp"] = df["kp"].astype(int)
        else:
            raise ValueError("Column 'kp' must be integer keypoint index.")

    # Required numeric columns
    for c in ("x", "y"):
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Column '{c}' must be numeric.")

    # Optional columns but force to be numeric
    for c in OPTIONAL_COLS:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Column '{c}' must be numeric if present.")
