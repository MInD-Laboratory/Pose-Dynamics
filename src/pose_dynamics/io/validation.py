from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLS = ("t", "kp", "x", "y")
OPTIONAL_COLS = ("z", "conf", "subject", "trial")


def coerce_pose_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized dtypes/columns for canonical pose DF."""
    out = df.copy()

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Pose DataFrame missing required columns: {missing}")

    # Add optional columns if absent
    for c in OPTIONAL_COLS:
        if c not in out.columns:
            out[c] = np.nan

    # Coerce dtypes
    out["t"] = pd.to_numeric(out["t"], errors="raise")
    out["kp"] = pd.to_numeric(out["kp"], errors="raise").astype(int)

    for c in ("x", "y", "z", "conf"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def validate_pose_df(df: pd.DataFrame) -> None:
    """Validate canonical pose DF without modifying it."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Pose DataFrame missing required columns: {missing}")

    # t
    if not pd.api.types.is_numeric_dtype(df["t"]):
        raise ValueError("Column 't' must be numeric (seconds).")
    if not np.isfinite(df["t"].to_numpy()).all():
        raise ValueError("Column 't' must be finite.")
    if (df["t"].diff().dropna() < 0).any():
        raise ValueError("Column 't' must be nondecreasing.")

    # kp
    if not pd.api.types.is_integer_dtype(df["kp"]):
        raise ValueError("Column 'kp' must be integer dtype.")
    if (df["kp"] < 0).any():
        raise ValueError("Column 'kp' must be >= 0.")

    # x,y numeric
    for c in ("x", "y"):
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Column '{c}' must be numeric.")

    # optional numeric if present
    for c in ("z", "conf"):
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Column '{c}' must be numeric if present.")

    # conf range if present & not-null
    if "conf" in df.columns:
        conf = df["conf"].dropna().to_numpy()
        if conf.size and ((conf < 0).any() or (conf > 1).any()):
            raise ValueError("Column 'conf' must be in [0, 1] when present.")
