# io_utils.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from datetime import datetime


# -------------------------------------------------------------
# Directory helpers
# -------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    """Return ISO-like timestamp without special characters."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# -------------------------------------------------------------
# DataFrame and array saving
# -------------------------------------------------------------

def save_df(df: pd.DataFrame, path: str | Path):
    """Save DataFrame to CSV (UTF-8, no index)."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return str(path)


def save_parquet(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return str(path)


def save_array(arr: np.ndarray, path: str | Path):
    """Save numpy array to NPZ."""
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, arr)
    return str(path)


# -------------------------------------------------------------
# JSON I/O
# -------------------------------------------------------------

def save_json(data: Dict[str, Any], path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return str(path)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------------------
# RQA Stats Writer
# -------------------------------------------------------------

def write_rqa_stats(
    analysis_name: str,
    params: Dict[str, Any],
    stats: Dict[str, Any],
    err_code: int = 0,
    out_dir: str | Path = "results/rqa",
):
    """
    Save RQA stats for a single analysis run.

    Saves a JSON file containing:
        {
          "analysis": "...",
          "err_code": 0 or nonzero,
          "params": {...},
          "metrics": {...}
        }
    """
    out_dir = ensure_dir(out_dir)

    fname = f"{analysis_name}_{timestamp()}.json"
    path = out_dir / fname

    payload = {
        "analysis": analysis_name,
        "err_code": err_code,
        "params": params,
        "metrics": stats,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return str(path)
