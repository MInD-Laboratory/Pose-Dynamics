# src/pose_dynamics/config.py
from __future__ import annotations
from dataclasses import dataclass
import os

# ---------------------------------------------------------------------
# Global state for active configuration
_CFG = None

def set_cfg(cfg):
    """Set the active configuration globally."""
    global _CFG
    _CFG = cfg

def get_cfg():
    """Get the current active configuration."""
    if _CFG is None:
        raise RuntimeError("Configuration has not been set. Call set_cfg() first.")
    return _CFG
# ---------------------------------------------------------------------

@dataclass
class Config:
    RAW_DIR: str = os.getenv("POSE_RAW_DIR", "data/raw")
    OUT_BASE: str = os.getenv("POSE_OUT_BASE", "data/processed")
    PARTICIPANT_INFO_FILE: str = "participant_info.csv"
    CONF_THRESH: float = 0.30
    MAX_INTERP_RUN: int = 60
    FILTER_ORDER: int = 4
    CUTOFF_HZ: float = 10.0
    WINDOW_SECONDS: int = 60
    WINDOW_OVERLAP: float = 0.5
