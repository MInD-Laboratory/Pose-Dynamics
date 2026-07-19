"""Linear kinematic metrics (the amplitude family) — a first-class standalone use."""
from __future__ import annotations

from .metrics import (
    kinematic_summary,
    per_frame_kinematics,
    region_kinematic_summary,
    summarise_signal,
)

__all__ = [
    "per_frame_kinematics",
    "summarise_signal",
    "kinematic_summary",
    "region_kinematic_summary",
]
