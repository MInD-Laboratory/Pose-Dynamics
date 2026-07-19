"""
pose-dynamics: a reproducible framework for nonlinear analysis of pose data.

The public data model lives in :mod:`pose_dynamics.data` and is re-exported here
for convenience.
"""
from __future__ import annotations

from .study import StudyConfig, run_study
from .data import (
    Dyad,
    PoseSchema,
    PoseSequence,
    ProvenanceEntry,
    ProvenanceLog,
    SchemaError,
    SharedClockError,
    load_pose_csv,
    parse_header,
)

__version__ = "0.0.1"

__all__ = [
    "Dyad",
    "PoseSchema",
    "PoseSequence",
    "ProvenanceEntry",
    "ProvenanceLog",
    "SchemaError",
    "SharedClockError",
    "load_pose_csv",
    "parse_header",
    "StudyConfig",
    "run_study",
    "__version__",
]
