"""Canonical data model: schema, PoseSequence, Dyad, loader, provenance."""
from __future__ import annotations

from .dyad import Dyad, SharedClockError
from .loader import load_pose_csv
from .pose_sequence import PoseSequence
from .provenance import ProvenanceEntry, ProvenanceLog
from .fixtures import example_dataset, example_fixture, generate_fixture
from .schema import PoseSchema, SchemaError, parse_header

__all__ = [
    "Dyad",
    "SharedClockError",
    "load_pose_csv",
    "PoseSequence",
    "ProvenanceEntry",
    "ProvenanceLog",
    "PoseSchema",
    "SchemaError",
    "parse_header",
    "example_fixture",
    "generate_fixture",
    "example_dataset",
]
