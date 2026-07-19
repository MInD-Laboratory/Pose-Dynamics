"""
Embedding selection (AMI / FNN) — framework-owned, human-committed.

The framework computes and presents the evidence; the researcher commits the
single ``(τ, m)`` applied across the study. Nothing here auto-selects a minimum
(build plan §3).
"""
from __future__ import annotations

from .ami import AmiCurve, ami_curve, cross_ami_curve
from .fnn import FnnCurve, fnn_curve
from .plots import plot_embedding_evidence, plot_embedding_variability
from .selection import (
    EmbeddingEvidence,
    EmbeddingParams,
    Signal,
    coordinate_channels,
    magnitude_channels,
    pool_signals,
    select_embedding,
)

__all__ = [
    "ami_curve",
    "cross_ami_curve",
    "AmiCurve",
    "fnn_curve",
    "FnnCurve",
    "EmbeddingEvidence",
    "EmbeddingParams",
    "Signal",
    "select_embedding",
    "coordinate_channels",
    "magnitude_channels",
    "pool_signals",
    "plot_embedding_evidence",
    "plot_embedding_variability",
]
