"""
RQA integration: a thin, single-decision wrapper over ``rqa-analysis``.

The framework owns normalization, plotting, and output; this package wraps only
the core recurrence routines and routes by analysis type. See :mod:`.wrapper`.
"""
from __future__ import annotations

from .params import RqaParams
from .result import METRIC_KEYS, RqaResult
from .wrapper import (
    run_auto_rqa,
    run_cross_drp,
    run_cross_rqa,
    run_drp,
    run_multivariate_cross_rqa,
)

__all__ = [
    "RqaParams",
    "RqaResult",
    "METRIC_KEYS",
    "run_auto_rqa",
    "run_cross_rqa",
    "run_multivariate_cross_rqa",
    "run_drp",
    "run_cross_drp",
]
