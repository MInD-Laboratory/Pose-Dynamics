"""
Preprocessing stages: masking, interpolation, filtering, and gap policy.

Canonical order (build plan §3):

    confidence mask -> provisional interpolation -> Butterworth filter
                    -> [embedding selection] -> principled re-interpolation

Every stage takes a :class:`~pose_dynamics.data.pose_sequence.PoseSequence` and
returns a new one with provenance extended, and each has a checkpoint plot in
:mod:`.plots`. There is no 2D-vs-3D branch anywhere in this package.
"""
from __future__ import annotations

from .filtering import butterworth_filter
from .gap_policy import (
    DataQualityReport,
    assess_quality,
    combine_reports,
    max_missing_run_per_keypoint,
)
from .interpolation import interpolate_gaps
from .masking import mask_low_confidence
from .plots import (
    plot_filter_checkpoint,
    plot_interpolation_checkpoint,
    plot_masking_checkpoint,
)

__all__ = [
    "mask_low_confidence",
    "interpolate_gaps",
    "butterworth_filter",
    "assess_quality",
    "DataQualityReport",
    "combine_reports",
    "max_missing_run_per_keypoint",
    "plot_masking_checkpoint",
    "plot_interpolation_checkpoint",
    "plot_filter_checkpoint",
]
