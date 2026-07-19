"""
Feature primitive library and config-driven composition.

The transform from a :class:`PoseSequence` to analysis-ready signals is a declared,
ordered pipeline of typed, composable steps. The three case studies are
different compositions of the same steps, with no case-specific branching.
"""
from __future__ import annotations

# Importing primitives registers them under their names.
from . import primitives  # noqa: F401
from .base import (Primitive, FeatureStep, build_primitive, build_step,
                   register, registered_primitives, available_steps)
from .geometry import (
    TransformParams,
    procrustes_anisotropic,
    procrustes_uniform,
)
from .pipeline import FeaturePipeline, PipelineValidationError
from .types import FeatureSet, PipelineContext, StreamType

__all__ = [
    "Primitive",
    "register",
    "build_primitive",
    "registered_primitives",
    "FeatureStep",
    "build_step",
    "available_steps",
    "FeaturePipeline",
    "PipelineValidationError",
    "FeatureSet",
    "PipelineContext",
    "StreamType",
    "TransformParams",
    "procrustes_uniform",
    "procrustes_anisotropic",
]
