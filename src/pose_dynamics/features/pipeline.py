"""
Config-driven feature pipeline: build, validate, run.

A feature pipeline is a declared, ordered list of primitives (build plan §4). It
is specified in config as a list of ``{"primitive": name, "params": {...}}`` steps,
built from the registry, and **validated before it runs**: each primitive's
declared input streams must already be available, so an invalid composition (e.g.
z-scoring before any feature exists) fails at config time, not mid-run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data.pose_sequence import PoseSequence
from .base import Primitive, build_primitive
from .types import PipelineContext, StreamType


class PipelineValidationError(ValueError):
    """Raised when a feature pipeline's declared stream types do not line up."""


@dataclass
class FeaturePipeline:
    """An ordered, validated composition of feature primitives."""

    steps: list[Primitive]

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: list[dict[str, Any]]) -> "FeaturePipeline":
        """Build a pipeline from a list of ``{"step", "params"}`` entries.

        (``"primitive"`` is accepted as a backwards-compatible alias for ``"step"``.)
        """
        steps: list[Primitive] = []
        for i, entry in enumerate(config):
            name = entry.get("step", entry.get("primitive"))
            if name is None:
                raise PipelineValidationError(
                    f"Step {i} has no 'step' key: {entry!r}")
            steps.append(build_primitive(name, entry.get("params")))
        pipe = cls(steps=steps)
        pipe.validate()
        return pipe

    def to_config(self) -> list[dict[str, Any]]:
        """Serialize back to a config list (round-trips through ``from_config``)."""
        return [s.describe() for s in self.steps]

    # ------------------------------------------------------------------
    def validate(self, initial: frozenset[StreamType] = frozenset({StreamType.POSE})) -> None:
        """Check the stream-type flow; raise :class:`PipelineValidationError` on mismatch.

        Starts with a POSE available (the loaded sequence). Each step requires its
        ``consumes`` streams to be present; then its ``produces`` streams are added.
        """
        available = set(initial)
        for i, step in enumerate(self.steps):
            missing = step.consumes - available
            if missing:
                names = sorted(s.value for s in missing)
                raise PipelineValidationError(
                    f"Step {i} ({step.name!r}) needs stream(s) {names} but they are "
                    f"not available yet. Available: {sorted(s.value for s in available)}. "
                    "Reorder the pipeline so a primitive that produces those streams "
                    "runs first (e.g. a feature primitive before z-scoring)."
                )
            available |= set(step.produces)

    def produces_signals(self) -> bool:
        """Whether the pipeline yields a SIGNALS (FeatureSet) stream."""
        available = {StreamType.POSE}
        for step in self.steps:
            available |= set(step.produces)
        return StreamType.SIGNALS in available

    # ------------------------------------------------------------------
    def run(self, pose: PoseSequence) -> PipelineContext:
        """Run the pipeline on a pose sequence, returning the final context."""
        self.validate()
        ctx = PipelineContext(pose=pose, features=None)
        for step in self.steps:
            ctx = step.apply(ctx)
        return ctx

    def __repr__(self) -> str:
        return f"FeaturePipeline({[s.name for s in self.steps]})"
