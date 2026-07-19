"""
Primitive base class and the name registry.

Each feature primitive declares the stream types it consumes and produces, so a
config can be validated before it runs. Primitives register under a name and are
referenced by that name in config (build plan §8: "implement the interface,
register it, reference it by name in config").
"""
from __future__ import annotations

from typing import Any, Callable

from .types import PipelineContext, StreamType


class Primitive:
    """Base class for a composable feature-pipeline primitive.

    Subclasses set the class attributes ``name``, ``consumes``, ``produces`` and
    implement :meth:`apply`. The constructor receives the config ``params`` and
    should validate them (raising ``ValueError`` on bad config).
    """

    name: str = ""
    consumes: frozenset[StreamType] = frozenset()
    produces: frozenset[StreamType] = frozenset()

    def apply(self, ctx: PipelineContext) -> PipelineContext:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Config-serializable description of this step instance."""
        return {"step": self.name, "params": self.params()}

    def params(self) -> dict[str, Any]:
        """Override to expose the resolved parameters for provenance."""
        return {}


_REGISTRY: dict[str, type[Primitive]] = {}


def register(cls: type[Primitive]) -> type[Primitive]:
    """Class decorator: register a primitive under its ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty 'name'.")
    if cls.name in _REGISTRY:
        raise ValueError(f"Primitive name {cls.name!r} is already registered.")
    _REGISTRY[cls.name] = cls
    return cls


def build_primitive(name: str, params: dict[str, Any] | None = None) -> Primitive:
    """Instantiate a registered primitive by name with the given params."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown primitive {name!r}. Registered primitives: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name](**(params or {}))


def registered_primitives() -> dict[str, type[Primitive]]:
    """Return a copy of the step registry (name -> class)."""
    return dict(_REGISTRY)


# Friendlier public aliases — a feature pipeline is a list of "steps".
FeatureStep = Primitive
build_step = build_primitive
available_steps = registered_primitives
