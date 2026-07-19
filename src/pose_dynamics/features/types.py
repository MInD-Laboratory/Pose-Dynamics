"""
Stream types and containers for the feature pipeline.

The transform from a :class:`PoseSequence` to analysis-ready signals is a declared,
ordered pipeline of composable primitives (build plan §4). Two kinds of data flow
through it:

- **POSE** — a :class:`PoseSequence` (the geometry stream);
- **SIGNALS** — a :class:`FeatureSet` of named 1-D time series (derived features).

Every primitive declares which streams it consumes and produces; the pipeline
validates that the declared types line up, so an invalid composition fails at
config-validation time rather than mid-run. There is no branch on ``dims`` — the
same primitives handle 2-D and 3-D poses.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np

from ..data.pose_sequence import PoseSequence
from ..data.provenance import ProvenanceLog


class StreamType(Enum):
    """The two data kinds that flow through a feature pipeline."""

    POSE = "pose"
    SIGNALS = "signals"


@dataclass
class FeatureSet:
    """A set of named, time-aligned 1-D signals derived from a pose sequence.

    All signals share one length ``n_frames`` (finite-difference primitives use
    same-length derivatives so features can be stacked and windowed together).

    Parameters
    ----------
    values : np.ndarray
        Array of shape ``(n_frames, n_features)``.
    names : list of str
        Feature names; ``len == n_features``.
    frame_rate : float
        Sampling rate in Hz.
    provenance : ProvenanceLog
        Ordered log of primitives applied.
    meta : dict
        Free-form metadata carried from the source sequence.
    """

    values: np.ndarray
    names: list[str]
    frame_rate: float
    provenance: ProvenanceLog = field(default_factory=ProvenanceLog)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        if self.values.ndim != 2:
            raise ValueError(f"FeatureSet values must be 2-D (frames, features); got {self.values.shape}.")
        if self.values.shape[1] != len(self.names):
            raise ValueError(
                f"names has {len(self.names)} entries but values has "
                f"{self.values.shape[1]} columns."
            )
        if len(set(self.names)) != len(self.names):
            dupes = {n for n in self.names if self.names.count(n) > 1}
            raise ValueError(f"Duplicate feature names: {sorted(dupes)}")

    @property
    def n_frames(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    def get(self, name: str) -> np.ndarray:
        """Return the single feature column named ``name``."""
        return self.values[:, self.names.index(name)]

    def added(
        self,
        new_names: list[str],
        new_values: np.ndarray,
        stage: str,
        params: dict[str, Any] | None = None,
    ) -> "FeatureSet":
        """Return a new FeatureSet with columns appended and provenance extended."""
        new_values = np.asarray(new_values, dtype=float)
        if new_values.ndim == 1:
            new_values = new_values[:, None]
        if new_values.shape[0] != self.n_frames:
            raise ValueError(
                f"new features have {new_values.shape[0]} frames but the set has "
                f"{self.n_frames}; features must stay time-aligned."
            )
        return replace(
            self,
            values=np.hstack([self.values, new_values]),
            names=self.names + list(new_names),
            provenance=self.provenance.appended(stage, params),
        )

    def replaced(
        self,
        names: list[str],
        values: np.ndarray,
        stage: str,
        params: dict[str, Any] | None = None,
    ) -> "FeatureSet":
        """Return a new FeatureSet whose columns are fully replaced (e.g. z-scoring)."""
        return replace(
            self,
            values=np.asarray(values, dtype=float),
            names=list(names),
            provenance=self.provenance.appended(stage, params),
        )

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.values, columns=self.names)

    def summary(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "n_features": self.n_features,
            "names": list(self.names),
            "frame_rate": self.frame_rate,
            "stages": self.provenance.stages,
        }

    def plot(self, names: list[str] | None = None, ax=None):
        """Checkpoint visual: plot selected feature time series."""
        import matplotlib.pyplot as plt

        names = names or self.names[: min(6, self.n_features)]
        if ax is None:
            _, ax = plt.subplots(figsize=(11, 4))
        t = np.arange(self.n_frames) / self.frame_rate
        for name in names:
            ax.plot(t, self.get(name), lw=0.9, label=name)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("feature value")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Feature signals")
        return ax

    def __repr__(self) -> str:
        return f"FeatureSet(frames={self.n_frames}, features={self.n_features}, names={self.names})"


@dataclass
class PipelineContext:
    """The mutable-by-replacement state threaded through a feature pipeline."""

    pose: PoseSequence | None = None
    features: FeatureSet | None = None

    def available(self) -> set[StreamType]:
        streams: set[StreamType] = set()
        if self.pose is not None:
            streams.add(StreamType.POSE)
        if self.features is not None:
            streams.add(StreamType.SIGNALS)
        return streams
