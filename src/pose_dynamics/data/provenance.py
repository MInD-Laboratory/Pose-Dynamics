"""
Provenance logging for the pose-dynamics data model.

Every :class:`~pose_dynamics.data.pose_sequence.PoseSequence` carries an ordered
log of the stages that have been applied to it. This is the audit trail that lets
any downstream result be traced back to the exact sequence of operations (and
their parameters) that produced it, satisfying the reproducibility requirement in
the build plan (§6, §7): "Every result artifact carries the full resolved config
that produced it."

The log is deliberately simple and serializable: a list of immutable records,
each naming a stage, the parameters it was called with, and an optional
timestamp. Records are append-only; stages never mutate earlier entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProvenanceEntry:
    """A single, immutable record of one stage applied to a sequence.

    Parameters
    ----------
    stage : str
        Name of the stage (e.g. ``"load"``, ``"confidence_mask"``).
    params : dict
        The parameters the stage was invoked with. Must be JSON-serializable
        values so the whole log can be written alongside results.
    timestamp : str
        ISO-8601 UTC timestamp of when the entry was recorded.
    note : str, optional
        Free-text human-readable annotation (e.g. "12 gaps filled").
    """

    stage: str
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now_iso)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation for serialization."""
        return {
            "stage": self.stage,
            "params": dict(self.params),
            "timestamp": self.timestamp,
            "note": self.note,
        }


@dataclass(frozen=True)
class ProvenanceLog:
    """An append-only, ordered log of :class:`ProvenanceEntry` records.

    The log is immutable: :meth:`appended` returns a *new* log with one more
    entry rather than mutating in place, mirroring the copy-on-write style of the
    data model so that provenance and data can never drift out of sync.
    """

    entries: tuple[ProvenanceEntry, ...] = ()

    def appended(
        self,
        stage: str,
        params: dict[str, Any] | None = None,
        note: str | None = None,
        timestamp: str | None = None,
    ) -> "ProvenanceLog":
        """Return a new log with one more entry appended.

        Parameters
        ----------
        stage : str
            Name of the stage.
        params : dict, optional
            Stage parameters (JSON-serializable). Defaults to ``{}``.
        note : str, optional
            Human-readable annotation.
        timestamp : str, optional
            Override the timestamp (mainly for deterministic tests). Defaults to
            the current UTC time.
        """
        entry = ProvenanceEntry(
            stage=stage,
            params=dict(params or {}),
            note=note,
            **({"timestamp": timestamp} if timestamp is not None else {}),
        )
        return replace(self, entries=self.entries + (entry,))

    @property
    def stages(self) -> list[str]:
        """The ordered list of stage names applied so far."""
        return [e.stage for e in self.entries]

    def to_list(self) -> list[dict[str, Any]]:
        """Return a list-of-dicts representation for serialization."""
        return [e.to_dict() for e in self.entries]

    def __iter__(self) -> Iterator[ProvenanceEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> ProvenanceEntry:
        return self.entries[idx]

    @classmethod
    def from_entries(cls, entries: Iterable[ProvenanceEntry]) -> "ProvenanceLog":
        """Build a log from an iterable of entries."""
        return cls(entries=tuple(entries))
