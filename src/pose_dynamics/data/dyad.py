"""
The :class:`Dyad` container for two-person data.

A ``Dyad`` holds **two independent** :class:`~pose_dynamics.data.pose_sequence.PoseSequence`
objects — it is *not* an array with a person axis. Per-person operations apply to
each member independently; cross-person operations (cross-recurrence) are only
defined on a ``Dyad``.

The container enforces a **shared-clock contract**: the two members must have equal
frame rates, be synchronized in start, and (after preprocessing) be equal in
length. Violations raise :class:`SharedClockError` *before* any cross-recurrence
runs, so a mismatch can never silently corrupt a coordination measure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .pose_sequence import PoseSequence


class SharedClockError(ValueError):
    """Raised when a dyad's two members violate the shared-clock contract."""


@dataclass
class Dyad:
    """A pair of independent pose sequences that share a clock.

    Parameters
    ----------
    a, b : PoseSequence
        The two members. They stay independent objects; the dyad never merges
        them into a single array.
    dyad_id : str, optional
        Identifier for the pair (e.g. participant-pair code).
    meta : dict
        Free-form metadata about the pair (condition, trial, etc.).
    require_equal_length : bool
        If ``True`` (default), the shared-clock check also requires equal frame
        counts. Set ``False`` only for the brief window between loading and the
        length-equalizing preprocessing step.
    """

    a: PoseSequence
    b: PoseSequence
    dyad_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    require_equal_length: bool = True

    def __post_init__(self) -> None:
        self.check_shared_clock(require_equal_length=self.require_equal_length)

    # ------------------------------------------------------------------
    # Shared-clock contract
    # ------------------------------------------------------------------
    def check_shared_clock(self, require_equal_length: bool | None = None) -> None:
        """Validate the shared-clock contract; raise if violated.

        Parameters
        ----------
        require_equal_length : bool, optional
            Override the instance's ``require_equal_length``. Cross-recurrence
            callers should invoke this with ``True`` right before running.

        Raises
        ------
        SharedClockError
            If frame rates differ, or (when required) frame counts differ.
        """
        if require_equal_length is None:
            require_equal_length = self.require_equal_length

        if self.a.frame_rate != self.b.frame_rate:
            raise SharedClockError(
                f"Dyad members have different frame rates "
                f"({self.a.frame_rate} Hz vs {self.b.frame_rate} Hz). Both people "
                "in a pair must be recorded at the same rate; resample one before "
                "pairing them."
            )

        if require_equal_length and self.a.n_frames != self.b.n_frames:
            raise SharedClockError(
                f"Dyad members have different lengths "
                f"({self.a.n_frames} vs {self.b.n_frames} frames). Cross-recurrence "
                "requires frame-to-frame correspondence; trim both trials to their "
                "overlapping window during preprocessing before running it."
            )

    @property
    def frame_rate(self) -> float:
        """The shared frame rate (guaranteed equal by the contract)."""
        return self.a.frame_rate

    @property
    def dims(self) -> int:
        """Shared spatial dimensionality of the two members."""
        if self.a.dims != self.b.dims:
            raise SharedClockError(
                f"Dyad members have different dimensionality "
                f"({self.a.dims}D vs {self.b.dims}D)."
            )
        return self.a.dims

    # ------------------------------------------------------------------
    # Per-person operations map over each member independently
    # ------------------------------------------------------------------
    def map(self, fn: Callable[[PoseSequence], PoseSequence]) -> "Dyad":
        """Apply a per-person function to each member, returning a new dyad.

        This is how per-person preprocessing runs on a pair: each member is
        transformed independently, never as a joint array. The shared-clock check
        is deferred (``require_equal_length=False``) because a stage such as
        interpolation may transiently change lengths before the length-equalizing
        step; re-validate explicitly before cross-recurrence.
        """
        return Dyad(
            a=fn(self.a),
            b=fn(self.b),
            dyad_id=self.dyad_id,
            meta=dict(self.meta),
            require_equal_length=False,
        )

    def summary(self) -> dict[str, Any]:
        """A compact summary of both members, for a notebook checkpoint."""
        return {
            "dyad_id": self.dyad_id,
            "frame_rate": self.frame_rate,
            "equal_length": self.a.n_frames == self.b.n_frames,
            "a": self.a.summary(),
            "b": self.b.summary(),
            "meta": dict(self.meta),
        }

    def __repr__(self) -> str:
        return f"Dyad(id={self.dyad_id!r}, a={self.a!r}, b={self.b!r})"
