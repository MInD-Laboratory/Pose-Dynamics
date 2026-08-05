"""
The single in-memory representation of pose data: :class:`PoseSequence`.

There is exactly one such class for all data, 2D and 3D alike. Three-dimensional
data is **not** a special case — it is simply ``dims == 3``. The class holds a
``(frames, keypoints, dims)`` coordinate array, an optional confidence array, a
boolean validity mask, a keypoint-name/skeleton definition, and provenance
metadata (source file, frame rate, and the ordered log of stages applied).

This class is the fix for the prototype's central structural flaw. It has **no**
2D-vs-3D branch: every operation is written against ``self.dims`` and NumPy's
last-axis broadcasting, so a 2D face, a 2D upper body, and a 3D full body flow
through identical code paths.

Instances are treated as copy-on-write: mutating operations return a *new*
``PoseSequence`` with an extended provenance log rather than editing in place.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .provenance import ProvenanceLog


@dataclass
class PoseSequence:
    """A single person's pose trajectory for a single trial.

    Parameters
    ----------
    coords : np.ndarray
        Float array of shape ``(n_frames, n_keypoints, dims)`` with ``dims`` in
        {2, 3}. Missing values are represented as ``np.nan``.
    keypoint_names : list of str
        One name per keypoint; ``len == n_keypoints``. For canonical CSVs without
        names, the loader supplies ``kp0..kpK-1``.
    frame_rate : float
        Sampling rate in Hz. Required — it drives filtering, interpolation caps,
        and the dyad shared-clock check. It is never inferred from the data.
    confidence : np.ndarray, optional
        Float array of shape ``(n_frames, n_keypoints)`` with per-keypoint
        detection confidence, or ``None`` if the source had no confidence columns.
    mask : np.ndarray, optional
        Boolean array of shape ``(n_frames, n_keypoints)``; ``True`` marks a valid
        (present) observation. If omitted, it is initialized from the finiteness
        of ``coords``.
    source_file : str, optional
        Path of the file this sequence was loaded from.
    provenance : ProvenanceLog
        Ordered log of stages applied. Defaults to an empty log.
    meta : dict
        Free-form metadata (participant id, condition, etc.).
    """

    coords: np.ndarray
    keypoint_names: list[str]
    frame_rate: float
    confidence: np.ndarray | None = None
    mask: np.ndarray | None = None
    source_file: str | None = None
    provenance: ProvenanceLog = field(default_factory=ProvenanceLog)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float)
        if self.coords.ndim != 3:
            raise ValueError(
                "coords must be a 3D array (n_frames, n_keypoints, dims); "
                f"got shape {self.coords.shape}."
            )
        n_frames, n_keypoints, dims = self.coords.shape
        if dims not in (2, 3):
            raise ValueError(
                f"Last axis of coords must be 2 (x, y) or 3 (x, y, z); got {dims}."
            )
        if len(self.keypoint_names) != n_keypoints:
            raise ValueError(
                f"keypoint_names has {len(self.keypoint_names)} entries but coords "
                f"has {n_keypoints} keypoints; they must match."
            )
        if self.frame_rate is None or self.frame_rate <= 0:
            raise ValueError(
                f"frame_rate must be a positive number of Hz; got {self.frame_rate!r}. "
                "Frame rate is required and is set from the study config, not inferred."
            )

        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=float)
            if self.confidence.shape != (n_frames, n_keypoints):
                raise ValueError(
                    f"confidence must have shape {(n_frames, n_keypoints)} "
                    f"(n_frames, n_keypoints); got {self.confidence.shape}."
                )

        if self.mask is None:
            # A frame/keypoint is valid iff all of its coordinates are finite.
            self.mask = np.all(np.isfinite(self.coords), axis=-1)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != (n_frames, n_keypoints):
                raise ValueError(
                    f"mask must have shape {(n_frames, n_keypoints)} "
                    f"(n_frames, n_keypoints); got {self.mask.shape}."
                )

    # ------------------------------------------------------------------
    # Shape properties (dims is data, never a branch)
    # ------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return self.coords.shape[0]

    @property
    def n_keypoints(self) -> int:
        return self.coords.shape[1]

    @property
    def dims(self) -> int:
        """Number of spatial dimensions (2 or 3). This is data, not a type."""
        return self.coords.shape[2]

    @property
    def duration_s(self) -> float:
        """Trial duration in seconds implied by frame count and frame rate."""
        return self.n_frames / self.frame_rate

    @property
    def has_confidence(self) -> bool:
        return self.confidence is not None

    # ------------------------------------------------------------------
    # Copy-on-write helpers
    # ------------------------------------------------------------------
    def with_stage(
        self,
        stage: str,
        params: dict[str, Any] | None = None,
        *,
        coords: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> "PoseSequence":
        """Return a new sequence with data replaced and provenance extended.

        This is the single entry point stages use to record what they did. Any of
        ``coords``/``confidence``/``mask``/``meta`` left as ``None`` are carried
        over unchanged (arrays are copied so the original is never aliased).

        Parameters
        ----------
        stage : str
            Stage name for the provenance entry.
        params : dict, optional
            Stage parameters (JSON-serializable) for the provenance entry.
        coords, confidence, mask : np.ndarray, optional
            Replacement arrays. Omitted ones are copied from ``self``.
        meta : dict, optional
            Metadata to merge over the existing ``meta``.
        note : str, optional
            Human-readable annotation for the provenance entry.
        """
        new_coords = self.coords.copy() if coords is None else np.asarray(coords, float)

        if confidence is None:
            new_conf = None if self.confidence is None else self.confidence.copy()
        else:
            new_conf = np.asarray(confidence, dtype=float)

        new_mask = self.mask.copy() if mask is None else np.asarray(mask, dtype=bool)
        new_meta = dict(self.meta)
        if meta:
            new_meta.update(meta)

        return replace(
            self,
            coords=new_coords,
            confidence=new_conf,
            mask=new_mask,
            meta=new_meta,
            provenance=self.provenance.appended(stage, params, note=note),
        )

    def copy(self) -> "PoseSequence":
        """Return a deep-ish copy (arrays copied; provenance is immutable)."""
        return replace(
            self,
            coords=self.coords.copy(),
            confidence=None if self.confidence is None else self.confidence.copy(),
            mask=self.mask.copy(),
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------
    # Inspectable-object contract: summary + checkpoint plot
    # ------------------------------------------------------------------
    def missing_fraction(self) -> float:
        """Overall fraction of invalid (masked-out) keypoint observations."""
        return float(1.0 - self.mask.mean())

    def per_keypoint_missing(self) -> np.ndarray:
        """Fraction missing per keypoint, shape ``(n_keypoints,)``."""
        return 1.0 - self.mask.mean(axis=0)

    def summary(self) -> dict[str, Any]:
        """A compact, serializable summary suitable for a notebook checkpoint."""
        return {
            "source_file": self.source_file,
            "n_frames": self.n_frames,
            "n_keypoints": self.n_keypoints,
            "dims": self.dims,
            "frame_rate": self.frame_rate,
            "duration_s": round(self.duration_s, 3),
            "has_confidence": self.has_confidence,
            "missing_fraction": round(self.missing_fraction(), 4),
            "stages_applied": self.provenance.stages,
            "meta": dict(self.meta),
        }

    def plot_coverage(self, ax=None):
        """Checkpoint visual: keypoint-presence over time (valid vs missing).

        Renders the validity mask as a frames × keypoints image so gaps are
        immediately visible. When there is no missing data the mask is
        uniformly valid and carries no information, so instead this plots the
        per-keypoint frame-to-frame displacement over time (a motion trace).
        ``matplotlib`` is imported lazily so the core install stays light.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt  # lazy import (optional dependency)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        title = self.source_file or "PoseSequence"

        if self.missing_fraction() == 0:
            t = np.arange(self.n_frames) / self.frame_rate
            disp = np.linalg.norm(np.diff(self.coords, axis=0, prepend=self.coords[:1]), axis=-1)
            for k in range(self.n_keypoints):
                ax.plot(t, disp[:, k], color="0.2", alpha=0.15, lw=0.6)
            ax.plot(t, disp.mean(axis=1), color="black", lw=1.5, label="mean displacement")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("frame-to-frame displacement")
            ax.set_title(f"{title}  ({self.dims}D, full coverage — motion trace)")
            ax.legend(loc="upper right", fontsize=8)
            return ax

        # mask is (frames, keypoints); show keypoints on y, time on x.
        ax.imshow(
            self.mask.T,
            aspect="auto",
            interpolation="nearest",
            cmap="Greys",
            vmin=0,
            vmax=1,
            extent=[0, self.duration_s, self.n_keypoints, 0],
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("keypoint index")
        ax.set_title(f"{title}  ({self.dims}D, {self.missing_fraction():.1%} missing)")
        return ax

    def plot_keypoints(self, frame: int | None = None, ax=None):
        """Plot the keypoints labelled with their index numbers.

        Use this to read off which index is which body part before choosing ROI
        indices for a feature pipeline. Uses the mean pose by default (a clean,
        stable layout); pass ``frame`` for a specific frame. 2D poses render as one
        upright panel (image y-axis inverted so the head is up); 3D poses render as
        two projections (front x–y and top-down x–z).

        Returns the primary ``matplotlib.axes.Axes`` (``ax.figure`` holds all panels).
        """
        import matplotlib.pyplot as plt  # lazy import

        pose = (self.coords[frame] if frame is not None
                else np.nanmean(self.coords, axis=0))          # (K, dims)

        def _scatter(a, i, j, invert_y=False):
            a.scatter(pose[:, i], pose[:, j], s=45, color="tab:blue", zorder=3)
            for k in range(self.n_keypoints):
                if np.isfinite(pose[k, i]) and np.isfinite(pose[k, j]):
                    a.annotate(str(k), (pose[k, i], pose[k, j]), fontsize=8,
                               xytext=(3, 3), textcoords="offset points",
                               color="black",
                               bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
            if invert_y:
                a.invert_yaxis()
            a.set_aspect("equal")

        title = self.source_file or "keypoints"
        if self.dims == 2:
            if ax is None:
                _, ax = plt.subplots(figsize=(6, 7))
            _scatter(ax, 0, 1, invert_y=True)                  # image coords -> head up
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title(f"{title} — {self.n_keypoints} keypoints (2D)")
            return ax

        if ax is None:
            _, axes = plt.subplots(1, 2, figsize=(11, 6))
        else:
            axes = np.atleast_1d(ax)
        _scatter(axes[0], 0, 1)                                # front view x-y
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_title("front (x–y)")
        if len(axes) > 1:
            _scatter(axes[1], 0, 2)                            # top-down x-z
            axes[1].set_xlabel("x"); axes[1].set_ylabel("z"); axes[1].set_title("top (x–z)")
        axes[0].figure.suptitle(f"{title} — {self.n_keypoints} keypoints (3D)")
        return axes[0]

    def __repr__(self) -> str:
        return (
            f"PoseSequence(frames={self.n_frames}, keypoints={self.n_keypoints}, "
            f"dims={self.dims}, fps={self.frame_rate:g}, "
            f"conf={self.has_confidence}, stages={self.provenance.stages})"
        )
