"""
Gap policy and data-quality reporting.

The build plan (§3) requires that gap handling be explicit and configurable: runs
longer than the interpolation cap are left missing, and the config declares what
happens to a trial exceeding a missingness threshold — **retain with gaps, flag,
or exclude**. Every excluded or flagged trial appears in a data-quality report and
is *never dropped silently*.

This module measures the missingness left after masking/interpolation and turns it
into an inspectable :class:`DataQualityReport`. It makes no decision to discard
data on its own; it records a *status* that the pipeline and the reproduction
layer act on transparently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np

from ..data.pose_sequence import PoseSequence

OnExceed = Literal["retain", "flag", "exclude"]


def max_missing_run_per_keypoint(seq: PoseSequence) -> np.ndarray:
    """Longest consecutive-missing run (in frames) for each keypoint.

    Returns an integer array of shape ``(n_keypoints,)``.
    """
    invalid = ~seq.mask  # (frames, keypoints); True = missing
    n_kp = seq.n_keypoints
    longest = np.zeros(n_kp, dtype=int)
    for k in range(n_kp):
        col = invalid[:, k]
        if not col.any():
            continue
        # Length of the longest True run.
        best = run = 0
        for v in col:
            run = run + 1 if v else 0
            if run > best:
                best = run
        longest[k] = best
    return longest


@dataclass
class DataQualityReport:
    """An inspectable summary of a trial's missingness and its policy status.

    Attributes
    ----------
    status : {"ok", "retain", "flag", "exclude"}
        ``"ok"`` if overall missingness is within threshold; otherwise the
        configured ``on_exceed`` action.
    excluded : bool
        Convenience flag, ``True`` iff ``status == "exclude"``.
    flagged_keypoints : list of str
        Keypoints whose individual missingness exceeds the per-keypoint threshold
        (candidates for dropping from analysis, per the manuscript's 20–30%
        guidance).
    """

    source_file: str | None
    n_frames: int
    n_keypoints: int
    dims: int
    frame_rate: float
    keypoint_names: list[str]
    missing_fraction: float
    per_keypoint_missing: np.ndarray
    max_gap_frames: np.ndarray
    threshold_frac: float
    per_keypoint_threshold: float
    on_exceed: OnExceed
    status: str
    flagged_keypoints: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def excluded(self) -> bool:
        return self.status == "exclude"

    @property
    def max_gap_seconds(self) -> float:
        return float(self.max_gap_frames.max() / self.frame_rate) if self.n_keypoints else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "status": self.status,
            "missing_fraction": round(self.missing_fraction, 4),
            "threshold_frac": self.threshold_frac,
            "max_gap_frames": int(self.max_gap_frames.max()) if self.n_keypoints else 0,
            "max_gap_seconds": round(self.max_gap_seconds, 3),
            "flagged_keypoints": list(self.flagged_keypoints),
            "reasons": list(self.reasons),
        }

    def to_row(self) -> dict[str, Any]:
        """Flat one-row dict for building a study-level data-quality table."""
        return {
            "source_file": self.source_file,
            "n_frames": self.n_frames,
            "dims": self.dims,
            "missing_fraction": self.missing_fraction,
            "max_gap_frames": int(self.max_gap_frames.max()) if self.n_keypoints else 0,
            "max_gap_seconds": self.max_gap_seconds,
            "status": self.status,
            "excluded": self.excluded,
            "n_flagged_keypoints": len(self.flagged_keypoints),
            "flagged_keypoints": ";".join(self.flagged_keypoints),
        }

    def plot(self, ax=None):
        """Checkpoint visual: per-keypoint missingness with the threshold line."""
        import matplotlib.pyplot as plt  # lazy import

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        idx = np.arange(self.n_keypoints)
        colors = [
            "tab:red" if name in self.flagged_keypoints else "tab:blue"
            for name in self.keypoint_names
        ]
        ax.bar(idx, self.per_keypoint_missing, color=colors)
        ax.axhline(
            self.per_keypoint_threshold,
            color="k",
            linestyle="--",
            linewidth=1,
            label=f"per-keypoint threshold ({self.per_keypoint_threshold:.0%})",
        )
        ax.set_xlabel("keypoint index")
        ax.set_ylabel("fraction missing")
        title = self.source_file or "trial"
        ax.set_title(f"{title}  —  status: {self.status}  ({self.missing_fraction:.1%} overall)")
        ax.legend(loc="upper right")
        return ax

    def __repr__(self) -> str:
        return (
            f"DataQualityReport(status={self.status!r}, "
            f"missing={self.missing_fraction:.2%}, "
            f"flagged={len(self.flagged_keypoints)})"
        )


def assess_quality(
    seq: PoseSequence,
    *,
    max_missing_frac: float = 0.30,
    per_keypoint_max_missing_frac: float = 0.30,
    on_exceed: OnExceed = "flag",
) -> DataQualityReport:
    """Measure missingness and assign a policy status.

    Intended to be called on the sequence *after masking and provisional
    interpolation*, so ``missing_fraction`` reflects what interpolation could not
    recover (the truly missing data).

    Parameters
    ----------
    seq : PoseSequence
        The sequence to assess.
    max_missing_frac : float
        Trial-level threshold. If overall missingness exceeds this, the trial's
        status becomes ``on_exceed``.
    per_keypoint_max_missing_frac : float
        Per-keypoint threshold; keypoints above it are listed in
        ``flagged_keypoints`` (candidates to drop from analysis).
    on_exceed : {"retain", "flag", "exclude"}
        What status to assign a trial that exceeds the trial-level threshold.

    Returns
    -------
    DataQualityReport
    """
    per_kp_missing = seq.per_keypoint_missing()
    overall = seq.missing_fraction()
    max_gaps = max_missing_run_per_keypoint(seq)

    flagged = [
        name
        for name, frac in zip(seq.keypoint_names, per_kp_missing)
        if frac > per_keypoint_max_missing_frac
    ]

    reasons: list[str] = []
    if overall > max_missing_frac:
        status = on_exceed
        reasons.append(
            f"overall missingness {overall:.1%} exceeds trial threshold "
            f"{max_missing_frac:.1%} -> {on_exceed}"
        )
    else:
        status = "ok"
    if flagged:
        reasons.append(
            f"{len(flagged)} keypoint(s) exceed per-keypoint threshold "
            f"{per_keypoint_max_missing_frac:.1%}: {flagged}"
        )

    return DataQualityReport(
        source_file=seq.source_file,
        n_frames=seq.n_frames,
        n_keypoints=seq.n_keypoints,
        dims=seq.dims,
        frame_rate=seq.frame_rate,
        keypoint_names=list(seq.keypoint_names),
        missing_fraction=overall,
        per_keypoint_missing=per_kp_missing,
        max_gap_frames=max_gaps,
        threshold_frac=max_missing_frac,
        per_keypoint_threshold=per_keypoint_max_missing_frac,
        on_exceed=on_exceed,
        status=status,
        flagged_keypoints=flagged,
        reasons=reasons,
    )


def combine_reports(reports: Iterable[DataQualityReport]):
    """Combine per-trial reports into a study-level data-quality table.

    Returns a ``pandas.DataFrame`` (one row per trial). Imported lazily so the
    core dependency footprint is unaffected.
    """
    import pandas as pd

    return pd.DataFrame([r.to_row() for r in reports])
