"""
Checkpoint plots for the preprocessing stages.

Each preprocessing stage must be visually checkable (build plan §3: "A stage that
cannot be visually checked is a design defect"). These helpers take the sequence
before and after a stage and draw the canonical checkpoint for that stage:

- :func:`plot_masking_checkpoint` — signal with masked samples highlighted;
- :func:`plot_interpolation_checkpoint` — filled vs. still-missing segments;
- :func:`plot_filter_checkpoint` — pre/post-filter overlay.

They plot a single keypoint/axis channel (a notebook typically loops over a few).
``matplotlib`` is imported lazily so the core install stays light.
"""
from __future__ import annotations

import numpy as np

from ..data.pose_sequence import PoseSequence


def _time_axis(seq: PoseSequence) -> np.ndarray:
    return np.arange(seq.n_frames) / seq.frame_rate


def _channel(seq: PoseSequence, keypoint: int, axis: int) -> np.ndarray:
    return seq.coords[:, keypoint, axis]


def plot_masking_checkpoint(
    before: PoseSequence,
    after: PoseSequence,
    keypoint: int = 0,
    axis: int = 0,
    ax=None,
):
    """Show the raw channel with masked-out samples marked in red."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5))
    t = _time_axis(before)
    raw = _channel(before, keypoint, axis)
    newly_masked = np.isfinite(raw) & ~np.isfinite(_channel(after, keypoint, axis))

    ax.plot(t, raw, color="0.6", lw=0.8, label="raw")
    ax.scatter(
        t[newly_masked], raw[newly_masked], s=10, color="tab:red",
        zorder=3, label=f"masked ({int(newly_masked.sum())})",
    )
    axis_name = "xyz"[axis]
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{before.keypoint_names[keypoint]} {axis_name}")
    ax.set_title("Confidence masking")
    ax.legend(loc="upper right")
    return ax


def plot_interpolation_checkpoint(
    before: PoseSequence,
    after: PoseSequence,
    keypoint: int = 0,
    axis: int = 0,
    ax=None,
):
    """Show interpolated samples (green) vs gaps left missing (red spans)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5))
    t = _time_axis(before)
    pre = _channel(before, keypoint, axis)
    post = _channel(after, keypoint, axis)

    filled = ~np.isfinite(pre) & np.isfinite(post)   # newly interpolated
    still_missing = ~np.isfinite(post)               # left as gaps

    ax.plot(t, post, color="tab:blue", lw=0.9, label="signal")
    ax.scatter(
        t[filled], post[filled], s=10, color="tab:green", zorder=3,
        label=f"interpolated ({int(filled.sum())})",
    )
    # Shade the runs that were left missing.
    _shade_runs(ax, t, still_missing, color="tab:red", alpha=0.15, label="left missing")
    axis_name = "xyz"[axis]
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{before.keypoint_names[keypoint]} {axis_name}")
    ax.set_title("Provisional interpolation")
    ax.legend(loc="upper right")
    return ax


def plot_filter_checkpoint(
    before: PoseSequence,
    after: PoseSequence,
    keypoint: int = 0,
    axis: int = 0,
    ax=None,
):
    """Overlay the pre-filter and post-filter signals."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5))
    t = _time_axis(before)
    ax.plot(t, _channel(before, keypoint, axis), color="0.6", lw=0.8, label="pre-filter")
    ax.plot(t, _channel(after, keypoint, axis), color="tab:orange", lw=1.1, label="post-filter")
    axis_name = "xyz"[axis]
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{before.keypoint_names[keypoint]} {axis_name}")
    ax.set_title("Butterworth filter")
    ax.legend(loc="upper right")
    return ax


def _shade_runs(ax, t, mask_bool, *, color, alpha, label=None):
    """Shade contiguous True runs of a boolean mask along the time axis."""
    labeled = False
    in_run = False
    start = 0
    for i, v in enumerate(mask_bool):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            ax.axvspan(t[start], t[i - 1], color=color, alpha=alpha,
                       label=None if labeled else label)
            labeled = True
    if in_run:
        ax.axvspan(t[start], t[-1], color=color, alpha=alpha,
                   label=None if labeled else label)
