"""
Sliding-window segmentation.

Windowing happens after feature extraction and before RQA (build plan §3). Windows
with more than a configured proportion of missing data are flagged. This module
provides the frame-index windows and a missing-fraction check; feature and RQA
code slice the signals with the returned index ranges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    """A single analysis window over a signal."""

    index: int
    start: int          # inclusive frame
    stop: int           # exclusive frame
    t_start: float      # seconds
    t_stop: float       # seconds
    missing_fraction: float
    flagged: bool

    @property
    def length(self) -> int:
        return self.stop - self.start


def make_windows(
    n_frames: int,
    frame_rate: float,
    window_s: float,
    overlap: float = 0.5,
    *,
    valid: np.ndarray | None = None,
    max_missing: float = 0.5,
) -> list[Window]:
    """Build overlapping windows over ``n_frames``.

    Parameters
    ----------
    n_frames : int
        Total number of frames.
    frame_rate : float
        Sampling rate (Hz), for converting to seconds.
    window_s : float
        Window length in seconds.
    overlap : float
        Fractional overlap between consecutive windows (0.5 = 50%).
    valid : np.ndarray, optional
        Boolean array of shape ``(n_frames,)`` marking valid frames; used to
        compute each window's missing fraction.
    max_missing : float
        Windows whose missing fraction exceeds this are ``flagged`` (not dropped).

    Returns
    -------
    list of Window
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1).")
    win = int(round(window_s * frame_rate))
    if win <= 0:
        raise ValueError("window length must be positive.")
    step = max(1, int(round(win * (1.0 - overlap))))

    windows: list[Window] = []
    idx = 0
    for start in range(0, n_frames - win + 1, step):
        stop = start + win
        if valid is not None:
            miss = float(1.0 - np.mean(valid[start:stop]))
        else:
            miss = 0.0
        windows.append(
            Window(
                index=idx,
                start=start,
                stop=stop,
                t_start=start / frame_rate,
                t_stop=stop / frame_rate,
                missing_fraction=miss,
                flagged=miss > max_missing,
            )
        )
        idx += 1
    return windows
