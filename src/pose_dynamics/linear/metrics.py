"""
Linear kinematic metrics — the amplitude family.

These quantify *how much* and *how fast* movement happens: displacement, velocity,
and acceleration, summarized by simple statistics (mean, SD, max, RMS). They are a
first-class use of the package on their own — a valid analysis is load → preprocess
→ linear metrics — and they also provide the magnitude-level context the paper pairs
with recurrence's organization-level view.

Everything here is a *reduction* (signal/pose → scalars), not a pipeline stage, so
it works directly on a :class:`~pose_dynamics.data.pose_sequence.PoseSequence`, a
:class:`~pose_dynamics.features.types.FeatureSet`, or a bare 1-D array — no config
required.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..data.pose_sequence import PoseSequence

DEFAULT_STATS = ("mean", "std", "min", "max", "rms")


# ----------------------------------------------------------------------
# Per-frame kinematics
# ----------------------------------------------------------------------
def per_frame_kinematics(seq: PoseSequence) -> dict[str, np.ndarray]:
    """Per-frame displacement / speed / acceleration magnitudes for each keypoint.

    Returns a dict of arrays (dimension-agnostic; works for 2-D and 3-D):

    - ``"displacement"`` : ``(T-1, K)`` frame-to-frame step length ``‖Δx‖``
    - ``"speed"``        : ``(T-1, K)`` velocity magnitude ``‖Δx‖·fps``
    - ``"acceleration"`` : ``(T-2, K)`` acceleration magnitude
    """
    coords = seq.coords
    fps = seq.frame_rate
    disp_vec = np.diff(coords, axis=0)                  # (T-1, K, D)
    vel_vec = disp_vec * fps
    acc_vec = np.diff(vel_vec, axis=0) * fps            # (T-2, K, D)
    return {
        "displacement": np.linalg.norm(disp_vec, axis=-1),
        "speed": np.linalg.norm(vel_vec, axis=-1),
        "acceleration": np.linalg.norm(acc_vec, axis=-1),
    }


# ----------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------
def summarise_signal(x, stats: Iterable[str] = DEFAULT_STATS, prefix: str = "") -> dict[str, float]:
    """Summarize a scalar time series with the requested statistics (NaN-aware)."""
    x = np.asarray(x, dtype=float)
    out: dict[str, float] = {}
    key = lambda s: f"{prefix}_{s}" if prefix else s
    if "mean" in stats:
        out[key("mean")] = float(np.nanmean(x))
    if "std" in stats:
        out[key("std")] = float(np.nanstd(x))
    if "min" in stats:
        out[key("min")] = float(np.nanmin(x))
    if "max" in stats:
        out[key("max")] = float(np.nanmax(x))
    if "rms" in stats:
        out[key("rms")] = float(np.sqrt(np.nanmean(np.square(x))))
    if "median" in stats:
        out[key("median")] = float(np.nanmedian(x))
    return out


def kinematic_summary(seq: PoseSequence, stats: Iterable[str] = DEFAULT_STATS):
    """Tidy per-keypoint summary of displacement/speed/acceleration.

    Returns a ``pandas.DataFrame`` with one row per keypoint and columns
    ``{quantity}_{stat}`` (e.g. ``speed_rms``, ``acceleration_max``).
    """
    import pandas as pd

    kin = per_frame_kinematics(seq)
    rows = []
    for k, name in enumerate(seq.keypoint_names):
        row: dict[str, object] = {"keypoint": name}
        for quantity, arr in kin.items():
            row.update(summarise_signal(arr[:, k], stats=stats, prefix=quantity))
        rows.append(row)
    return pd.DataFrame(rows)


def region_kinematic_summary(
    seq: PoseSequence,
    regions: dict[str, list[int]],
    stats: Iterable[str] = DEFAULT_STATS,
):
    """Per-region summary: aggregate a region's keypoints to a centroid, then summarize.

    ``regions`` maps a region name to keypoint indices. Returns a DataFrame with one
    row per region (movement of the region's centroid).
    """
    import pandas as pd

    rows = []
    for name, idx in regions.items():
        centroid = np.nanmean(seq.coords[:, np.asarray(idx, int), :], axis=1, keepdims=True)
        region_seq = PoseSequence(
            coords=centroid, keypoint_names=[name], frame_rate=seq.frame_rate,
        )
        kin = per_frame_kinematics(region_seq)
        row: dict[str, object] = {"region": name}
        for quantity, arr in kin.items():
            row.update(summarise_signal(arr[:, 0], stats=stats, prefix=quantity))
        rows.append(row)
    return pd.DataFrame(rows)
