"""
Gap-filling by run-limited linear interpolation.

Short gaps are filled by linear interpolation; gaps longer than a cap are left as
missing (build plan §3, and the ``(m-1)τ`` argument in the manuscript). This same
function serves both interpolation passes:

- the **provisional** pass, under a conservative fixed cap, run right after
  masking to give the filter continuity and to feed AMI/FNN;
- the **principled** pass later, under the ``(m-1)τ`` cap, re-run *from the masked
  signal* once ``(m, τ)`` are known.

Interpolation is run-based and honest about edges: only an interior gap (bounded
by observed values on both sides) whose length is ``<= max_gap`` is filled. Gaps
longer than the cap, and gaps at the start/end of the trial where there is nothing
to interpolate between, are left as ``NaN``.
"""
from __future__ import annotations

import numpy as np

from ..data.pose_sequence import PoseSequence


def _interpolate_series_run_limited(
    x: np.ndarray, max_gap: int, edge_fill: bool = False
) -> tuple[np.ndarray, int, int]:
    """Fill interior NaN runs of length ``<= max_gap`` in a 1D series.

    With ``edge_fill=True``, leading and trailing runs are additionally filled by
    holding the nearest observed value constant, matching pandas'
    ``interpolate(limit_direction="both")``. This is off by default because a
    constant edge extension is extrapolation rather than interpolation: it
    invents samples that are bracketed on one side only. Enable it only to match
    an external pipeline (Case 1 does; see its preprocessing notes).

    Returns ``(filled, n_filled, n_left)`` where ``n_filled`` is the number of
    samples filled and ``n_left`` the number of missing samples left untouched.
    """
    out = x.copy()
    finite_idx = np.flatnonzero(np.isfinite(x))
    n_missing = int(np.sum(~np.isfinite(x)))
    if finite_idx.size < 2:
        # Nothing to interpolate between.
        return out, 0, n_missing

    n_filled = 0
    for a, b in zip(finite_idx[:-1], finite_idx[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        if gap <= max_gap:
            # linear ramp between the two bracketing observed samples
            out[a + 1 : b] = np.interp(
                np.arange(a + 1, b), [a, b], [x[a], x[b]]
            )
            n_filled += gap

    if edge_fill:
        first, last = finite_idx[0], finite_idx[-1]
        if first > 0 and first <= max_gap:
            out[:first] = x[first]
            n_filled += int(first)
        trailing = len(x) - 1 - last
        if trailing > 0 and trailing <= max_gap:
            out[last + 1 :] = x[last]
            n_filled += int(trailing)

    n_left = n_missing - n_filled
    return out, n_filled, n_left


def interpolate_gaps(
    seq: PoseSequence,
    max_gap: int,
    *,
    stage_name: str = "provisional_interpolation",
    edge_fill: bool = False,
) -> PoseSequence:
    """Fill short missing runs by linear interpolation, capped at ``max_gap``.

    Parameters
    ----------
    seq : PoseSequence
        Input sequence (typically the masked signal).
    max_gap : int
        Maximum consecutive-missing run length (in frames) that will be filled.
        Runs longer than this are left as ``NaN``.
    stage_name : str
        Provenance stage name. Defaults to ``"provisional_interpolation"``; the
        later principled pass passes ``"principled_interpolation"``.
    edge_fill : bool
        Also fill leading/trailing missing runs of length ``<= max_gap`` by
        holding the nearest observed value constant. Off by default (see
        :func:`_interpolate_series_run_limited`).

    Returns
    -------
    PoseSequence
        New sequence with short gaps filled and the validity mask/provenance
        updated. Interpolated samples become valid; over-long gaps stay invalid.
    """
    coords = seq.coords.copy()
    total_filled = 0
    total_left = 0
    for k in range(seq.n_keypoints):
        for d in range(seq.dims):
            filled, n_filled, n_left = _interpolate_series_run_limited(
                coords[:, k, d], max_gap, edge_fill
            )
            coords[:, k, d] = filled
            total_filled += n_filled
            total_left += n_left

    mask = np.all(np.isfinite(coords), axis=-1)
    return seq.with_stage(
        stage_name,
        {"max_gap": int(max_gap), "edge_fill": bool(edge_fill)},
        coords=coords,
        mask=mask,
        note=f"filled {total_filled} samples; {total_left} left as missing (> {max_gap} frames)",
    )
