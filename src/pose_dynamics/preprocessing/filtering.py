"""
Zero-phase Butterworth low-pass filtering.

Filtering follows the recovered prototype behavior (numeric inventory §1.7–1.9):
a zero-phase (``filtfilt``) Butterworth low-pass is applied, and — crucially —
**gaps are temporarily filled to keep the filter continuous, then reinstated as
missing afterwards**. Without this, ``filtfilt`` would propagate ``NaN`` across the
whole signal.

The temporary fill uses forward/backward fill of the last/next observed value,
matching the manuscript's description ("gaps are temporarily filled ... then
reinstated"). Only the observed/interpolated samples receive filtered values; the
long gaps left by the interpolation stage are set back to ``NaN``.

Dimension-agnostic: the coordinate array is flattened to ``(frames, keypoints*dims)``
and every channel is filtered identically, so 2D and 3D share one path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from ..data.pose_sequence import PoseSequence


def butterworth_filter(
    seq: PoseSequence,
    cutoff_hz: float,
    order: int = 4,
    btype: str = "low",
    by_segment: bool = False,
) -> PoseSequence:
    """Apply a zero-phase Butterworth filter, preserving missing-data gaps.

    Parameters
    ----------
    seq : PoseSequence
        Input sequence (typically after provisional interpolation).
    cutoff_hz : float
        Cutoff frequency in Hz. Must be below Nyquist (``frame_rate / 2``).
    order : int
        Filter order (default 4).
    btype : str
        Filter type passed to ``scipy.signal.butter`` (default ``"low"``).
    by_segment : bool
        Filter each contiguous finite run independently instead of temporarily
        filling gaps and filtering the whole series. Matters only when gaps were
        left unfilled; see the implementation note. Off by default.

    Returns
    -------
    PoseSequence
        New sequence with filtered coordinates; ``NaN`` gaps are reinstated at
        their original positions and the validity mask is unchanged.
    """
    fs = seq.frame_rate
    nyq = fs / 2.0
    if not (0 < cutoff_hz < nyq):
        raise ValueError(
            f"cutoff_hz ({cutoff_hz} Hz) must be between 0 and Nyquist "
            f"({nyq} Hz at frame_rate {fs} Hz). Lower the cutoff or raise the "
            "frame rate."
        )

    b, a = signal.butter(order, cutoff_hz / nyq, btype=btype, analog=False)

    T = seq.n_frames
    flat = seq.coords.reshape(T, -1)          # (T, K*D)
    nan_flat = ~np.isfinite(flat)
    out = flat.copy()

    if by_segment:
        # Filter each contiguous finite run independently, leaving runs too short
        # for filtfilt's edge padding untouched. This differs from the whole-series
        # mode around gaps that were left unfilled: there, the temporary ffill/bfill
        # below creates a constant plateau whose smoothing bleeds into the observed
        # samples on either side. Segment-wise filtering treats each side as its own
        # signal instead. Case 1 uses this to match its source pipeline.
        padlen = 3 * (max(len(a), len(b)) - 1)
        for c in range(flat.shape[1]):
            col = out[:, c]
            i = 0
            while i < T:
                while i < T and not np.isfinite(col[i]):
                    i += 1
                if i >= T:
                    break
                j = i
                while j < T and np.isfinite(col[j]):
                    j += 1
                if j - i > padlen:
                    try:
                        col[i:j] = signal.filtfilt(b, a, col[i:j])
                    except ValueError:
                        pass
                i = j
    else:
        # Temporary fill for continuity: forward then backward fill each channel.
        filled = pd.DataFrame(flat).ffill().bfill().to_numpy()
        # Channels that had at least one observed sample are now fully finite and
        # can be filtered; channels that were entirely missing stay NaN.
        valid_cols = np.isfinite(filled).all(axis=0)
        if valid_cols.any():
            # filtfilt requires the signal to be longer than the edge padding.
            default_padlen = 3 * max(len(a), len(b))
            padlen = min(default_padlen, T - 1) if T > 1 else 0
            filtered = signal.filtfilt(
                b, a, filled[:, valid_cols], axis=0, padlen=padlen
            )
            out[:, valid_cols] = filtered

    # Reinstate gaps as missing.
    out[nan_flat] = np.nan
    coords = out.reshape(seq.coords.shape)

    return seq.with_stage(
        "butterworth_filter",
        {"cutoff_hz": float(cutoff_hz), "order": int(order), "btype": btype,
         "fs": float(fs), "by_segment": bool(by_segment)},
        coords=coords,
        note=f"zero-phase order-{order} {btype}-pass at {cutoff_hz} Hz"
             + (" (per finite segment)" if by_segment else ""),
    )
