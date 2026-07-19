"""
Confidence masking stage.

Marks low-confidence keypoint observations as missing (``NaN``) so they are
excluded from interpolation, filtering, and analysis. This is the first
preprocessing stage after loading (build plan §3).

The stage is dimension-agnostic: it operates on the ``(frames, keypoints)``
confidence array and blanks the whole coordinate vector of a low-confidence
keypoint, whether that vector is 2D or 3D.
"""
from __future__ import annotations

import numpy as np

from ..data.pose_sequence import PoseSequence


def mask_low_confidence(seq: PoseSequence, threshold: float) -> PoseSequence:
    """Set keypoints with confidence below ``threshold`` to missing.

    Parameters
    ----------
    seq : PoseSequence
        Input sequence. Must carry a confidence array; if it does not, the stage
        is a no-op that records why (some modalities, e.g. Case 3 ZED, have no
        confidence and no masking).
    threshold : float
        Minimum confidence to keep. Observations with ``confidence < threshold``
        have all their coordinates set to ``NaN`` and are marked invalid.

    Returns
    -------
    PoseSequence
        New sequence with low-confidence points blanked and the validity mask and
        provenance updated.
    """
    if seq.confidence is None:
        return seq.with_stage(
            "confidence_mask",
            {"threshold": threshold, "applied": False},
            note="no confidence array; masking skipped",
        )

    low = seq.confidence < threshold  # (frames, keypoints)
    coords = seq.coords.copy()
    coords[low, :] = np.nan
    mask = np.all(np.isfinite(coords), axis=-1) & seq.mask

    n_masked = int(low.sum())
    total = low.size
    return seq.with_stage(
        "confidence_mask",
        {"threshold": threshold, "applied": True},
        coords=coords,
        mask=mask,
        note=f"masked {n_masked}/{total} observations ({n_masked / total:.2%})",
    )
