"""
Loading canonical pose CSVs into :class:`PoseSequence`.

The loader is the single entry point from the canonical file format to the data
model. It parses and validates the header (:mod:`.schema`), reads the numeric
body, reshapes into a ``(frames, keypoints, dims)`` array plus an optional
confidence array, and records the ``load`` provenance entry.

There is no 2D-vs-3D branch: the same code loads a 2D face, a 2D upper body, and a
3D full body — the only difference is the ``dims`` reported by the schema.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pose_sequence import PoseSequence
from .schema import PoseSchema, SchemaError, parse_header


def load_pose_csv(
    path: str | Path,
    frame_rate: float,
    *,
    keypoint_names: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> PoseSequence:
    """Load a canonical wide CSV into a :class:`PoseSequence`.

    Parameters
    ----------
    path : str or Path
        Path to the canonical CSV (one row per frame, columns ``x0, y0[, z0]
        [, c0], x1, ...``).
    frame_rate : float
        Sampling rate in Hz. Required; comes from the study config, never inferred.
    keypoint_names : list of str, optional
        Names for the keypoints, length == number of keypoints. If omitted,
        ``kp0..kpK-1`` are used. A wrong-length list is a hard error, since it
        would silently mislabel anatomy.
    meta : dict, optional
        Metadata to attach (participant id, condition, ...).

    Returns
    -------
    PoseSequence

    Raises
    ------
    SchemaError
        If the file is missing, empty, has a non-conforming header, or contains
        non-numeric values — each with an actionable message.
    """
    path = Path(path)
    if not path.exists():
        raise SchemaError(
            f"File not found: {path}. Check the path in your config points to an "
            "existing canonical CSV."
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise SchemaError(
            f"File is empty: {path}. A canonical file needs a header row plus at "
            "least one frame of data."
        ) from exc

    schema = parse_header(list(df.columns))

    if len(df) == 0:
        raise SchemaError(
            f"File has a header but no data rows: {path}. Expected at least one "
            "frame of pose coordinates."
        )

    coords, confidence = _assemble_arrays(df, schema, source=str(path))

    names = _resolve_keypoint_names(keypoint_names, schema.n_keypoints)

    seq = PoseSequence(
        coords=coords,
        keypoint_names=names,
        frame_rate=frame_rate,
        confidence=confidence,
        source_file=str(path),
        meta=dict(meta or {}),
    )
    # Record the load with the facts inferred from the header.
    return seq.with_stage(
        "load",
        params={
            "path": str(path),
            "frame_rate": frame_rate,
            "dims": schema.dims,
            "n_keypoints": schema.n_keypoints,
            "has_confidence": schema.has_confidence,
        },
        note=f"loaded {seq.n_frames} frames",
    )


def _assemble_arrays(
    df: pd.DataFrame,
    schema: PoseSchema,
    source: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Reshape a validated DataFrame into (coords, confidence) arrays."""
    n_frames = len(df)
    coords = np.empty((n_frames, schema.n_keypoints, schema.dims), dtype=float)
    confidence = (
        np.empty((n_frames, schema.n_keypoints), dtype=float)
        if schema.has_confidence
        else None
    )

    for k in range(schema.n_keypoints):
        cols = schema.columns_for[k]
        for axis_idx, axis in enumerate(schema.spatial_axes):
            coords[:, k, axis_idx] = _numeric_column(df, cols[axis], source)
        if confidence is not None:
            confidence[:, k] = _numeric_column(df, cols["c"], source)

    return coords, confidence


def _numeric_column(df: pd.DataFrame, col: str, source: str) -> np.ndarray:
    """Return a column as float, raising an actionable error on bad values."""
    series = df[col]
    converted = pd.to_numeric(series, errors="coerce")
    # A value that was non-empty but failed to parse becomes NaN here while the
    # original was not already NaN/blank -> flag it as a data error.
    newly_nan = converted.isna() & series.notna() & (series.astype(str).str.strip() != "")
    if newly_nan.any():
        bad_rows = list(np.flatnonzero(newly_nan.to_numpy())[:5])
        raise SchemaError(
            f"Column '{col}' in {source} contains non-numeric values "
            f"(first offending row indices: {bad_rows}). Pose coordinates and "
            "confidence must be numbers; blanks are allowed for missing data but "
            "text is not."
        )
    return converted.to_numpy(dtype=float)


def _resolve_keypoint_names(
    keypoint_names: list[str] | None,
    n_keypoints: int,
) -> list[str]:
    """Validate provided names or generate default ``kp{i}`` names."""
    if keypoint_names is None:
        return [f"kp{i}" for i in range(n_keypoints)]
    if len(keypoint_names) != n_keypoints:
        raise SchemaError(
            f"keypoint_names has {len(keypoint_names)} entries but the file has "
            f"{n_keypoints} keypoints. The names must line up one-to-one with the "
            "keypoint columns; fix the skeleton definition in your config."
        )
    return list(keypoint_names)
