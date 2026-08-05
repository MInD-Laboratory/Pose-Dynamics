"""
Canonical CSV schema: parsing and validation.

The package supports **one** input format (build plan §2): a wide CSV with one
row per frame and one file per person per trial. Columns are grouped per keypoint
as ``x0, y0[, z0][, c0], x1, y1[, z1][, c1], ...`` where the axis letter is one of
``x``/``y``/``z``/``c`` (``c`` = per-keypoint confidence) and the trailing integer
is the 0-based keypoint index.

Crucially, **dimensionality (2D vs 3D) and the presence of confidence are inferred
from the header, not configured**. This module does that inference and validates
the header, raising :class:`SchemaError` with messages a non-programmer can act on.

There is no 2D-vs-3D branch here or downstream: the parser simply reports how many
spatial axes each keypoint has (``dims`` = 2 or 3), and the rest of the package
treats ``dims`` as data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Axis letters recognized in the canonical header. x/y are mandatory spatial
# axes; z is the optional third spatial axis; c is the optional confidence.
_SPATIAL_AXES = ("x", "y", "z")
_COLUMN_RE = re.compile(r"^(?P<axis>[xyzc])(?P<index>\d+)$")


class SchemaError(ValueError):
    """Raised when a CSV header does not match the canonical schema.

    The message is written for a researcher who is not a programmer: it says
    what is wrong and what to do about it, and it names the offending columns.
    """


@dataclass(frozen=True)
class PoseSchema:
    """The result of parsing a canonical header.

    Attributes
    ----------
    n_keypoints : int
        Number of keypoints (K). Indices are 0-based and contiguous: ``0..K-1``.
    dims : int
        Number of spatial dimensions per keypoint (2 or 3).
    has_confidence : bool
        Whether a per-keypoint confidence column (``c{k}``) is present.
    columns_for : dict
        Mapping ``keypoint_index -> {axis_letter: column_name}`` giving the exact
        source column names, so the loader can pull values in a stable order.
    """

    n_keypoints: int
    dims: int
    has_confidence: bool
    columns_for: dict[int, dict[str, str]]

    @property
    def spatial_axes(self) -> tuple[str, ...]:
        """The spatial axis letters present, in canonical order (x, y[, z])."""
        return _SPATIAL_AXES[: self.dims]


def parse_header(columns: list[str]) -> PoseSchema:
    """Parse and validate a canonical CSV header.

    Parameters
    ----------
    columns : list of str
        The column names, in file order. Leading/trailing whitespace is ignored.

    Returns
    -------
    PoseSchema
        The inferred schema.

    Raises
    ------
    SchemaError
        If the header does not conform, with an actionable message.
    """
    stripped = [c.strip() for c in columns]

    # 1) Every column must match the <axis><index> pattern.
    per_keypoint: dict[int, dict[str, str]] = {}
    unrecognized: list[str] = []
    for col in stripped:
        m = _COLUMN_RE.match(col)
        if not m:
            unrecognized.append(col)
            continue
        axis = m.group("axis")
        index = int(m.group("index"))
        axes = per_keypoint.setdefault(index, {})
        if axis in axes:
            raise SchemaError(
                f"Duplicate column '{col}': keypoint {index} already has an "
                f"'{axis}' column. Each keypoint must have each axis at most once."
            )
        axes[axis] = col

    if unrecognized:
        raise SchemaError(
            "These column names are not in the expected format: "
            f"{unrecognized}. Every column must be an axis letter followed by a "
            "keypoint number, e.g. 'x0', 'y0', 'z0' (3D), or 'c0' (confidence). "
            "Rename or remove any extra columns (timestamps, labels, etc.) before "
            "loading; the canonical file holds pose coordinates only."
        )

    if not per_keypoint:
        raise SchemaError(
            "No keypoint columns found. Expected a wide CSV with columns like "
            "'x0, y0, x1, y1, ...' (one group per keypoint)."
        )

    # 2) Keypoint indices must be contiguous and 0-based (0..K-1).
    indices = sorted(per_keypoint)
    n_keypoints = len(indices)
    expected = list(range(n_keypoints))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        extra = sorted(set(indices) - set(expected))
        parts = []
        if missing:
            parts.append(f"missing keypoint number(s) {missing}")
        if extra:
            parts.append(f"out-of-range keypoint number(s) {extra}")
        raise SchemaError(
            "Keypoint numbers must start at 0 and be consecutive with no gaps "
            f"(0 to {n_keypoints - 1}), but the header has {', '.join(parts)}. "
            "Check for a skipped or mislabeled keypoint."
        )

    # 3) Determine spatial dimensionality from the FIRST keypoint, then require
    #    every keypoint to agree. Mixed 2D/3D is a hard error, not a silent guess.
    def _spatial_of(k: int) -> tuple[str, ...]:
        return tuple(a for a in _SPATIAL_AXES if a in per_keypoint[k])

    first_spatial = _spatial_of(0)
    if first_spatial not in (("x", "y"), ("x", "y", "z")):
        have = "".join(sorted(per_keypoint[0]))
        raise SchemaError(
            f"Keypoint 0 must have at least 'x0' and 'y0' columns (and optionally "
            f"'z0' for 3D), but its columns are '{have}'. Every keypoint needs x "
            "and y at minimum."
        )
    dims = len(first_spatial)

    mismatched: list[int] = []
    conf_flags: set[bool] = set()
    for k in indices:
        if _spatial_of(k) != first_spatial:
            mismatched.append(k)
        conf_flags.add("c" in per_keypoint[k])

    if mismatched:
        raise SchemaError(
            f"Keypoint 0 is {dims}D (axes {list(first_spatial)}), but keypoint(s) "
            f"{mismatched} have a different set of spatial axes. All keypoints in a "
            "file must share the same dimensionality — a file is either fully 2D or "
            "fully 3D."
        )

    # 4) Confidence must be all-or-nothing across keypoints.
    if len(conf_flags) != 1:
        with_conf = [k for k in indices if "c" in per_keypoint[k]]
        without = [k for k in indices if "c" not in per_keypoint[k]]
        raise SchemaError(
            "Confidence columns must be present for either all keypoints or none. "
            f"Keypoint(s) {with_conf} have a 'c' column but {without} do not. "
            "Add the missing confidence columns or remove all of them."
        )
    has_confidence = conf_flags.pop()

    return PoseSchema(
        n_keypoints=n_keypoints,
        dims=dims,
        has_confidence=has_confidence,
        columns_for=per_keypoint,
    )
