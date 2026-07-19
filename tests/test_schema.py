"""Tests for canonical header parsing and validation."""
from __future__ import annotations

import pytest

from pose_dynamics.data.schema import SchemaError, parse_header


def test_infers_2d_no_confidence():
    schema = parse_header(["x0", "y0", "x1", "y1"])
    assert schema.dims == 2
    assert schema.n_keypoints == 2
    assert schema.has_confidence is False
    assert schema.spatial_axes == ("x", "y")


def test_infers_2d_with_confidence():
    schema = parse_header(["x0", "y0", "c0", "x1", "y1", "c1"])
    assert schema.dims == 2
    assert schema.has_confidence is True
    assert schema.columns_for[1]["c"] == "c1"


def test_infers_3d_with_confidence():
    schema = parse_header(["x0", "y0", "z0", "c0", "x1", "y1", "z1", "c1"])
    assert schema.dims == 3
    assert schema.has_confidence is True
    assert schema.spatial_axes == ("x", "y", "z")


def test_infers_3d_no_confidence():
    schema = parse_header(["x0", "y0", "z0", "x1", "y1", "z1"])
    assert schema.dims == 3
    assert schema.has_confidence is False


def test_header_whitespace_is_tolerated():
    schema = parse_header([" x0 ", "y0", "x1 ", " y1"])
    assert schema.n_keypoints == 2


def test_column_order_independent():
    # Axes may be interleaved in any order as long as names are unambiguous.
    schema = parse_header(["y0", "x0", "y1", "x1"])
    assert schema.dims == 2
    assert schema.n_keypoints == 2


def test_rejects_unrecognized_column():
    with pytest.raises(SchemaError, match="not in the expected format"):
        parse_header(["x0", "y0", "timestamp"])


def test_rejects_noncontiguous_keypoints():
    with pytest.raises(SchemaError, match="consecutive"):
        parse_header(["x0", "y0", "x2", "y2"])


def test_rejects_not_starting_at_zero():
    with pytest.raises(SchemaError, match="consecutive"):
        parse_header(["x1", "y1", "x2", "y2"])


def test_rejects_mixed_dimensionality():
    with pytest.raises(SchemaError, match="same dimensionality|fully 2D or"):
        parse_header(["x0", "y0", "z0", "x1", "y1"])


def test_rejects_partial_confidence():
    with pytest.raises(SchemaError, match="all keypoints or none"):
        parse_header(["x0", "y0", "c0", "x1", "y1"])


def test_rejects_missing_y():
    with pytest.raises(SchemaError, match="x and y|at least"):
        parse_header(["x0", "x1", "y1"])


def test_rejects_duplicate_axis():
    with pytest.raises(SchemaError, match="Duplicate"):
        parse_header(["x0", "y0", "x0", "y1"])


def test_rejects_empty_header():
    with pytest.raises(SchemaError, match="No keypoint columns"):
        parse_header([])
