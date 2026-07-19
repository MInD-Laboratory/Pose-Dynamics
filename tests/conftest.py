"""Shared test fixtures: synthetic canonical CSVs for 2D and 3D pose data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def make_canonical_df(
    n_frames: int,
    n_keypoints: int,
    dims: int,
    with_confidence: bool,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a canonical wide DataFrame (x0,y0[,z0][,c0], x1, ...)."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    axes = ("x", "y", "z")[:dims]
    for k in range(n_keypoints):
        for axis in axes:
            data[f"{axis}{k}"] = rng.normal(size=n_frames)
        if with_confidence:
            data[f"c{k}"] = rng.uniform(0.2, 1.0, size=n_frames)
    return pd.DataFrame(data)


def write_canonical_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def face2d_csv(tmp_path: Path) -> Path:
    """2D face: 70 keypoints, with confidence (Case 1 shape)."""
    df = make_canonical_df(120, 70, dims=2, with_confidence=True, seed=1)
    return write_canonical_csv(tmp_path / "face2d.csv", df)


@pytest.fixture
def body2d_csv(tmp_path: Path) -> Path:
    """2D upper body: 25 keypoints, with confidence (Case 2 shape)."""
    df = make_canonical_df(120, 25, dims=2, with_confidence=True, seed=2)
    return write_canonical_csv(tmp_path / "body2d.csv", df)


@pytest.fixture
def body3d_csv(tmp_path: Path) -> Path:
    """3D full body: 38 keypoints, no confidence (Case 3 shape)."""
    df = make_canonical_df(120, 38, dims=3, with_confidence=False, seed=3)
    return write_canonical_csv(tmp_path / "body3d.csv", df)
