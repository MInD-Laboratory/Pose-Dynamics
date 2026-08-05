"""
Bundled synthetic fixture.

A small, deterministic canonical CSV used by the quickstart notebook so a new user
sees the pipeline work *before* their own data can confuse matters. It is designed
so every checkpoint teaches something:

- confidence dropouts create maskable gaps (short ones fillable, one long one not);
- high-frequency noise rides on a clear oscillation, so the filter overlay visibly
  smooths without flattening;
- the signals are quasi-periodic, so AMI shows an interpretable first minimum and
  FNN a knee — the shapes the human-in-the-loop step asks you to read;
- the dynamics are low-dimensional, so the recurrence plot has visible diagonal
  structure.

``example_fixture()`` returns the path to the bundled CSV. ``generate_fixture()``
regenerates it deterministically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_NAME = "synthetic_2d.csv"
FRAME_RATE = 60.0
N_KEYPOINTS = 6
DURATION_S = 90.0


def example_fixture() -> Path:
    """Path to the bundled synthetic canonical CSV (2-D, with confidence)."""
    return Path(__file__).parent / "fixtures" / FIXTURE_NAME


def generate_fixture(seed: int = 0) -> pd.DataFrame:
    """Build the synthetic fixture as a canonical wide DataFrame (deterministic)."""
    rng = np.random.default_rng(seed)
    n = int(DURATION_S * FRAME_RATE)
    t = np.arange(n) / FRAME_RATE

    data: dict[str, np.ndarray] = {}
    for k in range(N_KEYPOINTS):
        # Each keypoint oscillates near a base position at ~1 Hz (period 60 frames),
        # with a per-keypoint phase, slow drift, and high-frequency measurement noise.
        period = 1.0 + 0.15 * k                      # seconds
        phase = 0.4 * k
        base_x, base_y = 100.0 + 40 * k, 200.0 - 20 * k
        osc_x = 30 * np.sin(2 * np.pi * t / period + phase)
        osc_y = 20 * np.cos(2 * np.pi * t / period + phase)
        drift = 5 * np.sin(2 * np.pi * t / 45.0)     # slow postural drift
        hf_noise_x = 2.0 * rng.standard_normal(n)    # filterable jitter
        hf_noise_y = 2.0 * rng.standard_normal(n)

        x = base_x + osc_x + drift + hf_noise_x
        y = base_y + osc_y + drift + hf_noise_y

        # confidence: high, with a few dropouts (low confidence -> masked as missing)
        conf = np.full(n, 0.92) + 0.03 * rng.standard_normal(n)
        # short fillable gaps (~0.3 s) scattered on some keypoints
        for start in rng.integers(200, n - 200, size=3):
            conf[start : start + int(0.3 * FRAME_RATE)] = 0.10
        # one long gap (> 1 s) on keypoint 2, which interpolation must leave missing
        if k == 2:
            conf[int(30 * FRAME_RATE) : int(31.5 * FRAME_RATE)] = 0.05

        data[f"x{k}"] = x
        data[f"y{k}"] = y
        data[f"c{k}"] = np.clip(conf, 0.0, 1.0)

    return pd.DataFrame(data)


def write_fixture(path: str | Path | None = None, seed: int = 0) -> Path:
    """Generate and write the fixture CSV (used to (re)create the bundled file)."""
    path = Path(path) if path is not None else example_fixture()
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_fixture(seed=seed).to_csv(path, index=False, float_format="%.3f")
    return path


def example_dataset(n_trials: int = 3, out_dir: str | Path | None = None) -> Path:
    """Write a small folder of synthetic canonical CSVs and return its path.

    Deterministic (one seed per trial). Used by the "run your dataset" notebook to
    demonstrate a batch run before the user points it at their own folder.
    """
    import tempfile

    out_dir = Path(out_dir) if out_dir is not None else Path(tempfile.gettempdir()) / "pose_dynamics_example_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_trials):
        write_fixture(out_dir / f"trial{i + 1:02d}.csv", seed=i)
    return out_dir
