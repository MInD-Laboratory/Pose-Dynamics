"""Smoke tests for preprocessing checkpoint plots (headless)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np  # noqa: E402

from pose_dynamics.data import PoseSequence  # noqa: E402
from pose_dynamics.preprocessing import (  # noqa: E402
    assess_quality,
    butterworth_filter,
    interpolate_gaps,
    mask_low_confidence,
    plot_filter_checkpoint,
    plot_interpolation_checkpoint,
    plot_masking_checkpoint,
)


def _demo_seq():
    rng = np.random.default_rng(0)
    T, K = 300, 2
    conf = rng.uniform(0.4, 1.0, size=(T, K))
    conf[40:50, 0] = 0.1
    conf[120:210, 1] = 0.1
    t = np.arange(T) / 60.0
    coords = np.sin(2 * np.pi * 1.0 * t)[:, None, None] * np.ones((1, K, 2))
    return PoseSequence(
        coords=coords + rng.normal(scale=0.05, size=(T, K, 2)),
        keypoint_names=[f"kp{k}" for k in range(K)],
        frame_rate=60.0,
        confidence=conf,
        source_file="demo.csv",
    )


def test_all_checkpoint_plots_render():
    seq = _demo_seq()
    masked = mask_low_confidence(seq, threshold=0.30)
    interp = interpolate_gaps(masked, max_gap=60)
    filt = butterworth_filter(interp, cutoff_hz=10.0, order=4)
    rep = assess_quality(interp, on_exceed="flag")

    ax1 = plot_masking_checkpoint(seq, masked, keypoint=0)
    ax2 = plot_interpolation_checkpoint(masked, interp, keypoint=1)
    ax3 = plot_filter_checkpoint(interp, filt, keypoint=0)
    ax4 = rep.plot()
    ax5 = seq.plot_coverage()

    for ax in (ax1, ax2, ax3, ax4, ax5):
        assert ax is not None
        assert ax.has_data()
