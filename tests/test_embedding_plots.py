"""Smoke tests for embedding presentation plots (headless)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from pose_dynamics.embedding import (  # noqa: E402
    Signal,
    plot_embedding_evidence,
    plot_embedding_variability,
    select_embedding,
)


def _signals():
    rng = np.random.default_rng(0)
    sigs = []
    for i in range(15):
        p = 48 + rng.integers(-5, 6)
        t = np.arange(1200)
        x = np.sin(2 * np.pi * t / p) + rng.normal(scale=0.05, size=1200)
        sigs.append(Signal(f"s{i}", x, group={"keypoint": f"kp{i % 3}"}))
    return sigs


def test_evidence_plot_renders():
    ev = select_embedding(_signals())
    ax_ami, ax_fnn = plot_embedding_evidence(ev)
    assert ax_ami.has_data()
    assert ax_fnn.has_data()


def test_variability_plot_renders():
    ev = select_embedding(_signals())
    ax_tau, ax_m = plot_embedding_variability(ev, group_by="keypoint")
    assert ax_tau.has_data()
    assert ax_m.has_data()
