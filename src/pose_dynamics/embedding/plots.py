"""
Presentation plots for the human-in-the-loop embedding decision.

The framework proposes ``(τ, m)`` but the researcher commits it, so these plots
must make the evidence legible: every signal's curve, the aggregate, the spread,
the bounded grid, and the proposal — all in one view. A companion variability plot
shows whether the per-signal suggestions drift systematically across keypoints,
participants, or conditions (does holding ``(τ, m)`` fixed stay honest?).

``matplotlib`` is imported lazily so the core install stays light.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .selection import EmbeddingEvidence


def plot_embedding_evidence(evidence: EmbeddingEvidence, axes=None):
    """Two-panel AMI + FNN evidence figure with the proposal marked.

    Returns the ``(ax_ami, ax_fnn)`` axes. Draw the justification alongside with
    ``print(evidence.justification)``.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax_ami, ax_fnn = axes

    _plot_ami(ax_ami, evidence)
    _plot_fnn(ax_fnn, evidence)
    return ax_ami, ax_fnn


def _faint_curves(ax, x, curves, color):
    for row in curves:
        finite = np.isfinite(row)
        if finite.any():
            ax.plot(x[finite], row[finite], color=color, alpha=0.12, lw=0.6)


def _plot_ami(ax, ev: EmbeddingEvidence):
    x = ev.ami_lags
    band = ev.ami_summary()
    _faint_curves(ax, x, ev.ami_curves, "tab:blue")
    ax.fill_between(x, band["p10"], band["p90"], color="tab:blue", alpha=0.18,
                    label="10-90% spread")
    ax.plot(x, band["median"], color="tab:blue", lw=2.2, label="median")

    lo, hi = ev.tau_grid
    ax.axvspan(lo, hi, color="0.85", alpha=0.5, zorder=0, label=f"grid τ∈[{lo},{hi}]")
    ax.axvline(ev.proposed_tau, color="tab:red", ls="--", lw=1.6,
               label=f"proposed τ={ev.proposed_tau}")
    ax.set_xlabel("delay τ (frames)")
    ax.set_ylabel("AMI (bits)")
    ax.set_title(f"Average Mutual Information  (n={ev.n_signals_used} signals)")
    ax.legend(loc="upper right", fontsize=8)


def _plot_fnn(ax, ev: EmbeddingEvidence):
    if ev.fnn_curves.size == 0:
        ax.text(0.5, 0.5, "FNN unavailable\n(signals too short)", ha="center", va="center")
        ax.set_axis_off()
        return
    x = ev.fnn_dims
    band = ev.fnn_summary()
    _faint_curves(ax, x, ev.fnn_curves, "tab:green")
    ax.fill_between(x, band["p10"], band["p90"], color="tab:green", alpha=0.18,
                    label="10-90% spread")
    ax.plot(x, band["median"], color="tab:green", lw=2.2, marker="o", ms=4, label="median")

    lo, hi = ev.m_grid
    ax.axvspan(lo, hi, color="0.85", alpha=0.5, zorder=0, label=f"grid m∈[{lo},{hi}]")
    ax.axhline(ev.fnn_tol, color="0.4", ls=":", lw=1.2, label=f"tol {ev.fnn_tol:.0f}%")
    ax.axvline(ev.proposed_m, color="tab:red", ls="--", lw=1.6,
               label=f"proposed m={ev.proposed_m}")
    ax.set_xlabel("embedding dimension m")
    ax.set_ylabel("false nearest neighbours (%)")
    ax.set_title(f"False Nearest Neighbours  (τ={ev.fnn_tau})")
    ax.legend(loc="upper right", fontsize=8)


def plot_embedding_variability(evidence: EmbeddingEvidence, group_by: str, axes=None):
    """Per-signal suggested ``τ`` and ``m`` grouped by a metadata key.

    ``group_by`` is a key present in each signal's ``group`` dict (e.g.
    ``"keypoint"``, ``"trial"``, or a condition label). Systematic drift across
    groups is a cue that a single fixed ``(τ, m)`` may not be honest.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax_tau, ax_m = axes

    groups_tau: dict[Any, list[float]] = defaultdict(list)
    groups_m: dict[Any, list[float]] = defaultdict(list)
    for g, t, m in zip(evidence.groups, evidence.per_signal_tau, evidence.per_signal_m):
        key = g.get(group_by, "?")
        if np.isfinite(t):
            groups_tau[key].append(t)
        if np.isfinite(m):
            groups_m[key].append(m)

    _grouped_box(ax_tau, groups_tau, evidence.proposed_tau, evidence.tau_grid, group_by, "τ")
    _grouped_box(ax_m, groups_m, evidence.proposed_m, evidence.m_grid, group_by, "m")
    return ax_tau, ax_m


def _grouped_box(ax, groups: dict[Any, list[float]], proposed, grid, group_by, symbol):
    keys = list(groups.keys())
    data = [groups[k] for k in keys]
    if not data:
        ax.text(0.5, 0.5, "no per-signal suggestions", ha="center", va="center")
        ax.set_axis_off()
        return
    positions = np.arange(len(keys))
    ax.boxplot(data, positions=positions, widths=0.6, showfliers=False)
    for i, vals in enumerate(data):
        jitter = (np.random.default_rng(i).uniform(-0.15, 0.15, size=len(vals)))
        ax.scatter(positions[i] + jitter, vals, s=8, color="tab:blue", alpha=0.4, zorder=3)
    ax.axhspan(grid[0], grid[1], color="0.85", alpha=0.5, zorder=0)
    ax.axhline(proposed, color="tab:red", ls="--", lw=1.4, label=f"proposed {symbol}={proposed}")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(k) for k in keys], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel(group_by)
    ax.set_ylabel(f"per-signal suggested {symbol}")
    ax.set_title(f"Variability of {symbol} across {group_by}")
    ax.legend(loc="upper right", fontsize=8)
