"""
Case 3 Principal-Movements (PCA) diagnostic.

Reproduces the manuscript's whole-body PCA: global PCA on the centred,
Procrustes-aligned, Butterworth-filtered 38-keypoint 3-D poses across all trials
and participants, where the first ~14 principal movements capture ~96% of postural
variance. Unlike the CRQA figure, this needs the *aligned coordinates*, so it runs
the full geometry pipeline (recovered from the prototype): centre on pelvis,
canonicalise each trial's mean pose into a body frame, build a global template,
and align each trial with one rigid Procrustes transform.

PCA was used only for exploratory visualization / data-quality confirmation; the
recurrence analyses use the five-keypoint subset, not PC scores.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...diagnostics import PCAModel, fit_pca
from ...features.geometry import body_frame_rotation, procrustes_uniform
from ...preprocessing import butterworth_filter
from . import config as C
from .reproduce import load_and_resample

# Body-frame axis keypoints used by the prototype's canonicalisation.
_PELVIS, _L_SH, _R_SH, _NECK = 0, 10, 11, 3
# zero-phase filtfilt needs more samples than its pad length
# (3*max(len(a), len(b)) - 1 = 26 for the order-4 Butterworth used here).
_MIN_FRAMES = 30


def _canonicalise_mean(pose_mean: np.ndarray) -> np.ndarray:
    """Express a mean pose in its body-fixed frame (pelvis at origin)."""
    P = pose_mean - pose_mean[_PELVIS]
    R = body_frame_rotation(pose_mean, _PELVIS, _L_SH, _R_SH, _NECK)
    return P @ R


def build_aligned_dataset(files: list[str | Path], progress: bool = True):
    """Load, filter, centre, and rigidly align every trial to a global template.

    Returns ``(X, n_keypoints)`` where ``X`` is ``(total_frames, n_keypoints*3)``
    of aligned coordinates ready for PCA.
    """
    # Pass 1: filtered, pelvis-centred sequences + their canonicalised means.
    # Empty or near-empty exports (a truncated recording) cannot be filtered or
    # aligned, so they are skipped and reported rather than crashing the fit.
    seqs, means, skipped = [], [], []
    for i, f in enumerate(files):
        raw = load_and_resample(f)
        if raw.n_frames < _MIN_FRAMES:
            skipped.append((Path(f).name, raw.n_frames))
            continue
        seq = butterworth_filter(raw, C.FILTER_CUTOFF, C.FILTER_ORDER)
        coords = seq.coords - seq.coords[:, _PELVIS:_PELVIS + 1, :]  # centre each frame
        seqs.append(coords)
        means.append(_canonicalise_mean(coords.mean(axis=0)))
        if progress and (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{len(files)}")

    if not seqs:
        raise ValueError("no usable trials: every file was empty or too short.")
    if skipped:
        print(f"  skipped {len(skipped)} unusable trial(s): "
              + ", ".join(f"{n} ({t} frames)" for n, t in skipped))

    # Global template: mean of canonicalised means, refined by Procrustes.
    template0 = np.mean(means, axis=0)
    refined = [procrustes_uniform(m, template0, allow_scale=False).apply(m) for m in means]
    template = np.mean(refined, axis=0)

    # Pass 2: align each trial with one rigid transform (trial mean -> template).
    aligned = []
    for coords in seqs:
        tp = procrustes_uniform(coords.mean(axis=0), template, allow_scale=False)
        a = coords @ tp.L + tp.t                       # one transform for all frames
        aligned.append(a.reshape(a.shape[0], -1))       # (T, K*3)
    X = np.vstack(aligned)
    return X, seqs[0].shape[1]


def run_pca_diagnostic(files: list[str | Path], n_components: int = 20, progress: bool = True):
    """Fit the global PCA and return ``(model, n_keypoints)``."""
    X, n_kp = build_aligned_dataset(files, progress=progress)
    _, model = fit_pca(X, n_components=n_components)
    return model, n_kp


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_variance(model: PCAModel, target: float = 0.96, ax=None):
    """Scree / cumulative-variance plot; marks the components needed for ``target``."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    cum = model.cumulative_variance()
    k = model.n_components_for(target)
    ax.plot(np.arange(1, len(cum) + 1), cum * 100, "-o", ms=4, color="tab:blue")
    ax.axhline(target * 100, color="k", ls="--", lw=1)
    ax.axvline(k, color="tab:red", ls="--", lw=1.4, label=f"{k} PMs = {cum[k-1]*100:.1f}%")
    ax.set_xlabel("number of principal movements")
    ax.set_ylabel("cumulative variance (%)")
    ax.set_title("Global PCA — cumulative variance")
    ax.legend(loc="lower right")
    return ax


def _draw_skeleton(ax, pose: np.ndarray, color: str, lw: float, ms: float, zorder: int) -> None:
    """Render one frontal-view pose as a stick figure (bones + joint markers)."""
    n_kp = pose.shape[0]
    for a, b in C.SKELETON_EDGES:
        if a < n_kp and b < n_kp:
            ax.plot(pose[[a, b], 0], pose[[a, b], 1], "-", color=color, lw=lw,
                    solid_capstyle="round", zorder=zorder)
    ax.plot(pose[:, 0], pose[:, 1], "o", color=color, ms=ms, mec="none", zorder=zorder)


def plot_principal_movements(model: PCAModel, n_kp: int, n_pms: int = 14,
                             target_rms: float = 0.25, axes=None):
    """Visualize the first ``n_pms`` PMs as min (grey) vs max (black) frontal postures.

    Each PM is amplified so that the RMS per-keypoint displacement equals
    ``target_rms`` (manuscript: 0.25 units), then rendered as ``mean ± a·PM``
    stick figures joined by the ZED skeleton bones. Every panel shares one set of
    axis limits so postures are directly comparable across PMs, and the panel
    frames are removed.
    """
    import matplotlib.pyplot as plt

    ncols = 7
    nrows = int(np.ceil(n_pms / ncols))
    if axes is None:
        _, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.9 * nrows))
    axes = np.asarray(axes).flatten()

    mean_pose = model.mean_.reshape(n_kp, 3)

    # Pass 1: amplified extreme postures for every PM, so the panels can share
    # one common scale rather than each auto-scaling to its own extent.
    poses = []
    for i in range(n_pms):
        comp = model.components_[i].reshape(n_kp, 3)
        # amplitude so RMS keypoint displacement == target_rms (unit-norm comp)
        a = target_rms * np.sqrt(n_kp) / (np.linalg.norm(comp) + 1e-12)
        poses.append((mean_pose - a * comp, mean_pose + a * comp))

    allp = np.concatenate([p for pair in poses for p in pair])
    pad = 0.06 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]))
    xc, yc = allp[:, 0].mean(), allp[:, 1].mean()
    half = 0.5 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) + pad
    xlim = (xc - half, xc + half)
    ylim = (yc - half, yc + half)

    for i, (lo, hi) in enumerate(poses):
        ax = axes[i]
        # frontal view: x horizontal, y vertical; grey = min, black = max
        _draw_skeleton(ax, lo, "0.75", lw=3.0, ms=4.0, zorder=1)
        _draw_skeleton(ax, hi, "black", lw=3.0, ms=4.0, zorder=2)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"PC{i+1}", fontsize=10)
        ax.text(0.02, -0.02, f"{model.explained_variance_ratio_[i]*100:.1f}%",
                transform=ax.transAxes, ha="left", va="top", fontsize=9)
    for j in range(n_pms, len(axes)):
        axes[j].set_axis_off()
    axes[0].figure.tight_layout()
    return axes
