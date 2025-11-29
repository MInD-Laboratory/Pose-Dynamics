# utils/viz_utils.py
"""
Generic visualization utilities for pose / time-series analysis.

This module is intentionally kept free of experiment-specific assumptions
(filenames, conditions, dataset layouts, etc.). It only depends on NumPy,
Pandas, and Matplotlib.

Typical uses:
- Compare raw vs filtered trajectories
- Make quick diagnostic videos of preprocessing
- Draw skeletons from (n_points, 3) arrays and an edge list
- Visualize PCA movement components as animated skeletons
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ----------------------------------------------------------------------
# Basic stats helper
# ----------------------------------------------------------------------

def sem(series: Sequence[float]) -> float:
    """Standard error of the mean (ignores NaNs)."""
    s = pd.Series(series, dtype=float)
    return s.std(ddof=1) / np.sqrt(s.count()) if s.count() > 0 else np.nan


# ----------------------------------------------------------------------
# Time-series / filtering diagnostics
# ----------------------------------------------------------------------

def plot_filtering_comparison(
    raw_array: np.ndarray,
    filtered_array: np.ndarray,
    fps: float,
    keypoint_labels: Sequence[str],
    keypoint_indices: Sequence[int],
    save_path: Optional[str | Path] = None,
):
    """
    Plot raw vs filtered trajectories for selected keypoints and coordinates.

    Parameters
    ----------
    raw_array : (T, n_points*D)
        Raw flattened pose array (D=2 or 3).
    filtered_array : (T, n_points*D)
        Filtered flattened pose array, same shape as raw_array.
    fps : float
        Sampling rate (Hz).
    keypoint_labels : list of str
        Labels for each keypoint to plot (for titles).
    keypoint_indices : list of int
        Indices of keypoints (0-based) to plot.
    save_path : str or Path, optional
        If provided, save figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    raw_array = np.asarray(raw_array, float)
    filtered_array = np.asarray(filtered_array, float)

    if raw_array.shape != filtered_array.shape:
        raise ValueError("raw_array and filtered_array must have the same shape")

    n_frames, n_dim = raw_array.shape
    n_points = n_dim // 3  # assumes 3D; will still work if Z is unused/zero
    time = np.arange(n_frames) / fps

    raw_3d = raw_array.reshape(n_frames, n_points, 3)
    filt_3d = filtered_array.reshape(n_frames, n_points, 3)

    n_keypoints = len(keypoint_indices)
    fig, axes = plt.subplots(n_keypoints, 3, figsize=(15, 3 * n_keypoints), squeeze=False)

    coord_names = ["X", "Y", "Z"]

    for row, (label, pt_idx) in enumerate(zip(keypoint_labels, keypoint_indices)):
        for coord in range(3):
            ax = axes[row, coord]

            # Guard against out-of-range indices
            if pt_idx < 0 or pt_idx >= n_points:
                ax.text(0.5, 0.5, f"keypoint {pt_idx} out of range",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
                continue

            raw_traj = raw_3d[:, pt_idx, coord]
            filt_traj = filt_3d[:, pt_idx, coord]

            ax.plot(time, raw_traj, alpha=0.5, linewidth=1, label="Raw")
            ax.plot(time, filt_traj, linewidth=2, label="Filtered")

            # SNR (guarding against zeros)
            noise = raw_traj - filt_traj
            rms_noise = np.sqrt(np.nanmean(noise ** 2))
            rms_signal = np.sqrt(np.nanmean(filt_traj ** 2))
            if rms_noise > 0 and rms_signal > 0:
                snr_db = 20 * np.log10(rms_signal / rms_noise)
            else:
                snr_db = np.nan

            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel(f"{coord_names[coord]} position", fontsize=9)
            ax.set_title(f"{label} – {coord_names[coord]} (SNR: {snr_db:.1f} dB)", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved filtering comparison to {save_path}")

    return fig


def create_preprocessing_video(
    raw_array: np.ndarray,
    filtered_array: np.ndarray,
    fps: float,
    keypoint_labels: Sequence[str],
    keypoint_indices: Sequence[int],
    skeleton_edges: Sequence[Tuple[int, int]],
    duration_sec: float = 10.0,
    save_path: str | Path = "preprocessing_demo.mp4",
):
    """
    Create an animated video showing a 3D skeleton (filtered) on the left
    and time-series traces (raw vs filtered) on the right.

    Parameters
    ----------
    raw_array : (T, n_points*3)
        Raw pose data, flattened.
    filtered_array : (T, n_points*3)
        Filtered pose data, flattened.
    fps : float
        Sampling rate of the *pose* data (and video).
    keypoint_labels : list of str
        Labels for time-series panels.
    keypoint_indices : list of int
        Indices of keypoints to show traces for.
    skeleton_edges : list of (i, j)
        Pairs of keypoint indices defining the skeleton edges.
    duration_sec : float
        Duration of the video in seconds (capped by data length).
    save_path : str or Path
        Output .mp4 path (requires ffmpeg installed).

    Returns
    -------
    None
    """
    raw_array = np.asarray(raw_array, float)
    filtered_array = np.asarray(filtered_array, float)

    if raw_array.shape != filtered_array.shape:
        raise ValueError("raw_array and filtered_array must have the same shape")

    n_points = raw_array.shape[1] // 3
    n_frames = min(int(duration_sec * fps), raw_array.shape[0])

    raw_3d = raw_array[:n_frames].reshape(n_frames, n_points, 3)
    filt_3d = filtered_array[:n_frames].reshape(n_frames, n_points, 3)

    # Simple camera→world transform: (x,y,z) -> (X,Y,Z) with Z vertical
    def to_world_coords(pts: np.ndarray) -> np.ndarray:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        X = x          # left-right
        Y = -z         # depth (flip so positive is "towards viewer")
        Z = y          # up
        return np.stack([X, Y, Z], axis=-1)

    # Precompute global bounds
    all_pts_world = to_world_coords(filt_3d.reshape(-1, 3))
    x_min, x_max = all_pts_world[:, 0].min(), all_pts_world[:, 0].max()
    y_min, y_max = all_pts_world[:, 1].min(), all_pts_world[:, 1].max()
    z_min, z_max = all_pts_world[:, 2].min(), all_pts_world[:, 2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.6

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    fig = plt.figure(figsize=(16, 8))

    # Left: 3D skeleton
    ax_skel = fig.add_subplot(121, projection="3d")

    # Right: time series for selected keypoints
    n_ts = len(keypoint_indices)
    grid = fig.add_gridspec(n_ts, 2, left=0.55, right=0.95, hspace=0.3)
    ax_ts = [fig.add_subplot(grid[i, :]) for i in range(n_ts)]

    time = np.arange(n_frames) / fps
    lines_raw, lines_filt, lines_current = [], [], []

    for ax, label, pt_idx in zip(ax_ts, keypoint_labels, keypoint_indices):
        coord = 1  # vertical in raw data (y)

        raw_traj = raw_3d[:, pt_idx, coord]
        filt_traj = filt_3d[:, pt_idx, coord]

        line_raw, = ax.plot(time, raw_traj, "r-", alpha=0.3, linewidth=1, label="Raw")
        line_filt, = ax.plot(time, filt_traj, "b-", alpha=0.3, linewidth=1, label="Filtered")
        line_curr, = ax.plot([], [], "ko", markersize=6, label="Current")

        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Vertical position", fontsize=9)
        ax.set_title(f"{label} – vertical position", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, duration_sec)

        lines_raw.append(line_raw)
        lines_filt.append(line_filt)
        lines_current.append(line_curr)

    def update(frame: int):
        ax_skel.clear()

        pts_world = to_world_coords(filt_3d[frame])  # (n_points, 3)

        # Plot points
        ax_skel.scatter(
            pts_world[:, 0], pts_world[:, 1], pts_world[:, 2],
            c="blue", s=25, alpha=0.7
        )

        # Plot edges
        for i, j in skeleton_edges:
            if i < n_points and j < n_points:
                ax_skel.plot(
                    [pts_world[i, 0], pts_world[j, 0]],
                    [pts_world[i, 1], pts_world[j, 1]],
                    [pts_world[i, 2], pts_world[j, 2]],
                    "b-", linewidth=2, alpha=0.6,
                )

        ax_skel.view_init(elev=15, azim=frame * 360 / max(1, n_frames - 1))
        ax_skel.set_xlabel("X", fontsize=9)
        ax_skel.set_ylabel("Y", fontsize=9)
        ax_skel.set_zlabel("Z", fontsize=9)
        ax_skel.set_title(f"3D Skeleton (t={frame / fps:.2f}s)", fontsize=12, fontweight="bold")

        ax_skel.set_xlim(x_mid - max_range, x_mid + max_range)
        ax_skel.set_ylim(y_mid - max_range, y_mid + max_range)
        ax_skel.set_zlim(z_mid - max_range, z_mid + max_range)

        # Update time-series
        for i, (pt_idx, line_raw, line_filt, line_curr, ax) in enumerate(
            zip(keypoint_indices, lines_raw, lines_filt, lines_current, ax_ts)
        ):
            coord = 1
            t_curr = frame / fps
            y_curr = filt_3d[frame, pt_idx, coord]

            line_curr.set_data([t_curr], [y_curr])
            line_raw.set_data(time[: frame + 1], raw_3d[: frame + 1, pt_idx, coord])
            line_filt.set_data(time[: frame + 1], filt_3d[: frame + 1, pt_idx, coord])

        return ax_skel, *lines_current, *lines_raw, *lines_filt

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(save_path, writer=writer)
    plt.close(fig)

    print(f"✓ Saved preprocessing video to {save_path}")
    print(f"  Duration: {duration_sec:.1f}s, FPS: {fps}, Frames: {n_frames}")


# ----------------------------------------------------------------------
# Skeleton drawing helpers (generic)
# ----------------------------------------------------------------------

def draw_skeleton(
    ax,
    points: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    color: str = "k",
    lw: float = 1.5,
    alpha: float = 1.0,
):
    """
    Draw a skeleton on a 3D axis.

    Parameters
    ----------
    ax : matplotlib 3D axis
    points : (n_points, 3)
        XYZ coordinates of joints.
    edges : list of (i, j)
        Pairs of indices defining segments.
    color : str
        Line color.
    lw : float
        Line width.
    alpha : float
        Line alpha.
    """
    points = np.asarray(points, float)
    segs = [(points[i], points[j]) for i, j in edges]
    lc = Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection3d(lc)


def draw_floor(
    ax,
    width: float = 2.0,
    length: float = 3.0,
    z0: float = 0.0,
    color: str = "gray",
    alpha: float = 0.3,
):
    """Draw a simple rectangular floor patch on a 3D axis."""
    x = np.array([-width / 2, width / 2, width / 2, -width / 2, -width / 2])
    y = np.array([-length / 2, -length / 2, length / 2, length / 2, -length / 2])
    z = np.full_like(x, z0)
    ax.plot(x, y, z, color=color, alpha=alpha)


def style_3d_axes(
    ax,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
    box_aspect: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    view_elev: float = 15.0,
    view_azim: float = -70.0,
    hide_ticks: bool = True,
):
    """Apply common styling to 3D skeleton plots."""
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if zlim is not None:
        ax.set_zlim(*zlim)

    ax.set_box_aspect(box_aspect)
    ax.view_init(elev=view_elev, azim=view_azim)

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    ax.grid(False)


# ----------------------------------------------------------------------
# PCA component animation (generic, with optional pose transform)
# ----------------------------------------------------------------------

def create_pm_animation_3dgrid(
    pca,
    X: np.ndarray,
    mean_vector: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    save_path: str | Path,
    n_components: int = 3,
    n_frames: int = 200,
    scale: float = 2.0,
    fps: int = 40,
    pose_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    """
    Create 3D animations of PCA movement components (grid of subplots).

    The PCA is assumed to be fit on flattened pose vectors (n_points*3),
    already centered/normalized as desired. This function:
        - takes principal components,
        - oscillates each one sinusoidally,
        - reconstructs poses around the mean,
        - optionally applies a pose_transform (e.g. camera→world mapping),
        - draws them as animated skeletons.

    Parameters
    ----------
    pca : sklearn-like PCA object
        Must have .components_ and .explained_variance_ratio_ and .transform.
    X : (N, n_points*3)
        Data used to compute PCA (for variance scaling).
    mean_vector : (n_points*3,)
        Mean pose vector (same dimensionality as PCA space).
    edges : list of (i, j)
        Skeleton edges.
    save_path : str or Path
        Output video path (.mp4).
    n_components : int
        Number of leading PCs to animate.
    n_frames : int
        Number of animation frames.
    scale : float
        Scaling factor on PC amplitude (in SD units).
    fps : int
        Frames per second for video.
    pose_transform : callable, optional
        Function `P -> P_disp`, where P is (n_points, 3) in PCA space,
        P_disp is in display coordinates. If None, identity is used.

    Returns
    -------
    None
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mean_vector = np.asarray(mean_vector, float)
    X = np.asarray(X, float)
    n_points = mean_vector.size // 3

    if pose_transform is None:
        def pose_transform(P):  # type: ignore[redefined-outer-name]
            return P

    # PC scores to get typical variance per component
    pc_scores = pca.transform(X)
    t = np.linspace(0, 2 * np.pi, n_frames)

    all_positions = []
    for i in range(min(n_components, pca.components_.shape[0])):
        # amplitude scaled by std of scores for that PC
        amp = scale * np.std(pc_scores[:, i])
        coeffs = np.sin(t) * amp  # (n_frames,)
        reconstruction = np.outer(coeffs, pca.components_[i]) + mean_vector  # (n_frames, D)
        positions = reconstruction.reshape(n_frames, n_points, 3)  # (n_frames, n_points, 3)
        all_positions.append(positions)
    all_positions = np.array(all_positions)  # (n_components, n_frames, n_points, 3)

    # Prepare global bounds in display space
    all_disp = []
    for i in range(all_positions.shape[0]):
        for f in range(all_positions.shape[1]):
            all_disp.append(pose_transform(all_positions[i, f]))
    all_disp = np.stack(all_disp, axis=0)  # (n_components*n_frames, n_points, 3)
    mins = all_disp.reshape(-1, 3).min(axis=0)
    maxs = all_disp.reshape(-1, 3).max(axis=0)
    pad = 0.05 * (maxs - mins)
    mins -= pad
    maxs += pad

    # Mean pose in display space
    base_pose = pose_transform(mean_vector.reshape(n_points, 3))

    # Figure + subplots
    n_components = all_positions.shape[0]
    ncols = min(3, n_components)
    nrows = int(np.ceil(n_components / ncols))
    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    axes, artists = [], []

    for i in range(n_components):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        axes.append(ax)

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        style_3d_axes(ax, hide_ticks=True)

        var_pct = float(pca.explained_variance_ratio_[i] * 100.0)
        ax.set_title(f"PC{i+1}: {var_pct:.1f}% var")

        # Grey mean skeleton
        draw_skeleton(ax, base_pose, edges, color="gray", lw=1.0, alpha=0.6)
        ax.scatter(base_pose[:, 0], base_pose[:, 1], base_pose[:, 2],
                   c="gray", s=8, alpha=0.6)

        # Animated skeleton (black)
        pose0 = pose_transform(all_positions[i, 0])
        segs_init = [(pose0[a], pose0[b]) for a, b in edges]
        lc_black = Line3DCollection(segs_init, colors="black", linewidths=1.5)
        ax.add_collection3d(lc_black)
        dots_black = ax.scatter(pose0[:, 0], pose0[:, 1], pose0[:, 2],
                                c="black", s=10)
        artists.append((lc_black, dots_black))

    def update(frame: int):
        updated = []
        for i, ax in enumerate(axes):
            pose = pose_transform(all_positions[i, frame])

            segs = [(pose[a], pose[b]) for a, b in edges]
            artists[i][0].set_segments(segs)
            artists[i][1]._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])

            updated.append(artists[i][0])
            updated.append(artists[i][1])
        return updated

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    ani.save(save_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"[DONE] PCA 3D grid animation saved to {save_path}")
