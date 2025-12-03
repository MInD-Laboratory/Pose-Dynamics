"""
MOSAIC Visualization Module

Plotting utilities for alignment diagnostics, PCA animations, and skeleton visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Tuple, Optional, Callable
import os
import math

# ============================================================================
# SKELETON UTILITIES
# ============================================================================

# Skeleton connections from config
SKELETON_CONNECTIONS = {
    "face": [    
        ("Nose", "LEye"),     
        ("Nose", "REye"),     
        ("Nose", "Neck"),     
    ],
    "body": [
        ("Neck", "RShoulder"),
        ("Neck", "LShoulder"),
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
        ("Neck", "MidHip"),
        ("MidHip", "RHip"),
        ("RHip", "RKnee"),
        ("RKnee", "RAnkle"),
        ("MidHip", "LHip"),
        ("LHip", "LKnee"),
        ("LKnee", "LAnkle"),
    ],
    "arm": [
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
    ]
}


def get_skeleton_pairs(
    keypoint_names: List[str], 
    sets: Tuple[str, ...] = ("body", "arm", "face")
) -> List[Tuple[int, int]]:
    """
    Get pairs of keypoint indices for skeleton connections.
    
    Parameters
    ----------
    keypoint_names : list of str
        Column names for keypoints (e.g., ['Nose_x_offset', 'Nose_y_offset', ...])
    sets : tuple of str
        Which skeleton connection sets to include (default: all)
        
    Returns
    -------
    list of tuples
        List of (point_index1, point_index2) for each connection
    """
    pairs = []
    
    for set_name in sets:
        if set_name not in SKELETON_CONNECTIONS:
            continue
            
        for kp1, kp2 in SKELETON_CONNECTIONS[set_name]:
            try:
                # Find indices for both keypoints
                idx1 = next(i for i, col in enumerate(keypoint_names[::2]) 
                          if kp1 in col)
                idx2 = next(i for i, col in enumerate(keypoint_names[::2]) 
                          if kp2 in col)
                pairs.append((idx1, idx2))
            except StopIteration:
                # One or both keypoints not found
                continue
                
    return pairs


# ============================================================================
# ALIGNMENT DIAGNOSTICS
# ============================================================================

def plot_alignment_diagnostics(
    global_template: np.ndarray,
    raw_windows: List[Tuple],
    expected_cols: List[str],
    align_keypoints: Callable,
    n_samples: int = 2,
    procrustes: bool = True,
    ref_lengths: Optional[Dict] = None,
    allow_rotation: bool = True,
    reference: str = "Torso"
) -> None:
    """
    Overlay the global template with sample trial mean skeletons.
    
    Shows three stages: before alignment, after alignment, and after limb rescaling.
    
    Parameters
    ----------
    global_template : np.ndarray
        Global template pose (n_points, 2)
    raw_windows : list of tuples
        List of (window_df, metadata) tuples
    expected_cols : list of str
        Column names for keypoints
    align_keypoints : callable
        Alignment function to use
    n_samples : int
        Number of sample trials to show (default: 2)
    procrustes : bool
        Use Procrustes alignment (default: True)
    ref_lengths : dict or None
        Reference limb lengths for rescaling
    allow_rotation : bool
        Allow rotation in Procrustes (default: True)
    reference : str
        Reference point for centering (default: "Torso")
    """
    from .alignment import batch_apply_fixed_lengths, rebuild_aligned_dataframe
    
    # Setup
    n_points = len(expected_cols) // 2
    sample_indices = np.linspace(0, len(raw_windows)-1, 
                                min(n_samples, len(raw_windows))).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].set_title("Before Alignment")
    axes[1].set_title("After Alignment")
    axes[2].set_title("After Limb Rescaling")

    for ax in axes:
        ax.set_aspect('equal')
        ax.axis("off")

    skeleton_pairs = get_skeleton_pairs(expected_cols, sets=("body", "arm", "face"))
    colors = plt.cm.tab10.colors

    for i, idx in enumerate(sample_indices):
        window_df, _ = raw_windows[idx]
        color = colors[i % len(colors)]

        # Raw
        trial_mean = window_df.mean().values.reshape(n_points, 2)
        axes[0].scatter(trial_mean[:, 0], trial_mean[:, 1], alpha=0.6, s=10,
                       label=f"Participant {i+1}", color=color)
        for i1, i2 in skeleton_pairs:
            axes[0].plot([trial_mean[i1, 0], trial_mean[i2, 0]],
                        [trial_mean[i1, 1], trial_mean[i2, 1]],
                        alpha=0.4, color=color, linewidth=1)

        # Aligned
        aligned_X, _ = align_keypoints(
            window_df, expected_cols,
            reference=reference,
            template=global_template,
            use_procrustes=procrustes,
            allow_rotation=allow_rotation,
            allow_scale=True  # Match main pipeline setting (old code default)
        )
        aligned_mean = aligned_X.mean(axis=0).reshape(n_points, 2)
        axes[1].scatter(aligned_mean[:, 0], aligned_mean[:, 1], 
                       alpha=0.6, s=10, color=color)
        for i1, i2 in skeleton_pairs:
            axes[1].plot([aligned_mean[i1, 0], aligned_mean[i2, 0]],
                        [aligned_mean[i1, 1], aligned_mean[i2, 1]],
                        alpha=0.4, color=color, linewidth=1)

        # Limb-rescaled
        if ref_lengths:
            poses = aligned_X.reshape(-1, n_points, 2)
            poses_rescaled = batch_apply_fixed_lengths(poses, ref_lengths)
            rescaled_mean = poses_rescaled.mean(axis=0)
            
            axes[2].scatter(rescaled_mean[:, 0], rescaled_mean[:, 1],
                           alpha=0.6, s=10, color=color)
            for i1, i2 in skeleton_pairs:
                axes[2].plot([rescaled_mean[i1, 0], rescaled_mean[i2, 0]],
                            [rescaled_mean[i1, 1], rescaled_mean[i2, 1]],
                            alpha=0.4, color=color, linewidth=1)

    # Global template on all three plots
    for ax in axes:
        # Skeleton lines
        for i1, i2 in skeleton_pairs:
            ax.plot([global_template[i1, 0], global_template[i2, 0]],
                   [global_template[i1, 1], global_template[i2, 1]],
                   c="black", linewidth=2, alpha=0.8, zorder=10)

        # Keypoints
        ax.scatter(global_template[:, 0], global_template[:, 1],
                  c="black", label="Global Template" if ax == axes[0] else None,
                  s=12, zorder=11)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    gt_index = labels.index("Global Template")
    handles = [handles[gt_index]] + handles[:gt_index] + handles[gt_index+1:]
    labels = ["Global Template"] + labels[:gt_index] + labels[gt_index+1:]

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels),
        fontsize=10,
        markerscale=1.5,
        frameon=False
    )

    plt.show()


# ============================================================================
# PCA ANIMATIONS
# ============================================================================

def create_pc_animation(
    pca,
    X: pd.DataFrame,
    keypoint_names: List[str],
    save_path: str,
    mean_vector: np.ndarray,
    n_components: int = 3,
    n_frames: int = 250,
    scale: float = 2.0,
    mode: str = "sine",
    fps: int = 40,
    ref_lengths: Optional[Dict] = None,
    **kwargs
) -> None:
    """
    Create PCA movement animations showing principal component motions.
    
    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA model
    X : pd.DataFrame
        Centered pose data
    keypoint_names : list of str
        Column names for keypoints
    save_path : str
        Path to save animation file
    mean_vector : np.ndarray
        Mean pose vector
    n_components : int
        Number of PCs to animate (default: 3)
    n_frames : int
        Number of frames in animation (default: 250)
    scale : float
        Scaling factor for PC motion (default: 2.0)
    mode : str
        Animation mode: "sine" for synthetic oscillation (default)
    fps : int
        Frames per second (default: 40)
    ref_lengths : dict or None
        Reference limb lengths for anatomical constraints
    **kwargs : dict
        Additional parameters (width, height, zoom, etc.)
    """
    from .alignment import batch_apply_fixed_lengths
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    coords_per_point = 2
    n_points = len(keypoint_names) // coords_per_point

    # Build PC trajectories
    all_data = []
    pc_scores = pca.transform(X)
    
    for i in range(n_components):
        # Synthetic sine wave oscillation for this PC
        t = np.linspace(0, 2*np.pi, n_frames)
        pc_trajectory = scale * np.sin(t) * np.std(pc_scores[:, i])
        
        # Reconstruct poses
        pc_motion = np.zeros((n_frames, pca.n_components_))
        pc_motion[:, i] = pc_trajectory
        
        poses_recon = pca.inverse_transform(pc_motion)
        poses_recon += mean_vector
        
        all_data.append(poses_recon.reshape(n_frames, n_points, coords_per_point))

    all_data = np.array(all_data)
    base_pose = mean_vector.reshape(n_points, coords_per_point)

    # Compute axis limits
    all_x = np.concatenate([all_data[:, :, :, 0].ravel(), base_pose[:, 0]])
    all_y = np.concatenate([all_data[:, :, :, 1].ravel(), base_pose[:, 1]])

    zoom = kwargs.get('zoom_factor', 1.0)
    x_center = (all_x.max() + all_x.min()) / 2
    y_center = (all_y.max() + all_y.min()) / 2
    x_range = (all_x.max() - all_x.min()) * zoom
    y_range = (all_y.max() - all_y.min()) * zoom

    ax_min_x = x_center - x_range / 2
    ax_max_x = x_center + x_range / 2
    ax_min_y = y_center - y_range / 2
    ax_max_y = y_center + y_range / 2

    # Setup figure
    ncols = min(3, n_components)
    nrows = math.ceil(n_components / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    skeleton_pairs = get_skeleton_pairs(keypoint_names)
    animated_artists = []

    for i, ax in enumerate(axes[:n_components]):
        ax.set_xlim(ax_min_x, ax_max_x)
        ax.set_ylim(ax_max_y, ax_min_y)  # Flip y-axis for image coordinates
        ax.set_aspect('equal')
        ax.set_title(f'PC {i+1}')
        ax.axis('off')

        # Plot skeleton
        lines = []
        for i1, i2 in skeleton_pairs:
            line, = ax.plot([], [], 'b-', linewidth=2)
            lines.append(line)
        
        scatter = ax.scatter([], [], c='red', s=20)
        animated_artists.append((lines, scatter, i))

    for ax in axes[n_components:]:
        ax.axis('off')

    def update(frame):
        for lines, scatter, pc_idx in animated_artists:
            pose = all_data[pc_idx, frame]
            
            # Update skeleton lines
            for line, (i1, i2) in zip(lines, skeleton_pairs):
                line.set_data([pose[i1, 0], pose[i2, 0]],
                            [pose[i1, 1], pose[i2, 1]])
            
            # Update keypoints
            scatter.set_offsets(pose)
        
        return sum([list(lines) + [scatter] for lines, scatter, _ in animated_artists], [])

    ani = animation.FuncAnimation(fig, update, frames=n_frames, 
                                 interval=1000/fps, blit=True)
    ani.save(save_path, writer='ffmpeg', fps=fps)
    plt.close()
    print(f"[DONE] Animation saved to {save_path}")
