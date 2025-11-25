"""
Keypoint processing utilities for pose-based behavioral analysis.

This module provides functions for:
- Keypoint mapping and selection
- Data quality control
- Temporal filtering
- Visualization of preprocessing effects
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path


def extract_keypoint_subset(aligned_array, keypoint_indices):
    """
    Extract subset of keypoints from aligned array.
    
    Parameters:
        aligned_array: (T, n_points*3) array of aligned keypoints
        keypoint_indices: list of keypoint indices to keep
        
    Returns:
        extracted_array: (T, n_selected*3) array with selected keypoints only
        
    Example:
        # Keep only nose, shoulders, and wrists
        subset = extract_keypoint_subset(data, [5, 12, 13, 16, 17])
    """
    n_frames = aligned_array.shape[0]
    n_points_total = aligned_array.shape[1] // 3
    
    # Reshape to (T, n_points, 3)
    coords_3d = aligned_array.reshape(n_frames, n_points_total, 3)
    
    # Extract selected keypoints
    coords_selected = coords_3d[:, keypoint_indices, :]
    
    # Flatten back to (T, n_selected*3)
    extracted_array = coords_selected.reshape(n_frames, -1)
    
    return extracted_array


def check_data_quality(aligned_array, trial_name, fps=30, verbose=True):
    """
    Run comprehensive quality control checks on aligned pose data.
    
    Parameters:
        aligned_array: (T, n_points*3) flattened keypoint array
        trial_name: str identifier for reporting
        fps: sampling rate in Hz
        verbose: whether to print detailed report
        
    Returns:
        quality_report: dict with quality metrics and pass/fail flags
        
    Quality checks performed:
        - NaN values
        - Extreme values (>5m from origin after centering)
        - Frozen keypoints (tracking failure)
        - Sudden jumps (tracking glitches)
        - Duplicate frames
        - Movement irregularity
    
    Example:
        report = check_data_quality(data, "P001_T1_P1", fps=30)
        if not report['passed']:
            print(f"Trial failed: {report['issues']}")
    """
    n_frames, n_dims = aligned_array.shape
    n_points = n_dims // 3
    
    # Reshape for analysis
    coords_3d = aligned_array.reshape(n_frames, n_points, 3)
    
    quality_report = {
        'trial_name': trial_name,
        'n_frames': n_frames,
        'duration_sec': n_frames / fps,
        'issues': [],
        'warnings': [],
        'passed': True
    }
    
    # 1. Check for NaN values
    n_nans = np.isnan(aligned_array).sum()
    if n_nans > 0:
        pct_nan = 100 * n_nans / aligned_array.size
        quality_report['issues'].append(f"{n_nans} NaN values ({pct_nan:.2f}%)")
        quality_report['passed'] = False
    
    # 2. Check for extreme values (likely tracking errors)
    extreme_threshold = 5.0  # meters from origin after centering
    n_extreme = np.sum(np.abs(coords_3d) > extreme_threshold)
    if n_extreme > 0:
        pct_extreme = 100 * n_extreme / coords_3d.size
        if pct_extreme > 1.0:  # More than 1% extreme values is concerning
            quality_report['issues'].append(
                f"{n_extreme} extreme values >±{extreme_threshold}m ({pct_extreme:.2f}%)"
            )
            quality_report['passed'] = False
        else:
            quality_report['warnings'].append(
                f"{n_extreme} extreme values detected ({pct_extreme:.3f}%)"
            )
    
    # 3. Check for constant/frozen keypoints (tracking failure)
    for pt_idx in range(n_points):
        pt_coords = coords_3d[:, pt_idx, :]
        pt_std = np.std(pt_coords, axis=0)
        
        if np.any(pt_std < 0.001):  # <1mm variation
            quality_report['warnings'].append(
                f"Keypoint {pt_idx} nearly constant (std={pt_std.max():.6f})"
            )
    
    # 4. Check for sudden jumps (tracking glitches)
    velocity = np.diff(coords_3d, axis=0)  # (T-1, n_points, 3)
    speed = np.linalg.norm(velocity, axis=2)  # (T-1, n_points)
    
    for pt_idx in range(n_points):
        pt_speed = speed[:, pt_idx]
        median_speed = np.median(pt_speed)
        max_speed = np.max(pt_speed)
        
        jump_threshold = 5 * median_speed
        n_jumps = np.sum(pt_speed > jump_threshold)
        
        if n_jumps > 0:
            quality_report['warnings'].append(
                f"Keypoint {pt_idx}: {n_jumps} sudden jumps "
                f"(max={max_speed:.3f} m/frame, threshold={jump_threshold:.3f})"
            )
    
    # 5. Check for duplicate consecutive frames
    frame_diffs = np.diff(coords_3d, axis=0)
    duplicate_frames = np.all(frame_diffs == 0, axis=(1, 2))
    n_duplicates = np.sum(duplicate_frames)
    
    if n_duplicates > n_frames * 0.05:  # >5% duplicates is concerning
        pct_dup = 100 * n_duplicates / n_frames
        quality_report['warnings'].append(
            f"{n_duplicates} duplicate frames ({pct_dup:.1f}%)"
        )
    
    # 6. Check sampling regularity
    total_displacement = np.sum(speed, axis=1)  # Total body movement per frame
    cv = np.std(total_displacement) / (np.mean(total_displacement) + 1e-8)
    
    if cv > 2.0:  # High coefficient of variation
        quality_report['warnings'].append(
            f"High movement irregularity (CV={cv:.2f})"
        )
    
    # Print report if verbose
    if verbose:
        status = "✓ PASSED" if quality_report['passed'] else "✗ FAILED"
        print(f"\n{status}: {trial_name}")
        print(f"  Duration: {quality_report['duration_sec']:.1f}s "
              f"({quality_report['n_frames']} frames)")
        
        if quality_report['issues']:
            print("  Issues:")
            for issue in quality_report['issues']:
                print(f"    ✗ {issue}")
        
        if quality_report['warnings']:
            print("  Warnings:")
            for warning in quality_report['warnings']:
                print(f"    ⚠ {warning}")
        
        if not quality_report['issues'] and not quality_report['warnings']:
            print("  No issues detected")
    
    return quality_report


def apply_butterworth_filter(aligned_array, fps=30, cutoff_hz=10, order=4):
    """
    Apply zero-phase Butterworth low-pass filter to remove high-frequency noise.
    
    Parameters:
        aligned_array: (T, n_points*3) flattened keypoint array
        fps: sampling rate in Hz
        cutoff_hz: cutoff frequency in Hz (default 10 Hz for human movement)
        order: filter order (default 4, higher = sharper cutoff)
        
    Returns:
        filtered_array: (T, n_points*3) filtered keypoints
        filter_info: dict with filter parameters
        
    Notes:
        - Uses filtfilt for zero-phase filtering (no temporal distortion)
        - Skips dimensions with NaN values
        - Common cutoff frequencies:
            * 6 Hz: slow movements (e.g., quiet standing)
            * 10 Hz: general human movement (default)
            * 15 Hz: faster movements (e.g., sports)
    
    Example:
        filtered, info = apply_butterworth_filter(data, fps=30, cutoff_hz=10)
        print(f"Filter: {info['order']}th order, {info['cutoff_hz']} Hz cutoff")
    """
    nyquist = fps / 2.0
    normal_cutoff = cutoff_hz / nyquist
    
    # Design Butterworth filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply zero-phase filtering
    filtered = np.zeros_like(aligned_array)
    
    for dim in range(aligned_array.shape[1]):
        if np.any(np.isnan(aligned_array[:, dim])):
            # Skip filtering for dimensions with NaNs
            filtered[:, dim] = aligned_array[:, dim]
        else:
            filtered[:, dim] = signal.filtfilt(b, a, aligned_array[:, dim])
    
    filter_info = {
        'fps': fps,
        'cutoff_hz': cutoff_hz,
        'order': order,
        'nyquist_hz': nyquist,
        'normalized_cutoff': normal_cutoff
    }
    
    return filtered, filter_info


def plot_filtering_comparison(raw_array, filtered_array, fps, 
                              keypoint_labels, keypoint_indices,
                              save_path=None):
    """
    Create comparison plot showing raw vs filtered trajectories.
    
    Parameters:
        raw_array: (T, n_points*3) raw aligned keypoints
        filtered_array: (T, n_points*3) filtered keypoints
        fps: sampling rate in Hz
        keypoint_labels: list of str labels for keypoints to plot
        keypoint_indices: list of int indices for keypoints to plot
        save_path: optional path to save figure
        
    Returns:
        fig: matplotlib figure object
        
    Example:
        fig = plot_filtering_comparison(
            raw, filtered, fps=30,
            keypoint_labels=['head', 'l_hand', 'l_heel'],
            keypoint_indices=[5, 9, 19],
            save_path='filtering_comparison.png'
        )
    """
    n_frames = raw_array.shape[0]
    n_points = raw_array.shape[1] // 3
    time = np.arange(n_frames) / fps
    
    # Reshape
    raw_3d = raw_array.reshape(n_frames, n_points, 3)
    filt_3d = filtered_array.reshape(n_frames, n_points, 3)
    
    n_keypoints = len(keypoint_indices)
    fig, axes = plt.subplots(n_keypoints, 3, figsize=(15, 3*n_keypoints))
    
    if n_keypoints == 1:
        axes = axes.reshape(1, -1)
    
    coord_names = ['X', 'Y', 'Z']
    
    for i, (label, pt_idx) in enumerate(zip(keypoint_labels, keypoint_indices)):
        for coord in range(3):
            ax = axes[i, coord]
            
            # Plot raw (red, transparent)
            ax.plot(time, raw_3d[:, pt_idx, coord], 
                   color='#e74c3c', alpha=0.5, linewidth=1, label='Raw')
            
            # Plot filtered (blue, bold)
            ax.plot(time, filt_3d[:, pt_idx, coord], 
                   color='#3498db', linewidth=2, label='Filtered')
            
            # Compute SNR
            noise = raw_3d[:, pt_idx, coord] - filt_3d[:, pt_idx, coord]
            rms_noise = np.sqrt(np.mean(noise**2))
            rms_signal = np.sqrt(np.mean(filt_3d[:, pt_idx, coord]**2))
            snr_db = 20 * np.log10(rms_signal / (rms_noise + 1e-10))
            
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel(f'{coord_names[coord]} position (m)', fontsize=10)
            ax.set_title(f'{label} - {coord_names[coord]} (SNR: {snr_db:.1f} dB)', 
                        fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved filtering comparison to {save_path}")
    
    return fig


def create_preprocessing_video(raw_array, filtered_array, fps,
                               keypoint_labels, keypoint_indices,
                               skeleton_edges, duration_sec=10,
                               save_path='preprocessing_demo.mp4'):
    """
    Create animated video showing skeleton and time series comparison.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    n_points = raw_array.shape[1] // 3
    n_frames = min(int(duration_sec * fps), raw_array.shape[0])

    # Reshape to (T, n_points, 3)
    raw_3d = raw_array[:n_frames].reshape(n_frames, n_points, 3)
    filt_3d = filtered_array[:n_frames].reshape(n_frames, n_points, 3)

    # ---- helper: camera -> plotting coordinates ----
    # input: (x, y, z) where y is vertical (camera convention)
    # output: (X, Y, Z) where Z is vertical for Matplotlib
    def to_world_coords(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        X = x           # left-right
        Y = -z          # depth (flip so they come to the front)
        Z = y          # up
        return np.stack([X, Y, Z], axis=-1)

    # Precompute limits in world coords for consistent axes
    all_pts_world = to_world_coords(filt_3d.reshape(-1, 3))
    x_min, x_max = all_pts_world[:, 0].min(), all_pts_world[:, 0].max()
    y_min, y_max = all_pts_world[:, 1].min(), all_pts_world[:, 1].max()
    z_min, z_max = all_pts_world[:, 2].min(), all_pts_world[:, 2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.6

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    # ---- set up figure ----
    fig = plt.figure(figsize=(16, 8))

    # Left: 3D skeleton
    ax_skel = fig.add_subplot(121, projection='3d')

    # Right: Time series
    n_ts = len(keypoint_indices)
    gs = fig.add_gridspec(n_ts, 2, left=0.55, right=0.95, hspace=0.3)
    ax_ts = [fig.add_subplot(gs[i, :]) for i in range(n_ts)]

    time = np.arange(n_frames) / fps
    lines_raw, lines_filt, lines_current = [], [], []

    for i, (label, pt_idx) in enumerate(zip(keypoint_labels, keypoint_indices)):
        ax = ax_ts[i]
        coord = 1  # use original Y (vertical in the raw data) for time series

        line_raw, = ax.plot(time, raw_3d[:, pt_idx, coord],
                            'r-', alpha=0.3, linewidth=1, label='Raw')
        line_filt, = ax.plot(time, filt_3d[:, pt_idx, coord],
                             'b-', alpha=0.3, linewidth=1, label='Filtered')

        line_curr, = ax.plot([], [], 'ko', markersize=8, label='Current')

        lines_raw.append(line_raw)
        lines_filt.append(line_filt)
        lines_current.append(line_curr)

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Vertical position (m)', fontsize=10)
        ax.set_title(f'{label} - Vertical Position', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, duration_sec)

    # ---- animation update ----
    def update(frame):
        ax_skel.clear()

        # Transform this frame to world coords
        pts_world = to_world_coords(filt_3d[frame])  # (n_points, 3)

        # Plot keypoints
        ax_skel.scatter(pts_world[:, 0], pts_world[:, 1], pts_world[:, 2],
                        c='blue', s=30, alpha=0.6)

        # Plot skeleton edges
        for (i, j) in skeleton_edges:
            if i < n_points and j < n_points:
                ax_skel.plot(
                    [pts_world[i, 0], pts_world[j, 0]],
                    [pts_world[i, 1], pts_world[j, 1]],
                    [pts_world[i, 2], pts_world[j, 2]],
                    'b-', linewidth=2, alpha=0.5
                )

        # Rotate around vertical axis (Z is up now)
        ax_skel.view_init(elev=15, azim=frame * 360 / n_frames)

        ax_skel.set_xlabel('X (m)', fontsize=10)      # left-right
        ax_skel.set_ylabel('Y (m)', fontsize=10)      # depth
        ax_skel.set_zlabel('Z (m)', fontsize=10)      # up

        ax_skel.set_title(f'3D Skeleton (t={frame / fps:.2f}s)',
                          fontsize=14, fontweight='bold')

        # Consistent axis limits
        ax_skel.set_xlim(x_mid - max_range, x_mid + max_range)
        ax_skel.set_ylim(y_mid - max_range, y_mid + max_range)
        ax_skel.set_zlim(z_mid - max_range, z_mid + max_range)

        # Update time series
        for i, (pt_idx, line_curr) in enumerate(zip(keypoint_indices, lines_current)):
            coord = 1  # original vertical axis in data
            t_curr = frame / fps
            y_curr = filt_3d[frame, pt_idx, coord]
            line_curr.set_data([t_curr], [y_curr])

            lines_raw[i].set_data(time[:frame + 1], raw_3d[:frame + 1, pt_idx, coord])
            lines_filt[i].set_data(time[:frame + 1], filt_3d[:frame + 1, pt_idx, coord])
            lines_raw[i].set_alpha(0.5)
            lines_filt[i].set_alpha(1.0)

        return ax_skel, *lines_current, *lines_raw, *lines_filt

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / fps, blit=False)

    writer = FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(save_path, writer=writer)
    plt.close()

    print(f"✓ Saved preprocessing video to {save_path}")
    print(f"  Duration: {duration_sec}s, FPS: {fps}, Frames: {n_frames}")
