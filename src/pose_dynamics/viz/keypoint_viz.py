"""
Keypoint visualization utilities for pose data analysis.
Includes interactive and static plotting of 3D keypoints with indices.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def plot_keypoints_with_indices_static(
    pose_array: np.ndarray,
    title: str = "Keypoints with Indices",
    figsize: Tuple[int, int] = (14, 6),
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot 3D keypoints with their indices labeled (static matplotlib version).

    Parameters:
        pose_array: (n_points, 3) array or (n_points*3,) flattened array
        title: Plot title
        figsize: Figure size (width, height)
        show: Whether to call plt.show()

    Returns:
        matplotlib Figure object
    """
    # Handle flattened input
    if pose_array.ndim == 1:
        n_points = len(pose_array) // 3
        pose_array = pose_array.reshape(n_points, 3)

    fig = plt.figure(figsize=figsize)

    # Front view (x, y)
    ax1 = fig.add_subplot(121)
    ax1.scatter(pose_array[:, 0], pose_array[:, 1], s=50, c='blue', alpha=0.6)
    for i in range(len(pose_array)):
        ax1.annotate(
            str(i),
            (pose_array[i, 0], pose_array[i, 1]),
            fontsize=8,
            ha='right',
            color='red'
        )
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Front View (X-Y)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Side view (z, y)
    ax2 = fig.add_subplot(122)
    ax2.scatter(pose_array[:, 2], pose_array[:, 1], s=50, c='green', alpha=0.6)
    for i in range(len(pose_array)):
        ax2.annotate(
            str(i),
            (pose_array[i, 2], pose_array[i, 1]),
            fontsize=8,
            ha='right',
            color='red'
        )
    ax2.set_xlabel('Z')
    ax2.set_ylabel('Y')
    ax2.set_title('Side View (Z-Y)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_keypoints_interactive(
    pose_array: np.ndarray,
    title: str = "Interactive Keypoint Visualization",
    point_size: int = 8,
    show: bool = True
):
    """
    Create interactive 3D plot of keypoints with hover labels showing indices.
    Uses plotly for interactivity.

    Parameters:
        pose_array: (n_points, 3) array or (n_points*3,) flattened array
        title: Plot title
        point_size: Size of scatter points
        show: Whether to display the plot

    Returns:
        plotly Figure object
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "plotly is required for interactive plots. "
            "Install with: pip install plotly"
        )

    # Handle flattened input
    if pose_array.ndim == 1:
        n_points = len(pose_array) // 3
        pose_array = pose_array.reshape(n_points, 3)

    # Create subplots: 3D view + 2D front view + 2D side view
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}, {'type': 'scatter'}]],
        subplot_titles=('3D View', 'Front View (X-Y)', 'Side View (Z-Y)'),
        horizontal_spacing=0.1
    )

    # Prepare hover text with indices and coordinates
    hover_text = [
        f"Index: {i}<br>X: {pose_array[i,0]:.3f}<br>Y: {pose_array[i,1]:.3f}<br>Z: {pose_array[i,2]:.3f}"
        for i in range(len(pose_array))
    ]

    # 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=pose_array[:, 0],
            y=pose_array[:, 1],
            z=pose_array[:, 2],
            mode='markers+text',
            marker=dict(size=point_size, color='blue', opacity=0.7),
            text=[str(i) for i in range(len(pose_array))],
            textposition='top center',
            textfont=dict(size=8, color='red'),
            hovertext=hover_text,
            hoverinfo='text',
            name='Keypoints'
        ),
        row=1, col=1
    )

    # Front view (X-Y)
    fig.add_trace(
        go.Scatter(
            x=pose_array[:, 0],
            y=pose_array[:, 1],
            mode='markers+text',
            marker=dict(size=point_size, color='blue', opacity=0.7),
            text=[str(i) for i in range(len(pose_array))],
            textposition='top right',
            textfont=dict(size=8, color='red'),
            hovertext=hover_text,
            hoverinfo='text',
            name='Front'
        ),
        row=1, col=2
    )

    # Side view (Z-Y)
    fig.add_trace(
        go.Scatter(
            x=pose_array[:, 2],
            y=pose_array[:, 1],
            mode='markers+text',
            marker=dict(size=point_size, color='green', opacity=0.7),
            text=[str(i) for i in range(len(pose_array))],
            textposition='top right',
            textfont=dict(size=8, color='red'),
            hovertext=hover_text,
            hoverinfo='text',
            name='Side'
        ),
        row=1, col=3
    )

    # Update layout
    fig.update_layout(
        title=title,
        height=500,
        showlegend=False,
        hovermode='closest'
    )

    # Update 3D scene
    fig.update_scenes(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    )

    # Update 2D axes
    fig.update_xaxes(title_text='X', scaleanchor='y', scaleratio=1, row=1, col=2)
    fig.update_yaxes(title_text='Y', row=1, col=2)
    fig.update_xaxes(title_text='Z', scaleanchor='y', scaleratio=1, row=1, col=3)
    fig.update_yaxes(title_text='Y', row=1, col=3)

    if show:
        fig.show()

    return fig

