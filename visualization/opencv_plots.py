"""
Project ECHO — OpenCV Vision Plots
=====================================
Visualization of the vision pipeline outputs: gaze heatmaps,
optical flow, visual complexity, and scene change detection.

Plots:
    1. Gaze heatmap overlay on game frame
    2. Optical flow quiver plot
    3. Visual complexity time series
    4. Scene change detection plot
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from visualization.plot_config import (
    FIGSIZE_DEFAULT, FIGSIZE_WIDE, FIGSIZE_SMALL,
    ECHO_PALETTE, save_figure, setup_style,
)

setup_style()


def plot_gaze_heatmap(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    title: str = "Attention Heatmap Overlay",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Display a gaze/attention heatmap overlaid on a game frame.

    Args:
        frame: BGR game frame.
        heatmap: 2-D normalized heatmap [0, 1].
        alpha: Overlay transparency.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    axes[0].imshow(frame_rgb)
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(heatmap, cmap="hot", interpolation="bilinear")
    axes[1].set_title("Attention Heatmap")
    axes[1].axis("off")

    # Overlay
    h, w = frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir="vision")
    return fig


def plot_optical_flow(
    flow: np.ndarray,
    frame: Optional[np.ndarray] = None,
    step: int = 16,
    title: str = "Optical Flow Field",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Quiver plot of optical flow vectors.

    Args:
        flow: (H, W, 2) optical flow field from Farneback.
        frame: Optional BGR frame for background.
        step: Sampling step for quiver grid.
    """
    h, w = flow.shape[:2]
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb, alpha=0.5)

    y, x = np.mgrid[step//2:h:step, step//2:w:step]
    fx = flow[step//2:h:step, step//2:w:step, 0]
    fy = flow[step//2:h:step, step//2:w:step, 1]
    magnitude = np.sqrt(fx**2 + fy**2)

    quiver = ax.quiver(x, y, fx, -fy, magnitude, cmap="viridis",
                       scale=50, width=0.003, alpha=0.8)
    plt.colorbar(quiver, ax=ax, label="Flow Magnitude")

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(title)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    if save_name:
        save_figure(fig, save_name, subdir="vision")
    return fig


def plot_visual_complexity_series(
    timestamps: np.ndarray,
    complexities: np.ndarray,
    edge_densities: Optional[np.ndarray] = None,
    color_entropies: Optional[np.ndarray] = None,
    title: str = "Visual Complexity Over Time",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Time series of visual complexity metrics.

    Args:
        timestamps: Time axis.
        complexities: Composite visual complexity [0, 1].
        edge_densities: Edge density component.
        color_entropies: Color entropy component.
    """
    n_plots = 1 + (edge_densities is not None) + (color_entropies is not None)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    idx = 0
    axes[idx].plot(timestamps, complexities, color=ECHO_PALETTE[0], linewidth=1.5)
    axes[idx].set_ylabel("Visual Complexity")
    axes[idx].set_ylim(-0.05, 1.05)
    axes[idx].set_title(title)
    idx += 1

    if edge_densities is not None:
        axes[idx].plot(timestamps, edge_densities, color=ECHO_PALETTE[1], linewidth=1.5)
        axes[idx].set_ylabel("Edge Density")
        axes[idx].set_ylim(-0.05, 1.05)
        idx += 1

    if color_entropies is not None:
        axes[idx].plot(timestamps, color_entropies, color=ECHO_PALETTE[2], linewidth=1.5)
        axes[idx].set_ylabel("Color Entropy")
        axes[idx].set_ylim(-0.05, 1.05)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir="vision")
    return fig


def plot_scene_changes(
    timestamps: np.ndarray,
    change_scores: np.ndarray,
    threshold: float = 0.15,
    title: str = "Scene Change Detection",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Scene change score over time with threshold.

    Args:
        timestamps: Time axis.
        change_scores: 1 - histogram correlation values.
        threshold: Scene change threshold.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    ax.plot(timestamps, change_scores, color=ECHO_PALETTE[0], linewidth=1.2, alpha=0.8)
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.6,
               label=f"Threshold ({threshold})")

    # Highlight detected changes
    changes = change_scores > threshold
    if np.any(changes):
        ax.scatter(
            timestamps[changes], change_scores[changes],
            c=ECHO_PALETTE[4], s=50, zorder=5, label="Scene Change",
            edgecolors='white', linewidth=0.5,
        )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Change Score (1 - correlation)")
    ax.set_title(title)
    ax.legend()

    if save_name:
        save_figure(fig, save_name, subdir="vision")
    return fig
