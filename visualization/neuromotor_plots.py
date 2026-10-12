"""
Project ECHO — Neuromotor & Keystroke Plots
=============================================
Visualization of Sigma-Lognormal model output and
inter-keystroke interval distributions.

Plots:
    1. Velocity profile: original vs. reconstructed
    2. SNR bar chart across trajectories
    3. Stroke parameter scatter (mu vs. sigma)
    4. IKI distribution (histogram + KDE)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

from visualization.plot_config import (
    FIGSIZE_DEFAULT, FIGSIZE_SMALL, ECHO_PALETTE,
    save_figure, setup_style,
)

setup_style()


def plot_velocity_profile(
    times: np.ndarray,
    original: np.ndarray,
    reconstructed: np.ndarray,
    snr_db: float,
    title: str = "Sigma-Lognormal Velocity Reconstruction",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Original vs. reconstructed velocity profile.

    Args:
        times: Time array.
        original: Original velocity profile.
        reconstructed: Reconstructed velocity from lognormal strokes.
        snr_db: Reconstruction SNR in dB.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    ax.plot(times, original, color=ECHO_PALETTE[0], linewidth=1.5,
            label="Original", alpha=0.8)
    ax.plot(times, reconstructed, color=ECHO_PALETTE[1], linewidth=1.5,
            linestyle='--', label="Reconstructed", alpha=0.8)

    # Residual shading
    ax.fill_between(
        times, original, reconstructed,
        alpha=0.15, color=ECHO_PALETTE[3], label="Residual",
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Velocity (px/s)")
    ax.set_title(f"{title}\nSNR = {snr_db:.1f} dB")
    ax.legend()

    if save_name:
        save_figure(fig, save_name, subdir="neuromotor")
    return fig


def plot_snr_bar_chart(
    snr_values: List[float],
    labels: Optional[List[str]] = None,
    target_snr: float = 24.0,
    title: str = "Neuromotor Reconstruction SNR",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of SNR values across multiple trajectories.

    Args:
        snr_values: List of SNR values in dB.
        labels: Labels for each trajectory.
        target_snr: Target SNR threshold (drawn as horizontal line).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    n = len(snr_values)
    if labels is None:
        labels = [f"Traj {i+1}" for i in range(n)]

    colors = [ECHO_PALETTE[2] if s >= target_snr else ECHO_PALETTE[4] for s in snr_values]

    bars = ax.bar(labels, snr_values, color=colors, edgecolor='white', linewidth=0.8)
    ax.axhline(target_snr, color='red', linestyle='--', alpha=0.6, label=f"Target ({target_snr} dB)")

    # Value labels on bars
    for bar, val in zip(bars, snr_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("SNR (dB)")
    ax.set_title(title)
    ax.legend()

    if save_name:
        save_figure(fig, save_name, subdir="neuromotor")
    return fig


def plot_stroke_parameters(
    strokes: List[dict],
    title: str = "Lognormal Stroke Parameters",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of stroke parameters (mu vs. sigma, sized by amplitude).

    Args:
        strokes: List of stroke dicts with keys amplitude, t0, mu, sigma.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)

    mus = [s["mu"] for s in strokes]
    sigmas = [s["sigma"] for s in strokes]
    amplitudes = [s["amplitude"] for s in strokes]
    t0s = [s["t0"] for s in strokes]

    # Size proportional to amplitude, color by onset time
    scatter = ax.scatter(
        mus, sigmas,
        s=[a * 2 for a in amplitudes],
        c=t0s,
        cmap="viridis",
        alpha=0.7,
        edgecolors='white',
        linewidth=0.8,
    )

    plt.colorbar(scatter, ax=ax, label="Onset Time t₀ (s)")
    ax.set_xlabel("Log-time Mean (μ)")
    ax.set_ylabel("Log-time Std (σ)")
    ax.set_title(title)

    if save_name:
        save_figure(fig, save_name, subdir="neuromotor")
    return fig


def plot_iki_distribution(
    iki_values: np.ndarray,
    title: str = "Inter-Keystroke Interval Distribution",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram + KDE of inter-keystroke interval values.

    Args:
        iki_values: 1-D array of IKI values in milliseconds.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    sns.histplot(
        iki_values, bins=40, kde=True, ax=ax,
        color=ECHO_PALETTE[0], alpha=0.6,
        edgecolor='white', linewidth=0.5,
    )

    # Add summary stats
    mean_iki = np.mean(iki_values)
    median_iki = np.median(iki_values)
    ax.axvline(mean_iki, color=ECHO_PALETTE[1], linestyle='--',
               label=f"Mean: {mean_iki:.1f} ms")
    ax.axvline(median_iki, color=ECHO_PALETTE[2], linestyle=':',
               label=f"Median: {median_iki:.1f} ms")

    ax.set_xlabel("Inter-Keystroke Interval (ms)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    if save_name:
        save_figure(fig, save_name, subdir="neuromotor")
    return fig
