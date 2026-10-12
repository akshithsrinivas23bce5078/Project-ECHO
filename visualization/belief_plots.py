"""
Project ECHO — Belief State Plots
====================================
Visualization of POMDP belief states, trust/cognitive-load dynamics,
and Bayesian inference internals.

Plots:
    1. Trust vs. Cognitive Load scatter plot
    2. Belief state evolution (time series)
    3. POMDP transition matrix heatmap
    4. Belief distribution heatmap (15 states)
    5. Emotion distribution bar chart
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

from visualization.plot_config import (
    FIGSIZE_DEFAULT, FIGSIZE_WIDE, FIGSIZE_SQUARE, FIGSIZE_SMALL,
    EMOTION_COLORS, ECHO_PALETTE, save_figure, setup_style,
)

setup_style()


def plot_trust_vs_cognitive_load(
    df: pd.DataFrame,
    title: str = "Trust vs. Cognitive Load",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of trust_mean vs. cognitive_load colored by emotion.

    Args:
        df: DataFrame with columns trust_mean, cognitive_load, emotion_label.
        title: Plot title.
        save_name: If provided, save to results/plots/.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    for emotion, color in EMOTION_COLORS.items():
        mask = df["emotion_label"] == emotion
        if mask.any():
            ax.scatter(
                df.loc[mask, "trust_mean"],
                df.loc[mask, "cognitive_load"],
                c=color, label=emotion.capitalize(),
                alpha=0.6, s=30, edgecolors='white', linewidth=0.5,
            )

    ax.set_xlabel("Trust Mean")
    ax.set_ylabel("Cognitive Load")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="Emotion", loc="upper right")

    # Add quadrant annotations
    ax.axhline(0.5, color='grey', linestyle='--', alpha=0.3)
    ax.axvline(0.5, color='grey', linestyle='--', alpha=0.3)

    if save_name:
        save_figure(fig, save_name, subdir="belief")
    return fig


def plot_belief_evolution(
    df: pd.DataFrame,
    title: str = "Belief State Evolution Over Session",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-line time series of trust, cognitive load, and CLC.

    Args:
        df: DataFrame with columns time, trust_mean, cognitive_load, clc_raw.
    """
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE_WIDE, sharex=True)

    # Trust
    axes[0].plot(df["time"], df["trust_mean"], color=ECHO_PALETTE[0], linewidth=1.5)
    if "trust_variance" in df.columns:
        std = np.sqrt(df["trust_variance"])
        axes[0].fill_between(
            df["time"],
            df["trust_mean"] - std,
            df["trust_mean"] + std,
            alpha=0.2, color=ECHO_PALETTE[0],
        )
    axes[0].set_ylabel("Trust Mean")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(title)

    # Cognitive Load
    axes[1].plot(df["time"], df["cognitive_load"], color=ECHO_PALETTE[1], linewidth=1.5)
    axes[1].axhline(0.7, color='red', linestyle='--', alpha=0.4, label="High threshold")
    axes[1].axhline(0.3, color='green', linestyle='--', alpha=0.4, label="Low threshold")
    axes[1].set_ylabel("Cognitive Load")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=8)

    # CLC
    axes[2].plot(df["time"], df["clc_raw"], color=ECHO_PALETTE[2], linewidth=1.5)
    axes[2].axhspan(0.35, 0.55, alpha=0.1, color='green', label="Nominal range")
    axes[2].set_ylabel("CLC Raw")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(fontsize=8)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir="belief")
    return fig


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    state_labels: Optional[list] = None,
    title: str = "POMDP Transition Matrix T(s'|s)",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of the POMDP state transition matrix.

    Args:
        transition_matrix: (N, N) transition probability matrix.
        state_labels: Labels for each state.
    """
    n = transition_matrix.shape[0]
    if state_labels is None:
        from estimator.src.pomdp_model import state_from_index
        state_labels = [
            f"{t[:2]}T-{c[:2]}C" for t, c in
            [state_from_index(i) for i in range(n)]
        ]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    sns.heatmap(
        transition_matrix,
        ax=ax,
        xticklabels=state_labels,
        yticklabels=state_labels,
        cmap="YlOrRd",
        annot=False,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "P(s'|s)"},
    )

    ax.set_xlabel("Next State s'")
    ax.set_ylabel("Current State s")
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)

    if save_name:
        save_figure(fig, save_name, subdir="belief")
    return fig


def plot_belief_distribution(
    belief: np.ndarray,
    title: str = "Current Belief Distribution",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of the belief distribution across the 15-state space.
    Arranged as (5 trust levels) × (3 cognitive load levels).

    Args:
        belief: 1-D array of length 15.
    """
    from estimator.src.pomdp_model import NUM_TRUST, NUM_COGLOAD, TRUST_LEVELS, COGLOAD_LEVELS

    belief_grid = belief.reshape(NUM_TRUST, NUM_COGLOAD)

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)

    sns.heatmap(
        belief_grid,
        ax=ax,
        xticklabels=COGLOAD_LEVELS,
        yticklabels=TRUST_LEVELS,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        linewidths=1,
        cbar_kws={"label": "Belief Probability"},
    )

    ax.set_xlabel("Cognitive Load Level")
    ax.set_ylabel("Trust Level")
    ax.set_title(title)

    if save_name:
        save_figure(fig, save_name, subdir="belief")
    return fig


def plot_emotion_distribution(
    df: pd.DataFrame,
    title: str = "Emotion Label Distribution",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of emotion label frequencies across a session.

    Args:
        df: DataFrame with column emotion_label.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)

    col = "emotion_label" if "emotion_label" in df.columns else "state"
    raw_counts = df[col].value_counts()
    
    # Ensure all states are shown even if count is 0
    labels = list(EMOTION_COLORS.keys())
    counts = [raw_counts.get(label, 0) for label in labels]
    colors = [EMOTION_COLORS[label] for label in labels]

    ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_xlabel("Emotion Label")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')

    # Add percentage labels
    total = sum(counts)
    for i, count in enumerate(counts):
        pct = (count / total * 100) if total > 0 else 0.0
        # only show percentage if count > 0 to avoid clutter, or show 0.0% if preferred
        ax.text(i, count + (total * 0.01 if total > 0 else 0.1), f"{pct:.1f}%", ha='center', fontsize=9)

    if save_name:
        save_figure(fig, save_name, subdir="belief")
    return fig


def plot_belief_over_time(
    df: pd.DataFrame,
    title: str = "POMDP Belief Over Time",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Vis 3: Line plots of b_F, b_J, b_A, b_T, b_G, b_N against time.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    # Check if belief columns exist, they should be named like belief_focused, belief_angry etc.
    belief_cols = [c for c in df.columns if c.startswith("belief_")]
    if not belief_cols:
        ax.text(0.5, 0.5, "No belief columns found", ha='center', va='center')
        return fig
        
    for col in belief_cols:
        state = col.replace("belief_", "")
        color = EMOTION_COLORS.get(state, "#333333")
        ax.plot(df["time"], df[col], label=f"P({state.capitalize()})", color=color, linewidth=2.0)
        
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Belief Probability", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_name: save_figure(fig, save_name, subdir="transition")
    return fig
