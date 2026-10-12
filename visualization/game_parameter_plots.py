"""
Project ECHO — Game Parameter Plots
=======================================
Visualization of DDA parameter adjustments across game genres.

Plots:
    1. Multi-panel parameter evolution (per genre)
    2. Parameter adjustment heatmap (emotion × cogload)
    3. Cross-genre grouped bar chart
    4. Radar/spider chart (per-genre sensitivity)
    5. Parameter delta vs. emotion confidence scatter
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional

from visualization.plot_config import (
    FIGSIZE_DEFAULT, FIGSIZE_WIDE, FIGSIZE_SQUARE, FIGSIZE_SMALL,
    GENRE_COLORS, EMOTION_COLORS, ECHO_PALETTE,
    save_figure, setup_style,
)

setup_style()


def plot_parameter_evolution(
    df: pd.DataFrame,
    genre: str,
    param_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
) -> List[plt.Figure]:
    """
    Line plots: individual parameter values over session time.

    Args:
        df: Session DataFrame with time and parameter columns.
        genre: Genre name for labeling.
        param_names: Specific parameters to plot (all if None).
    """
    if param_names is None:
        # Auto-detect parameter columns (exclude _delta columns)
        param_names = [
            c for c in df.columns
            if c not in ("time", "trust_mean", "trust_variance", "cognitive_load",
                         "clc_raw", "emotion_label", "emotion_confidence", "dda_action")
            and not c.endswith("_delta")
        ]

    figs = []
    colors = ECHO_PALETTE

    for i, param in enumerate(param_names):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["time"], df[param], color=colors[i % len(colors)], linewidth=1.5)
        ax.set_ylabel(param.replace("_", " ").title(), fontsize=10, fontweight='bold')
        ax.set_xlabel("Time (seconds)", fontsize=10, fontweight='bold')
        
        fig_title = title or f"DDA Parameter Evolution — {genre.upper()} — {param.replace('_', ' ').title()}"
        ax.set_title(fig_title, fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_name:
            save_figure(fig, f"{save_name}_{param}", subdir="game_params")
        
        figs.append(fig)

    return figs


def plot_parameter_heatmap(
    df: pd.DataFrame,
    param_name: str,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of mean parameter delta across (emotion × cognitive_load_bin).

    Args:
        df: Session DataFrame with emotion_label, cognitive_load, and param delta.
    """
    delta_col = f"{param_name}_delta"
    if delta_col not in df.columns:
        raise ValueError(f"Column {delta_col} not found in DataFrame")

    # Bin cognitive load into categories
    df_copy = df.copy()
    df_copy["cogload_bin"] = pd.cut(
        df_copy["cognitive_load"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    )

    pivot = df_copy.pivot_table(
        values=delta_col,
        index="emotion_label",
        columns="cogload_bin",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn_r",
        center=0,
        linewidths=1,
        cbar_kws={"label": f"Mean Δ {param_name}"},
    )

    ax.set_xlabel("Cognitive Load")
    ax.set_ylabel("Emotion")
    ax.set_title(title or f"Parameter Delta Heatmap: {param_name}")

    if save_name:
        save_figure(fig, save_name, subdir="game_params")
    return fig


def plot_cross_genre_comparison(
    all_sessions: Dict[str, pd.DataFrame],
    title: str = "Cross-Genre Mean Parameter Deltas",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart: average absolute parameter deltas across genres.

    Args:
        all_sessions: Dict mapping genre → session DataFrame.
    """
    genre_stats = {}

    for genre, df in all_sessions.items():
        delta_cols = [c for c in df.columns if c.endswith("_delta")]
        mean_abs_deltas = {
            c.replace("_delta", ""): df[c].abs().mean()
            for c in delta_cols
        }
        genre_stats[genre] = mean_abs_deltas

    # Build comparison DataFrame
    comparison = pd.DataFrame(genre_stats).T
    comparison.index.name = "Genre"

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    comparison.plot(kind="bar", ax=ax, width=0.8, edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Game Genre")
    ax.set_ylabel("Mean |Δ Parameter|")
    ax.set_title(title)
    ax.legend(
        title="Parameter", bbox_to_anchor=(1.05, 1), loc='upper left',
        fontsize=8, title_fontsize=9,
    )
    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir="game_params")
    return fig


def plot_genre_radar(
    df: pd.DataFrame,
    genre: str,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Radar/spider chart showing parameter sensitivity profile for a genre.

    Args:
        df: Session DataFrame.
        genre: Genre name.
    """
    delta_cols = [c for c in df.columns if c.endswith("_delta")]
    labels = [c.replace("_delta", "").replace("_", "\n") for c in delta_cols]
    values = [df[c].abs().mean() for c in delta_cols]

    n = len(labels)
    if n < 3:
        fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
        ax.text(0.5, 0.5, "Not enough parameters for radar chart",
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Normalize values for display
    max_val = max(values) if max(values) > 0 else 1.0
    values_norm = [v / max_val for v in values]

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_norm += [values_norm[0]]  # Close the polygon
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE, subplot_kw=dict(polar=True))

    color = GENRE_COLORS.get(genre, ECHO_PALETTE[0])
    ax.fill(angles, values_norm, alpha=0.25, color=color)
    ax.plot(angles, values_norm, color=color, linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title or f"Parameter Sensitivity — {genre.upper()}", pad=20)

    if save_name:
        save_figure(fig, save_name, subdir="game_params")
    return fig


def plot_delta_vs_confidence(
    df: pd.DataFrame,
    param_name: str,
    title: Optional[str] = None,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of parameter delta vs. emotion confidence.

    Args:
        df: Session DataFrame.
        param_name: Parameter to plot.
    """
    delta_col = f"{param_name}_delta"
    if delta_col not in df.columns:
        raise ValueError(f"Column {delta_col} not found")

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    for emotion, color in EMOTION_COLORS.items():
        mask = df["emotion_label"] == emotion
        if mask.any():
            ax.scatter(
                df.loc[mask, "emotion_confidence"],
                df.loc[mask, delta_col],
                c=color, label=emotion.capitalize(),
                alpha=0.5, s=25, edgecolors='white', linewidth=0.3,
            )

    ax.axhline(0, color='grey', linestyle='-', alpha=0.3)
    ax.set_xlabel("Emotion Confidence")
    ax.set_ylabel(f"Δ {param_name}")
    ax.set_title(title or f"Parameter Delta vs. Emotion Confidence: {param_name}")
    ax.legend(title="Emotion", fontsize=8)

    if save_name:
        save_figure(fig, save_name, subdir="game_params")
    return fig


def plot_dda_parameter_evolution_with_states(
    df: pd.DataFrame,
    param_names: List[str],
    title: Optional[str] = None,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Vis 4: Extended temporal graph showing multiple DDA parameter values 
    alongside the dominant state over adaptation periods.
    """
    # Filter to only params that exist in the dataframe
    valid_params = [p for p in param_names if p in df.columns]
    if not valid_params:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.text(0.5, 0.5, "No matching parameters found in session data", ha='center', va='center')
        return fig
        
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    # Plot parameter lines
    colors = plt.cm.tab20(np.linspace(0, 1, len(valid_params)))
    for i, p in enumerate(valid_params):
        ax.plot(df["time"], df[p], color=colors[i], linewidth=2.0, zorder=3, label=p.replace("_", " ").title())
    
    # Shade background by dominant emotion state
    if "dominant_state" in df.columns:
        state_col = "dominant_state"
    elif "emotion_label" in df.columns:
        state_col = "emotion_label"
    else:
        state_col = None
        
    if state_col:
        # Find contiguous segments of states
        states = df[state_col].values
        times = df["time"].values
        
        start_idx = 0
        for i in range(1, len(states)):
            if states[i] != states[i-1] or i == len(states) - 1:
                end_idx = i if i == len(states) - 1 else i
                state = states[i-1]
                color = EMOTION_COLORS.get(state, "#999999")
                
                ax.axvspan(times[start_idx], times[end_idx], alpha=0.3, color=color, zorder=1)
                start_idx = i
                
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Parameter Value", fontsize=12, fontweight='bold')
    ax.set_title(title or f"DDA Parameter Evolution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=2)
    
    # Create custom legend for states and parameters
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=color, alpha=0.3, label=state.capitalize())
        for state, color in EMOTION_COLORS.items()
    ]
    
    # Add parameter lines to legend
    from matplotlib.lines import Line2D
    for i, p in enumerate(valid_params):
        legend_patches.append(Line2D([0], [0], color=colors[i], lw=2, label=p.replace("_", " ").title()))
        
    # Place legend outside to avoid obscuring the graph with 20 lines
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.0, 0.5), title="States & Parameters", fontsize='small')

    
    plt.tight_layout()
    if save_name: save_figure(fig, save_name, subdir="transition")
    return fig
