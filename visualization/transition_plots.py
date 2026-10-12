"""
Project ECHO — Transition Visualization Plots
==================================================
Publication-quality plots for the 6x6 State Transition Matrix
and Cognitive DDA Control Model.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Any
import networkx as nx

from visualization.plot_config import (
    FIGSIZE_DEFAULT, FIGSIZE_WIDE, FIGSIZE_SQUARE, FIGSIZE_SMALL,
    EMOTION_COLORS, ECHO_PALETTE, save_figure, setup_style,
)

setup_style()

EMOTION_LABELS = ["angry", "focused", "thinking", "neutral", "joy", "given_up"]
TRANSITION_SUBDIR = "transition"


def plot_transition_heatmap(
    transition_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "6×6 State Transition Matrix",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Vis 1: 6x6 Transition Heatmap."""
    if labels is None: labels = EMOTION_LABELS
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    sns.heatmap(
        transition_matrix, ax=ax,
        xticklabels=[l.capitalize() for l in labels],
        yticklabels=[l.capitalize() for l in labels],
        annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=1, linecolor='white',
        cbar_kws={"label": "Transition Probability", "shrink": 0.8},
        vmin=0, vmax=1, square=True,
    )
    
    ax.set_xlabel("Next State (S_{t+1})", fontsize=12, fontweight='bold')
    ax.set_ylabel("Current State (S_t)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    if save_name: save_figure(fig, save_name, subdir=TRANSITION_SUBDIR)
    return fig


def plot_transition_graph(
    transition_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    threshold: float = 0.05,
    title: str = "State Transition Graph",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Vis 2: State Transition Graph using NetworkX."""
    if labels is None: labels = EMOTION_LABELS
    
    G = nx.DiGraph()
    for i, label in enumerate(labels):
        G.add_node(label.capitalize())
        
    for i in range(len(labels)):
        for j in range(len(labels)):
            weight = transition_matrix[i, j]
            if weight > threshold:
                G.add_edge(labels[i].capitalize(), labels[j].capitalize(), weight=weight)
                
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    pos = nx.circular_layout(G)
    
    # Node colors based on EMOTION_COLORS
    node_colors = [EMOTION_COLORS.get(node.lower(), "#999999") for node in G.nodes()]
    
    # Edge widths based on weights
    edges = G.edges()
    weights = [G[u][v]['weight'] * 5 for u, v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_weight="bold", font_color="white", ax=ax)
    
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, width=weights,
        arrowsize=20, edge_color="gray", connectionstyle='arc3,rad=0.1', ax=ax, alpha=0.7
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    if save_name: save_figure(fig, save_name, subdir=TRANSITION_SUBDIR)
    return fig


def plot_state_dda_heatmap(
    sensitivity_matrix: pd.DataFrame,
    title: str = "State × DDA Parameter Heatmap",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Vis 5: State x DDA Heatmap. Rows = States, Columns = DDA params."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    sns.heatmap(
        sensitivity_matrix, ax=ax,
        annot=True, fmt=".1f", cmap="coolwarm", center=0,
        linewidths=0.5, cbar_kws={"label": "Normalized Influence"},
        vmin=-1, vmax=1
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("DDA Parameter", fontsize=12, fontweight='bold')
    ax.set_ylabel("Player State", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    if save_name: save_figure(fig, save_name, subdir=TRANSITION_SUBDIR)
    return fig


def plot_clc_vs_transition(
    clc_values: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    title: str = "CLC vs State Transition Probability",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Vis 6: CLC vs State Transition (line plot)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    
    for state, probs in probs_dict.items():
        color = EMOTION_COLORS.get(state.lower(), "#333333")
        ax.plot(clc_values, probs, label=f"P({state.capitalize()})", color=color, linewidth=2.5)
        
    ax.set_xlabel("Cognitive Load Complexity (CLC)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Transition Probability", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_name: save_figure(fig, save_name, subdir=TRANSITION_SUBDIR)
    return fig


def plot_snr_vs_transition(
    snr_values: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    title: str = "SNR vs State Transition Probability",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Vis 7: SNR vs State Transition (line plot)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    
    for state, probs in probs_dict.items():
        color = EMOTION_COLORS.get(state.lower(), "#333333")
        ax.plot(snr_values, probs, label=f"P({state.capitalize()})", color=color, linewidth=2.5)
        
    ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Transition Probability", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_name: save_figure(fig, save_name, subdir=TRANSITION_SUBDIR)
    return fig


def plot_dda_change_histogram(
    dda_changes: np.ndarray,
    title: str = "DDA Parameter Change (Δp) Distribution",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Vis 8: Histogram of positive, negative, and zero DDA changes."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    
    bins = np.linspace(-1.0, 1.0, 41)
    sns.histplot(dda_changes, bins=bins, ax=ax, color=ECHO_PALETTE[0], kde=True)
    
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Calculate counts
    positive = np.sum(dda_changes > 1e-3)
    negative = np.sum(dda_changes < -1e-3)
    zero = np.sum(np.abs(dda_changes) <= 1e-3)
    
    textstr = f"Positive changes: {positive}\nNegative changes: {negative}\nZero/No change: {zero}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    ax.set_xlabel("Δp (Change Magnitude)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if save_name: save_figure(fig, save_name, subdir=TRANSITION_SUBDIR)
    return fig


def plot_state_duration_distribution(
    durations_dict: Dict[str, List[float]],
    title: str = "State Duration Distribution",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Vis 9: Violin or box plot of temporal persistence."""
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    
    data = []
    labels = []
    colors = []
    
    for state, durations in durations_dict.items():
        if durations:
            data.append(durations)
            labels.append(state.capitalize())
            colors.append(EMOTION_COLORS.get(state.lower(), "#333333"))
            
    if not data:
        ax.text(0.5, 0.5, "No duration data", ha='center', va='center')
        return fig
        
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
        
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylabel("Duration (steps)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if save_name: save_figure(fig, save_name, subdir=TRANSITION_SUBDIR)
    return fig


def plot_state_dda_flow(save_name: Optional[str] = None) -> plt.Figure:
    """Vis 10: State -> DDA -> Gameplay Flow conceptual plot."""
    # This is a conceptual flow diagram rendered using text/boxes in matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    boxes = [
        ("Observation (O_t)", 0.1, 0.5),
        ("Transition Matrix\nP(S_{t+1}|S_t)", 0.35, 0.5),
        ("Belief Vector\nb(s)", 0.6, 0.5),
        ("Belief-Weighted DDA\nΔp = Σ b(s)Δp(s)", 0.85, 0.5)
    ]
    
    for i, (text, x, y) in enumerate(boxes):
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='gray', linewidth=1.5))
                
        if i < len(boxes) - 1:
            ax.annotate('', xy=(boxes[i+1][1]-0.1, boxes[i+1][2]), 
                        xytext=(x+0.1, y),
                        arrowprops=dict(arrowstyle="->", lw=2, color='gray'))
                        
    ax.set_title("Belief-Weighted DDA Control Flow", fontsize=14, fontweight='bold', y=0.9)
    
    if save_name: save_figure(fig, save_name, subdir=TRANSITION_SUBDIR)
    return fig
