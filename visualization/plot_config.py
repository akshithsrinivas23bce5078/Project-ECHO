"""
Project ECHO — Plot Configuration
====================================
Research-grade matplotlib/seaborn style configuration for
publication-quality figures.

All plot modules import this to ensure consistent aesthetics
across all generated figures.
"""

import os
import time
import json
import glob
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================
# Style Configuration
# ============================================

# Research-grade defaults
PLOT_DPI = 150
FIGSIZE_DEFAULT = (12, 8)
FIGSIZE_WIDE = (16, 8)
FIGSIZE_SQUARE = (8, 8)
FIGSIZE_SMALL = (8, 6)
SAVE_FORMAT = "png"

# Colorblind-safe palette (Wong, 2011)
ECHO_PALETTE = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#D55E00",  # Red-Orange
    "#56B4E9",  # Light Blue
    "#F0E442",  # Yellow
    "#999999",  # Grey
]

# Emotion-specific colors — mapped to the 6 ACTUAL demo states only.
# Old DSTER model labels (frustrated, fatigued, engaged) are NOT included
# because they never appear in session data and cause phantom plot entries.
EMOTION_COLORS = {
    "angry": "#D55E00",
    "focused": "#0072B2",
    "thinking": "#CC79A7",
    "neutral": "#56B4E9",
    "joy": "#009E73",
    "given_up": "#999999",
}

# Game genre colors (focusing on the 4 primary games)
GENRE_COLORS = {
    "left_4_dead": "#D55E00",
    "alien_isolation": "#0072B2",
    "resident_evil_4": "#009E73",
    "silent_hill": "#CC79A7",
}

# Output directory
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'plots'
)


def setup_style():
    """Apply the ECHO research plot style globally."""
    plt.rcParams.update({
        'figure.figsize': FIGSIZE_DEFAULT,
        'figure.dpi': PLOT_DPI,
        'savefig.dpi': PLOT_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.framealpha': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
    })
    sns.set_palette(ECHO_PALETTE)


def get_output_dir(subdir: str = "") -> str:
    """Get the output directory, creating it if necessary."""
    out = os.path.join(DEFAULT_OUTPUT_DIR, subdir) if subdir else DEFAULT_OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    return out


def clear_output_dir(subdir: str = ""):
    """
    Remove all .png plot files and the old manifest from an output
    subdirectory so that only freshly generated plots remain after a run.

    This prevents stale plots from previous sessions from persisting
    alongside newly generated ones.

    Args:
        subdir: Subdirectory within the plots output folder to clear.
    """
    out_dir = get_output_dir(subdir)
    removed = 0
    for pattern in (f"*.{SAVE_FORMAT}", "_manifest.json"):
        for filepath in glob.glob(os.path.join(out_dir, pattern)):
            try:
                os.remove(filepath)
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"  [CLEANUP] Removed {removed} stale file(s) from {subdir or 'plots/'}")


def write_plot_manifest(subdir: str, filenames: list):
    """
    Write a JSON manifest recording which plots were generated and when.

    This makes it easy to verify plot freshness without inspecting file
    timestamps on disk.

    Args:
        subdir: Subdirectory the plots were saved into.
        filenames: List of plot basenames (without extension) that were saved.
    """
    out_dir = get_output_dir(subdir)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "subdir": subdir,
        "num_plots": len(filenames),
        "plots": [f"{name}.{SAVE_FORMAT}" for name in filenames],
    }
    manifest_path = os.path.join(out_dir, "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def save_figure(fig, filename: str, subdir: str = "", close: bool = True):
    """Save a figure to the output directory with a generation timestamp."""
    out_dir = get_output_dir(subdir)
    filepath = os.path.join(out_dir, f"{filename}.{SAVE_FORMAT}")

    # Add a small timestamp watermark so stale plots are immediately obvious
    timestamp_str = time.strftime("Generated: %Y-%m-%d %H:%M")
    fig.text(
        0.99, 0.01, timestamp_str,
        ha='right', va='bottom',
        fontsize=7, color='#aaaaaa', alpha=0.7,
        transform=fig.transFigure,
    )

    fig.savefig(filepath, format=SAVE_FORMAT, dpi=PLOT_DPI, bbox_inches='tight')
    if close:
        plt.close(fig)
    return filepath


# Apply style on import
setup_style()
