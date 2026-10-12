"""
Project ECHO — Evaluation Visualization Plots
==================================================
Publication-quality plots for supervised learning evaluation
of the DSTER emotion classifier.

Plots (10+ types):
    1. Confusion Matrix Heatmap (normalized)
    2. Pie Chart — Emotion Distribution (predictions + ground truth)
    3. Histogram — Confidence Score Distribution
    4. Bar Chart — Per-Class F1 Scores
    5. Grouped Bar Chart — Precision / Recall / F1
    6. Heatmap — Feature Correlation Matrix
    7. ROC Curves (per-class + macro average)
    8. Precision-Recall Curves
    9. Learning Curves (train vs. validation)
   10. Prediction Error Histogram (misclassification types)
   11. Feature Importance Bar Chart
   12. Confidence Calibration Scatter
   13. Cross-Validation Box Plot

All plots follow the ECHO research style configuration
(colorblind-safe palette, publication-quality formatting).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any


from visualization.plot_config import (
    FIGSIZE_SMALL, EMOTION_COLORS, ECHO_PALETTE, save_figure, setup_style,
    clear_output_dir, write_plot_manifest,
)

setup_style()

EMOTION_LABELS = ["angry", "focused", "thinking", "neutral", "joy", "given_up"]

# Evaluation subdirectory for all metric plots
EVAL_SUBDIR = "evaluation"


# ============================================================
# 1. CONFUSION MATRIX HEATMAP
# ============================================================

def plot_confusion_matrix_eval(
    confusion_mat: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "DSTER Emotion Classifier — Confusion Matrix",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Enhanced normalized confusion matrix with percentage annotations
    and marginal accuracy bars.

    Args:
        confusion_mat: (N, N) normalized confusion matrix.
        labels: Class label strings.
    """
    if labels is None:
        labels = EMOTION_LABELS

    n = len(labels)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create annotated heatmap
    annot_data = np.empty_like(confusion_mat, dtype=object)
    for i in range(n):
        for j in range(n):
            val = confusion_mat[i, j]
            annot_data[i, j] = f"{val:.2%}"

    sns.heatmap(
        confusion_mat, ax=ax,
        xticklabels=[l.capitalize() for l in labels],
        yticklabels=[l.capitalize() for l in labels],
        annot=annot_data, fmt="",
        cmap="Blues",
        linewidths=2,
        linecolor='white',
        cbar_kws={"label": "Normalized Frequency", "shrink": 0.8},
        vmin=0, vmax=1,
        square=True,
    )

    ax.set_xlabel("Predicted Label", fontsize=13, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # Add per-class accuracy annotation on diagonal
    for i in range(n):
        acc = confusion_mat[i, i]
        ax.text(i + 0.5, n + 0.15, f"{acc:.1%}",
                ha='center', va='top', fontsize=9, color=ECHO_PALETTE[2],
                fontweight='bold')

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 2. PIE CHART — EMOTION DISTRIBUTION
# ============================================================

def plot_emotion_pie_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Emotion Distribution — Ground Truth vs. Predictions",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side pie charts of ground truth vs. predicted class distributions.

    Args:
        y_true: Ground truth labels (integer).
        y_pred: Predicted labels (integer).
        labels: Class label strings.
    """
    if labels is None:
        labels = EMOTION_LABELS

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, data, subtitle in zip(
        axes, [y_true, y_pred], ["Ground Truth", "Predictions"]
    ):
        # Ensure all states are shown even if count is 0
        counts = [np.sum(data == i) for i in range(len(labels))]
        pie_labels = labels
        colors = [EMOTION_COLORS.get(label, "#999999") for label in labels]

        # Only show percentage text for non-zero slices to avoid overlap
        def my_autopct(pct):
            return f"{pct:1.1f}%" if pct > 0 else ""

        wedges, texts, autotexts = ax.pie(
            counts,
            labels=[l.capitalize() for l in pie_labels],
            colors=colors,
            autopct=my_autopct,
            startangle=90,
            pctdistance=0.75,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )

        # Style the percentage text
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight("bold")
        for text in texts:
            text.set_fontsize(11)

        ax.set_title(subtitle, fontsize=14, fontweight='bold', pad=15)

        # Add donut hole
        centre_circle = plt.Circle((0, 0), 0.5, fc='white')
        ax.add_artist(centre_circle)

        # Center text
        total = sum(counts)
        ax.text(0, 0, f"n={total}", ha='center', va='center',
                fontsize=14, fontweight='bold', color='#333333')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 3. HISTOGRAM — CONFIDENCE DISTRIBUTION
# ============================================================

def plot_confidence_histogram(
    y_scores: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "DSTER Confidence Score Distribution",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram of softmax confidence scores, split by correct vs. incorrect.

    Args:
        y_scores: (N, num_classes) softmax probabilities.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Class label strings.
    """
    if labels is None:
        labels = EMOTION_LABELS

    max_confs = np.max(y_scores, axis=1)
    correct = y_true == y_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: overall distribution split by correct/incorrect
    axes[0].hist(
        max_confs[correct], bins=30, alpha=0.7,
        color=ECHO_PALETTE[2], label="Correct", edgecolor='white'
    )
    axes[0].hist(
        max_confs[~correct], bins=30, alpha=0.7,
        color=ECHO_PALETTE[4], label="Incorrect", edgecolor='white'
    )
    axes[0].set_xlabel("Max Softmax Confidence", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Confidence: Correct vs. Incorrect", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    if correct.sum() > 0:
        axes[0].axvline(np.mean(max_confs[correct]), color=ECHO_PALETTE[2],
                        linestyle='--', alpha=0.8, linewidth=1.5)
    if (~correct).sum() > 0:
        axes[0].axvline(np.mean(max_confs[~correct]),
                        color=ECHO_PALETTE[4], linestyle='--', alpha=0.8, linewidth=1.5)

    # Right: per-class confidence distribution
    class_confs = []
    class_labels = []
    for i, label in enumerate(labels):
        mask = y_pred == i
        if mask.sum() > 0:
            class_confs.append(max_confs[mask])
            class_labels.append(label.capitalize())

    if class_confs:
        bp = axes[1].boxplot(
            class_confs, tick_labels=class_labels, patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch, label in zip(bp['boxes'], [l.lower() for l in class_labels]):
            patch.set_facecolor(EMOTION_COLORS.get(label, ECHO_PALETTE[0]))
            patch.set_alpha(0.7)

    axes[1].set_xlabel("Emotion Class", fontsize=12)
    axes[1].set_ylabel("Confidence Score", fontsize=12)
    axes[1].set_title("Per-Class Confidence Distribution", fontsize=13, fontweight='bold')

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 4. BAR CHART — PER-CLASS F1 SCORES
# ============================================================

def plot_f1_bar_chart(
    per_class_f1: Dict[str, float],
    macro_f1: float,
    target_f1: float = 0.95,
    title: str = "Per-Class F1 Scores — DSTER Emotion Classifier",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of F1 scores per emotion class with target threshold.

    Args:
        per_class_f1: Dict mapping label → F1 score.
        macro_f1: Macro-averaged F1 score.
        target_f1: Target F1 threshold (drawn as horizontal line).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    classes = list(per_class_f1.keys())
    f1_values = [per_class_f1[c] for c in classes]
    colors = [EMOTION_COLORS.get(c, ECHO_PALETTE[0]) for c in classes]

    bars = ax.bar(
        [c.capitalize() for c in classes], f1_values,
        color=colors, edgecolor='white', linewidth=1.5, width=0.6,
    )

    # Value labels on bars
    for bar, val in zip(bars, f1_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Target line
    ax.axhline(target_f1, color='red', linestyle='--', alpha=0.6,
               linewidth=1.5, label=f"Target F1 ({target_f1})")

    # Macro F1 line
    ax.axhline(macro_f1, color=ECHO_PALETTE[5], linestyle='-.',
               alpha=0.6, linewidth=1.5, label=f"Macro F1 ({macro_f1:.3f})")

    ax.set_xlabel("Emotion Class", fontsize=12, fontweight='bold')
    ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 5. GROUPED BAR CHART — PRECISION / RECALL / F1
# ============================================================

def plot_precision_recall_f1_bars(
    per_class_precision: Dict[str, float],
    per_class_recall: Dict[str, float],
    per_class_f1: Dict[str, float],
    title: str = "Precision / Recall / F1 per Emotion Class",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing precision, recall, and F1 per class.

    Args:
        per_class_precision: Dict mapping label → precision.
        per_class_recall: Dict mapping label → recall.
        per_class_f1: Dict mapping label → F1 score.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    classes = list(per_class_f1.keys())
    n = len(classes)
    x = np.arange(n)
    width = 0.25

    prec_vals = [per_class_precision[c] for c in classes]
    rec_vals = [per_class_recall[c] for c in classes]
    f1_vals = [per_class_f1[c] for c in classes]

    bars1 = ax.bar(x - width, prec_vals, width, label='Precision',
                   color=ECHO_PALETTE[0], edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, rec_vals, width, label='Recall',
                   color=ECHO_PALETTE[1], edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, f1_vals, width, label='F1 Score',
                   color=ECHO_PALETTE[2], edgecolor='white', linewidth=1)

    # Value annotations
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f"{height:.2f}", ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

    ax.set_xlabel("Emotion Class", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 6. HEATMAP — FEATURE CORRELATION MATRIX
# ============================================================

def plot_feature_correlation_heatmap(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Input Feature Correlation Matrix",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of Pearson correlation between input feature channels.

    Args:
        X: (N, seq_len, num_channels) input data — will average over seq_len.
        feature_names: Names for each feature channel.
    """
    if feature_names is None:
        feature_names = ["Press Duration", "IKI Latency", "Key Velocity", "Flight Time"]

    # Average over sequence dimension
    X_mean = X.mean(axis=1) if X.ndim == 3 else X
    n_features = min(X_mean.shape[1], len(feature_names))
    X_mean = X_mean[:, :n_features]

    corr = np.corrcoef(X_mean.T)

    fig, ax = plt.subplots(figsize=(8, 7))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, mask=mask,
        xticklabels=feature_names[:n_features],
        yticklabels=feature_names[:n_features],
        annot=True, fmt=".3f",
        cmap="RdBu_r", center=0,
        linewidths=2, linecolor='white',
        vmin=-1, vmax=1,
        square=True,
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.8},
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 7. ROC CURVES
# ============================================================

def plot_roc_curves_eval(
    per_class_roc: Dict[str, Dict[str, Any]],
    per_class_auc: Dict[str, float],
    macro_auc: float,
    labels: Optional[List[str]] = None,
    title: str = "Per-Class ROC Curves — DSTER Emotion Classifier",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Per-class ROC curves with AUC values and macro average.

    Args:
        per_class_roc: Dict mapping label → {"fpr": [...], "tpr": [...]}.
        per_class_auc: Dict mapping label → AUC value.
        macro_auc: Macro-averaged AUC.
        labels: Class label strings.
    """
    if labels is None:
        labels = EMOTION_LABELS

    fig, ax = plt.subplots(figsize=(10, 8))

    for label in labels:
        if label not in per_class_roc or not per_class_roc[label]["fpr"]:
            continue
        fpr = per_class_roc[label]["fpr"]
        tpr = per_class_roc[label]["tpr"]
        auc_val = per_class_auc.get(label, 0)
        color = EMOTION_COLORS.get(label, ECHO_PALETTE[0])

        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{label.capitalize()} (AUC = {auc_val:.3f})")

    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5, label="Random Baseline")

    # Macro AUC annotation
    ax.text(0.65, 0.15, f"Macro AUC = {macro_auc:.3f}",
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="orange", alpha=0.9),
            transform=ax.transAxes)

    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 8. PRECISION-RECALL CURVES
# ============================================================

def plot_precision_recall_curves(
    per_class_pr: Dict[str, Dict[str, Any]],
    per_class_avg_precision: Dict[str, float],
    labels: Optional[List[str]] = None,
    title: str = "Precision-Recall Curves — DSTER Emotion Classifier",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Per-class precision-recall curves with average precision.

    Args:
        per_class_pr: Dict mapping label → {"precision": [...], "recall": [...]}.
        per_class_avg_precision: Dict mapping label → average precision.
        labels: Class label strings.
    """
    if labels is None:
        labels = EMOTION_LABELS

    fig, ax = plt.subplots(figsize=(10, 8))

    for label in labels:
        if label not in per_class_pr or not per_class_pr[label]["precision"]:
            continue
        precision = per_class_pr[label]["precision"]
        recall = per_class_pr[label]["recall"]
        ap = per_class_avg_precision.get(label, 0)
        color = EMOTION_COLORS.get(label, ECHO_PALETTE[0])

        ax.plot(recall, precision, color=color, linewidth=2,
                label=f"{label.capitalize()} (AP = {ap:.3f})")

    # Baseline
    ax.axhline(0.2, color='grey', linestyle='--', alpha=0.4,
               label="Random Baseline (0.20)")

    ax.set_xlabel("Recall", fontsize=12, fontweight='bold')
    ax.set_ylabel("Precision", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 9. LEARNING CURVES
# ============================================================

def plot_learning_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    title: str = "DSTER Training & Validation Curves",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Training and validation loss/accuracy curves over epochs.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_accuracies: Training accuracy per epoch.
        val_accuracies: Validation accuracy per epoch.
    """
    has_accuracy = train_accuracies is not None
    n_plots = 2 if has_accuracy else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss
    axes[0].plot(epochs, train_losses, color=ECHO_PALETTE[0],
                 label="Train Loss", linewidth=2, marker='o', markersize=3)
    if val_losses:
        axes[0].plot(epochs, val_losses, color=ECHO_PALETTE[4],
                     label="Val Loss", linewidth=2, linestyle='--',
                     marker='s', markersize=3)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Loss Curves", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)

    # Accuracy
    if has_accuracy:
        axes[1].plot(epochs, train_accuracies, color=ECHO_PALETTE[2],
                     label="Train Accuracy", linewidth=2, marker='o', markersize=3)
        if val_accuracies:
            axes[1].plot(epochs, val_accuracies, color=ECHO_PALETTE[3],
                         label="Val Accuracy", linewidth=2, linestyle='--',
                         marker='s', markersize=3)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title("Accuracy Curves", fontsize=13, fontweight='bold')
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=10)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 10. PREDICTION ERROR HISTOGRAM
# ============================================================

def plot_prediction_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Misclassification Error Analysis",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of misclassification pairs (true → predicted).

    Args:
        y_true: Ground truth labels (integer).
        y_pred: Predicted labels (integer).
        labels: Class label strings.
    """
    if labels is None:
        labels = EMOTION_LABELS

    # Find misclassifications
    errors = y_true != y_pred
    if errors.sum() == 0:
        fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
        ax.text(0.5, 0.5, "No misclassifications! (Perfect accuracy)",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, fontweight='bold', color=ECHO_PALETTE[2])
        ax.set_title(title, fontsize=14, fontweight='bold')
        if save_name:
            save_figure(fig, save_name, subdir=EVAL_SUBDIR)
        return fig

    true_errors = y_true[errors]
    pred_errors = y_pred[errors]

    # Count error pairs
    error_pairs = {}
    for t, p in zip(true_errors, pred_errors):
        key = f"{labels[int(t)].capitalize()} → {labels[int(p)].capitalize()}"
        error_pairs[key] = error_pairs.get(key, 0) + 1

    # Sort by frequency
    sorted_pairs = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
    pair_labels = [p[0] for p in sorted_pairs[:12]]  # Top 12
    pair_counts = [p[1] for p in sorted_pairs[:12]]

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [ECHO_PALETTE[i % len(ECHO_PALETTE)] for i in range(len(pair_labels))]
    bars = ax.barh(pair_labels[::-1], pair_counts[::-1],
                   color=colors[::-1], edgecolor='white', linewidth=1.5)

    # Value labels
    for bar, count in zip(bars, pair_counts[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel("Error Count", fontsize=12, fontweight='bold')
    ax.set_ylabel("Misclassification (True → Predicted)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Summary annotation
    total_errors = errors.sum()
    total_samples = len(y_true)
    ax.text(0.95, 0.05,
            f"Total Errors: {total_errors}/{total_samples}\n"
            f"Error Rate: {total_errors/total_samples:.1%}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd",
                      edgecolor="#ffc107", alpha=0.9))

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 11. FEATURE IMPORTANCE BAR CHART
# ============================================================

def plot_feature_importance(
    feature_importance: Dict[str, float],
    title: str = "DSTER Feature Importance (Gradient-Based)",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of feature importance scores.

    Args:
        feature_importance: Dict mapping feature name → importance score.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    names = [f[0].replace("_", " ").title() for f in sorted_features]
    values = [f[1] for f in sorted_features]

    colors = ECHO_PALETTE[:len(names)]
    bars = ax.barh(names[::-1], values[::-1],
                   color=colors[::-1], edgecolor='white', linewidth=1.5,
                   height=0.5)

    # Value labels
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel("Relative Importance", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.2)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 12. CONFIDENCE CALIBRATION SCATTER
# ============================================================

def plot_confidence_calibration(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
    title: str = "Confidence Calibration Plot",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Reliability diagram: predicted confidence vs. actual accuracy.

    Args:
        y_true: Ground truth labels.
        y_scores: Predicted probabilities (N, num_classes).
        n_bins: Number of calibration bins.
    """
    max_confs = np.max(y_scores, axis=1)
    y_pred = np.argmax(y_scores, axis=1)
    correct = (y_true == y_pred).astype(float)

    # Bin by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (max_confs >= bin_edges[i]) & (max_confs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(max_confs[mask].mean())
            bin_counts.append(mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5, label="Perfect Calibration")
    ax.bar(bin_confs, bin_accs, width=1.0 / n_bins * 0.8,
           alpha=0.7, color=ECHO_PALETTE[0], edgecolor='white',
           label="Model Calibration")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax.set_title("Reliability Diagram", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Right: Confidence histogram
    ax2 = axes[1]
    ax2.hist(max_confs, bins=30, color=ECHO_PALETTE[1], alpha=0.7,
             edgecolor='white')
    ax2.set_xlabel("Predicted Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Confidence Distribution", fontsize=13, fontweight='bold')

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# 13. CROSS-VALIDATION BOX PLOT
# ============================================================

def plot_cross_validation_scores(
    cv_results: Dict[str, Any],
    title: str = "K-Fold Cross-Validation Results",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Box plot of cross-validation accuracy and F1 scores.

    Args:
        cv_results: Dict with 'accuracies' and 'f1_scores' lists.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [cv_results["accuracies"], cv_results["f1_scores"]]
    labels_cv = ["Accuracy", "Macro F1"]

    bp = ax.boxplot(
        data, tick_labels=labels_cv, patch_artist=True,
        widths=0.4,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )

    colors = [ECHO_PALETTE[0], ECHO_PALETTE[2]]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual fold points
    for i, scores in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(scores))
        ax.scatter(x, scores, alpha=0.8, s=50, color='black', zorder=5,
                   edgecolors='white', linewidth=0.5)

    # Annotate mean ± std
    for i, (scores, name) in enumerate(zip(data, labels_cv)):
        mean = np.mean(scores)
        std = np.std(scores)
        ax.text(i + 1, max(scores) + 0.02,
                f"μ={mean:.3f} ± {std:.3f}",
                ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir=EVAL_SUBDIR)
    return fig


# ============================================================
# MASTER PLOT GENERATOR
# ============================================================

def generate_all_evaluation_plots(
    results,
    X_data: Optional[np.ndarray] = None,
    cv_results: Optional[Dict] = None,
    training_history: Optional[Dict] = None,
    prefix: str = "dster",
) -> List[str]:
    """
    Generate all evaluation plots from an EvaluationResults object.

    Args:
        results: EvaluationResults from DSTEREvaluator.evaluate().
        X_data: Optional input data for feature correlation plots.
        cv_results: Optional cross-validation results.
        training_history: Optional dict with train/val loss/accuracy lists.
        prefix: Filename prefix for saved plots.

    Returns:
        List of saved plot filenames.
    """
    saved = []

    # Clean out stale plots from previous runs so only fresh plots remain
    clear_output_dir(EVAL_SUBDIR)

    # 1. Confusion Matrix
    if results.confusion_matrix_normalized is not None:
        plot_confusion_matrix_eval(
            results.confusion_matrix_normalized,
            labels=results.labels,
            save_name=f"{prefix}_confusion_matrix",
        )
        saved.append(f"{prefix}_confusion_matrix")
        print("  [OK] Confusion matrix heatmap")

    # 2. Pie Charts
    if results.y_true is not None and results.y_pred is not None:
        plot_emotion_pie_chart(
            results.y_true, results.y_pred,
            labels=results.labels,
            save_name=f"{prefix}_emotion_pie_chart",
        )
        saved.append(f"{prefix}_emotion_pie_chart")
        print("  [OK] Emotion distribution pie charts")

    # 3. Confidence Histogram
    if results.y_scores is not None:
        plot_confidence_histogram(
            results.y_scores, results.y_true, results.y_pred,
            labels=results.labels,
            save_name=f"{prefix}_confidence_histogram",
        )
        saved.append(f"{prefix}_confidence_histogram")
        print("  [OK] Confidence distribution histogram")

    # 4. F1 Bar Chart
    if results.per_class_f1:
        plot_f1_bar_chart(
            results.per_class_f1,
            results.macro_f1,
            save_name=f"{prefix}_f1_scores",
        )
        saved.append(f"{prefix}_f1_scores")
        print("  [OK] Per-class F1 bar chart")

    # 5. Precision / Recall / F1 Grouped Bars
    if results.per_class_precision:
        plot_precision_recall_f1_bars(
            results.per_class_precision,
            results.per_class_recall,
            results.per_class_f1,
            save_name=f"{prefix}_precision_recall_f1",
        )
        saved.append(f"{prefix}_precision_recall_f1")
        print("  [OK] Precision/Recall/F1 grouped bar chart")

    # 6. Feature Correlation Heatmap
    if X_data is not None:
        plot_feature_correlation_heatmap(
            X_data,
            save_name=f"{prefix}_feature_correlation",
        )
        saved.append(f"{prefix}_feature_correlation")
        print("  [OK] Feature correlation heatmap")

    # 7. ROC Curves
    if results.per_class_roc:
        plot_roc_curves_eval(
            results.per_class_roc,
            results.per_class_auc,
            results.macro_auc,
            labels=results.labels,
            save_name=f"{prefix}_roc_curves",
        )
        saved.append(f"{prefix}_roc_curves")
        print("  [OK] ROC curves with AUC")

    # 8. Precision-Recall Curves
    if results.per_class_pr:
        plot_precision_recall_curves(
            results.per_class_pr,
            results.per_class_avg_precision,
            labels=results.labels,
            save_name=f"{prefix}_pr_curves",
        )
        saved.append(f"{prefix}_pr_curves")
        print("  [OK] Precision-Recall curves")

    # 9. Learning Curves (if training history available)
    if training_history:
        plot_learning_curves(
            train_losses=training_history.get("train_losses", []),
            val_losses=training_history.get("val_losses"),
            train_accuracies=training_history.get("train_accuracies"),
            val_accuracies=training_history.get("val_accuracies"),
            save_name=f"{prefix}_learning_curves",
        )
        saved.append(f"{prefix}_learning_curves")
        print("  [OK] Learning curves")

    # 10. Prediction Error Histogram
    if results.y_true is not None and results.y_pred is not None:
        plot_prediction_error_histogram(
            results.y_true, results.y_pred,
            labels=results.labels,
            save_name=f"{prefix}_error_analysis",
        )
        saved.append(f"{prefix}_error_analysis")
        print("  [OK] Prediction error histogram")

    # 11. Feature Importance
    if results.feature_importance:
        plot_feature_importance(
            results.feature_importance,
            save_name=f"{prefix}_feature_importance",
        )
        saved.append(f"{prefix}_feature_importance")
        print("  [OK] Feature importance bar chart")

    # 12. Confidence Calibration
    if results.y_scores is not None:
        plot_confidence_calibration(
            results.y_true, results.y_scores,
            save_name=f"{prefix}_calibration",
        )
        saved.append(f"{prefix}_calibration")
        print("  [OK] Confidence calibration plot")

    # 13. Cross-Validation Box Plot
    if cv_results is not None:
        plot_cross_validation_scores(
            cv_results,
            save_name=f"{prefix}_cross_validation",
        )
        saved.append(f"{prefix}_cross_validation")
        print("  [OK] Cross-validation box plot")

    # Write a manifest so plot freshness can be verified without checking file timestamps
    if saved:
        write_plot_manifest(EVAL_SUBDIR, saved)

    return saved
