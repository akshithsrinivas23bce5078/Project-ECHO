"""
Project ECHO — DSTER Model Plots
====================================
Visualization of DSTER emotion classifier performance,
training curves, and latent space analysis.

Plots:
    1. Confusion matrix
    2. Training curves (loss + accuracy)
    3. t-SNE latent space embedding
    4. Attention weight heatmap
    5. Per-class ROC curves
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE

from visualization.plot_config import (
    FIGSIZE_DEFAULT, FIGSIZE_SMALL, FIGSIZE_SQUARE,
    EMOTION_COLORS, ECHO_PALETTE, save_figure, setup_style,
)

setup_style()

EMOTION_LABELS = ["frustrated", "focused", "fatigued", "neutral", "engaged"]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "DSTER Emotion Classification — Confusion Matrix",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Normalized confusion matrix for emotion classification.

    Args:
        y_true: Ground truth labels (integer indices).
        y_pred: Predicted labels (integer indices).
        labels: String labels for each class.
    """
    if labels is None:
        labels = EMOTION_LABELS

    cm = confusion_matrix(y_true, y_pred, normalize='true')

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    sns.heatmap(
        cm, ax=ax,
        xticklabels=[l.capitalize() for l in labels],
        yticklabels=[l.capitalize() for l in labels],
        annot=True, fmt=".2f",
        cmap="Blues",
        linewidths=1,
        cbar_kws={"label": "Normalized Frequency"},
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    if save_name:
        save_figure(fig, save_name, subdir="dster")
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    title: str = "DSTER Training Curves",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Loss and accuracy curves over training epochs.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch (optional).
        train_accuracies: Training accuracy per epoch (optional).
        val_accuracies: Validation accuracy per epoch (optional).
    """
    has_accuracy = train_accuracies is not None
    n_plots = 2 if has_accuracy else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, color=ECHO_PALETTE[0],
                 label="Train Loss", linewidth=1.5)
    if val_losses:
        axes[0].plot(epochs, val_losses, color=ECHO_PALETTE[1],
                     label="Val Loss", linewidth=1.5, linestyle='--')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Accuracy plot
    if has_accuracy:
        axes[1].plot(epochs, train_accuracies, color=ECHO_PALETTE[2],
                     label="Train Accuracy", linewidth=1.5)
        if val_accuracies:
            axes[1].plot(epochs, val_accuracies, color=ECHO_PALETTE[3],
                         label="Val Accuracy", linewidth=1.5, linestyle='--')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, subdir="dster")
    return fig


def plot_tsne_embedding(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None,
    perplexity: float = 30.0,
    title: str = "DSTER Latent Space (t-SNE)",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    t-SNE visualization of DSTER latent embeddings.

    Args:
        embeddings: (N, embed_dim) array of DSTER embeddings.
        labels: (N,) integer class labels.
        label_names: String names for each class.
        perplexity: t-SNE perplexity parameter.
    """
    if label_names is None:
        label_names = EMOTION_LABELS

    # Run t-SNE
    n_samples = embeddings.shape[0]
    perplexity = min(perplexity, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embedded = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    for cls_idx, name in enumerate(label_names):
        mask = labels == cls_idx
        if mask.any():
            color = EMOTION_COLORS.get(name, ECHO_PALETTE[cls_idx % len(ECHO_PALETTE)])
            ax.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=color, label=name.capitalize(),
                s=30, alpha=0.7, edgecolors='white', linewidth=0.3,
            )

    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title(title)
    ax.legend(title="Emotion")

    if save_name:
        save_figure(fig, save_name, subdir="dster")
    return fig


def plot_attention_weights(
    attention_matrix: np.ndarray,
    title: str = "Transformer Attention Weights",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of transformer self-attention weights.

    Args:
        attention_matrix: (seq_len, seq_len) attention weight matrix.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    sns.heatmap(
        attention_matrix, ax=ax,
        cmap="viridis",
        xticklabels=5,
        yticklabels=5,
        cbar_kws={"label": "Attention Weight"},
    )

    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(title)

    if save_name:
        save_figure(fig, save_name, subdir="dster")
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "DSTER Per-Class ROC Curves",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Per-class ROC curves with AUC values.

    Args:
        y_true: (N,) integer ground truth labels.
        y_scores: (N, num_classes) predicted probabilities.
        labels: Class label strings.
    """
    if labels is None:
        labels = EMOTION_LABELS

    n_classes = y_scores.shape[1]
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)

    for cls_idx in range(n_classes):
        binary_true = (y_true == cls_idx).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_scores[:, cls_idx])
        roc_auc = auc(fpr, tpr)

        color = EMOTION_COLORS.get(labels[cls_idx], ECHO_PALETTE[cls_idx % len(ECHO_PALETTE)])
        ax.plot(fpr, tpr, color=color, linewidth=1.5,
                label=f"{labels[cls_idx].capitalize()} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    if save_name:
        save_figure(fig, save_name, subdir="dster")
    return fig
