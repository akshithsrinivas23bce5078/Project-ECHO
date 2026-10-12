"""
Project ECHO — Supervised Learning Evaluation Metrics
=========================================================
Comprehensive evaluation suite for the DSTER emotion classifier.

Computes all standard supervised classification metrics:
    - Confusion matrix (normalized and raw)
    - Per-class precision, recall, F1-score
    - Macro/weighted/micro averaged metrics
    - ROC curves with AUC (per-class + macro)
    - Precision-Recall curves with average precision
    - Cohen's Kappa (inter-rater agreement)
    - Matthews Correlation Coefficient (MCC)
    - Cross-validation scores (K-fold)
    - Feature importance analysis

The DSTER model performs 5-class supervised classification:
    Classes: frustrated, focused, fatigued, neutral, engaged

Usage:
    from evaluation.evaluation_metrics import DSTEREvaluator
    evaluator = DSTEREvaluator(model)
    results = evaluator.evaluate(X_test, y_test)
    evaluator.save_report(results, "results/evaluation/")
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    accuracy_score,
    log_loss,
)
from sklearn.preprocessing import label_binarize


# Demo state labels (6-class: matching the live state machine output)
EMOTION_LABELS = ["angry", "focused", "thinking", "neutral", "joy", "given_up"]
NUM_CLASSES = len(EMOTION_LABELS)


@dataclass
class EvaluationResults:
    """Container for all evaluation metric results."""

    # Core metrics
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    micro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_auc: float = 0.0
    cohen_kappa: float = 0.0
    matthews_corrcoef: float = 0.0
    log_loss_value: float = 0.0

    # Per-class metrics
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_support: Dict[str, int] = field(default_factory=dict)

    # Confusion matrix
    confusion_matrix_raw: Optional[np.ndarray] = None
    confusion_matrix_normalized: Optional[np.ndarray] = None

    # ROC / AUC
    per_class_roc: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_class_auc: Dict[str, float] = field(default_factory=dict)
    macro_auc: float = 0.0

    # Precision-Recall
    per_class_pr: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_class_avg_precision: Dict[str, float] = field(default_factory=dict)

    # Confidence analysis
    confidence_distribution: Dict[str, List[float]] = field(default_factory=dict)
    mean_confidence_correct: float = 0.0
    mean_confidence_incorrect: float = 0.0

    # Feature importance
    feature_importance: Optional[Dict[str, float]] = None

    # Prediction arrays (for downstream plotting)
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_scores: Optional[np.ndarray] = None

    # Metadata
    num_samples: int = 0
    evaluation_time_ms: float = 0.0
    model_name: str = "DSTER Emotion Classifier"
    labels: List[str] = field(default_factory=lambda: list(EMOTION_LABELS))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "model_name": self.model_name,
            "num_samples": self.num_samples,
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
            "labels": self.labels,
            "overall_metrics": {
                "accuracy": round(self.accuracy, 6),
                "macro_f1": round(self.macro_f1, 6),
                "macro_precision": round(self.macro_precision, 6),
                "macro_recall": round(self.macro_recall, 6),
                "weighted_f1": round(self.weighted_f1, 6),
                "micro_f1": round(self.micro_f1, 6),
                "cohen_kappa": round(self.cohen_kappa, 6),
                "matthews_corrcoef": round(self.matthews_corrcoef, 6),
                "log_loss": round(self.log_loss_value, 6),
                "macro_auc": round(self.macro_auc, 6),
            },
            "per_class_metrics": {},
            "confidence_analysis": {
                "mean_confidence_correct": round(self.mean_confidence_correct, 4),
                "mean_confidence_incorrect": round(self.mean_confidence_incorrect, 4),
            },
        }

        for label in self.labels:
            result["per_class_metrics"][label] = {
                "precision": round(self.per_class_precision.get(label, 0), 6),
                "recall": round(self.per_class_recall.get(label, 0), 6),
                "f1_score": round(self.per_class_f1.get(label, 0), 6),
                "support": self.per_class_support.get(label, 0),
                "auc": round(self.per_class_auc.get(label, 0), 6),
                "avg_precision": round(self.per_class_avg_precision.get(label, 0), 6),
            }

        if self.feature_importance:
            result["feature_importance"] = {
                k: round(v, 6) for k, v in self.feature_importance.items()
            }

        return result


class DSTEREvaluator:
    """
    Comprehensive evaluation pipeline for the DSTER emotion classifier.

    Computes all standard supervised classification metrics and
    prepares data for visualization.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        labels: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.labels = labels or EMOTION_LABELS
        self.num_classes = len(self.labels)
        self.device = device

        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()

    def evaluate(
        self,
        X: Optional[torch.Tensor] = None,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        y_scores: Optional[np.ndarray] = None,
    ) -> EvaluationResults:
        """
        Run full evaluation pipeline.

        Either provide (X, y_true) for model inference,
        or provide (y_true, y_pred, y_scores) directly.

        Args:
            X: Input tensor (N, seq_len, num_channels) for model inference.
            y_true: Ground truth integer labels (N,).
            y_pred: Predicted integer labels (N,). Computed from model if None.
            y_scores: Predicted probabilities (N, num_classes). Computed if None.

        Returns:
            EvaluationResults with all metrics.
        """
        t0 = time.perf_counter()
        results = EvaluationResults(labels=list(self.labels))

        # Run inference if needed
        if y_pred is None or y_scores is None:
            if self.model is None or X is None:
                raise ValueError(
                    "Must provide either (model + X) for inference, "
                    "or (y_pred + y_scores) directly."
                )
            y_pred, y_scores = self._run_inference(X)

        if y_true is None:
            raise ValueError("y_true (ground truth labels) is required.")

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_scores = np.asarray(y_scores)

        results.num_samples = len(y_true)
        results.y_true = y_true
        results.y_pred = y_pred
        results.y_scores = y_scores

        # ---- Core Metrics ----
        results.accuracy = float(accuracy_score(y_true, y_pred))

        # Per-class precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(self.num_classes)), zero_division=0
        )
        for i, label in enumerate(self.labels):
            results.per_class_precision[label] = float(precision[i])
            results.per_class_recall[label] = float(recall[i])
            results.per_class_f1[label] = float(f1[i])
            results.per_class_support[label] = int(support[i])

        # Macro/Weighted/Micro Metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        _, _, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        _, _, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        results.macro_f1 = float(macro_f1)
        results.macro_precision = float(macro_precision)
        results.macro_recall = float(macro_recall)
        results.weighted_f1 = float(weighted_f1)
        results.micro_f1 = float(micro_f1)

        # Cohen's Kappa
        results.cohen_kappa = float(cohen_kappa_score(y_true, y_pred))

        # Matthews Correlation Coefficient
        results.matthews_corrcoef = float(matthews_corrcoef(y_true, y_pred))

        # Log Loss
        try:
            results.log_loss_value = float(log_loss(y_true, y_scores, labels=list(range(self.num_classes))))
        except ValueError:
            results.log_loss_value = float("nan")

        # ---- Confusion Matrix ----
        results.confusion_matrix_raw = confusion_matrix(
            y_true, y_pred, labels=list(range(self.num_classes))
        )
        results.confusion_matrix_normalized = confusion_matrix(
            y_true, y_pred, labels=list(range(self.num_classes)), normalize="true"
        )

        # ---- ROC Curves + AUC ----
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        all_fpr = []
        all_tpr = []

        for i, label in enumerate(self.labels):
            if y_true_bin[:, i].sum() == 0:
                # No positive samples for this class
                results.per_class_auc[label] = 0.0
                results.per_class_roc[label] = {"fpr": [], "tpr": [], "thresholds": []}
                continue

            fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = float(auc(fpr, tpr))

            results.per_class_roc[label] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }
            results.per_class_auc[label] = roc_auc
            all_fpr.append(fpr)
            all_tpr.append(tpr)

        # Macro AUC
        if results.per_class_auc:
            results.macro_auc = float(np.mean(list(results.per_class_auc.values())))

        # ---- Precision-Recall Curves ----
        for i, label in enumerate(self.labels):
            if y_true_bin[:, i].sum() == 0:
                results.per_class_avg_precision[label] = 0.0
                results.per_class_pr[label] = {"precision": [], "recall": []}
                continue

            pr_precision, pr_recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_scores[:, i]
            )
            avg_prec = float(average_precision_score(y_true_bin[:, i], y_scores[:, i]))

            results.per_class_pr[label] = {
                "precision": pr_precision.tolist(),
                "recall": pr_recall.tolist(),
            }
            results.per_class_avg_precision[label] = avg_prec

        # ---- Confidence Analysis ----
        max_confs = np.max(y_scores, axis=1)
        correct_mask = y_true == y_pred
        if correct_mask.sum() > 0:
            results.mean_confidence_correct = float(np.mean(max_confs[correct_mask]))
        if (~correct_mask).sum() > 0:
            results.mean_confidence_incorrect = float(np.mean(max_confs[~correct_mask]))

        # Per-class confidence distribution
        for i, label in enumerate(self.labels):
            mask = y_pred == i
            if mask.sum() > 0:
                results.confidence_distribution[label] = max_confs[mask].tolist()

        # ---- Feature Importance (gradient-based) ----
        if self.model is not None and X is not None:
            results.feature_importance = self._compute_feature_importance(X, y_true)

        results.evaluation_time_ms = (time.perf_counter() - t0) * 1000
        return results

    def _run_inference(
        self, X: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run batch inference through the DSTER model."""
        X = X.to(self.device)

        with torch.no_grad():
            logits, probs = self.model(X)

        y_pred = torch.argmax(probs, dim=-1).cpu().numpy()
        y_scores = probs.cpu().numpy()
        return y_pred, y_scores

    def _compute_feature_importance(
        self, X: torch.Tensor, y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute gradient-based feature importance.

        Uses the mean absolute gradient of the loss w.r.t. input features
        to estimate which keystroke features drive classification.
        """
        feature_names = [
            "press_duration",
            "iki_latency",
            "key_velocity",
            "flight_time",
        ]

        X_grad = X[:100].clone().detach().requires_grad_(True).to(self.device)
        y_sub = torch.tensor(y_true[:100], dtype=torch.long, device=self.device)

        self.model.train()  # Enable gradients
        logits, _ = self.model(X_grad)
        loss = F.cross_entropy(logits, y_sub)
        loss.backward()
        self.model.eval()

        # Mean absolute gradient per feature channel
        if X_grad.grad is not None:
            grad_importance = X_grad.grad.abs().mean(dim=(0, 1)).cpu().numpy()
            # Pad or truncate to match feature names
            n_features = min(len(feature_names), len(grad_importance))
            importance = {}
            total = grad_importance[:n_features].sum()
            if total > 0:
                for i in range(n_features):
                    importance[feature_names[i]] = float(
                        grad_importance[i] / total
                    )
            else:
                for i in range(n_features):
                    importance[feature_names[i]] = 1.0 / n_features
            return importance

        return {name: 1.0 / len(feature_names) for name in feature_names}

    def save_report(
        self,
        results: EvaluationResults,
        output_dir: str,
        filename: str = "evaluation_report.json",
    ) -> str:
        """
        Save evaluation results as a JSON report.

        Args:
            results: EvaluationResults to save.
            output_dir: Directory to write the report.
            filename: Report filename.

        Returns:
            Full path to the saved report.
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        report = results.to_dict()
        report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return filepath

    @staticmethod
    def generate_synthetic_data(
        num_samples: int = 1000,
        seq_len: int = 50,
        num_channels: int = 4,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate synthetic evaluation data for the DSTER model.

        Creates keystroke feature sequences with class-dependent
        statistical properties to simulate realistic evaluation data.

        Args:
            num_samples: Number of samples to generate.
            seq_len: Sequence length per sample.
            num_channels: Number of input channels.
            seed: Random seed.

        Returns:
            Tuple of (X, y_true) — input tensor and ground truth labels.
        """
        rng = np.random.RandomState(seed)

        # Class-dependent feature statistics (mean, std per channel)
        class_stats = {
            0: {"means": [0.8, 0.3, 0.9, 0.4], "stds": [0.15, 0.1, 0.12, 0.08]},  # angry
            1: {"means": [0.3, 0.7, 0.4, 0.6], "stds": [0.08, 0.08, 0.10, 0.07]},  # focused
            2: {"means": [0.5, 0.6, 0.5, 0.55], "stds": [0.10, 0.09, 0.11, 0.08]}, # thinking
            3: {"means": [0.5, 0.5, 0.5, 0.5], "stds": [0.10, 0.10, 0.10, 0.10]},  # neutral
            4: {"means": [0.4, 0.8, 0.6, 0.7], "stds": [0.09, 0.07, 0.08, 0.06]},  # joy
            5: {"means": [0.6, 0.2, 0.3, 0.5], "stds": [0.12, 0.12, 0.15, 0.10]},  # given_up
        }

        samples_per_class = num_samples // NUM_CLASSES
        X_list = []
        y_list = []

        for cls_idx in range(NUM_CLASSES):
            stats = class_stats[cls_idx]
            for _ in range(samples_per_class):
                sample = np.zeros((seq_len, num_channels))
                for ch in range(num_channels):
                    # Generate temporally correlated features
                    base = rng.normal(stats["means"][ch], stats["stds"][ch], seq_len)
                    # Add temporal smoothing
                    for t in range(1, seq_len):
                        base[t] = 0.7 * base[t] + 0.3 * base[t - 1]
                    sample[:, ch] = np.clip(base, 0, 1)

                X_list.append(sample)
                y_list.append(cls_idx)

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = np.array(y_list)

        # Shuffle
        perm = rng.permutation(len(y))
        X = X[perm]
        y = y[perm]

        return X, y


def compute_cross_validation_scores(
    model_class,
    model_kwargs: Dict,
    X: torch.Tensor,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    K-fold cross-validation for the DSTER model.

    Note: This creates new model instances per fold and evaluates
    without re-training (to assess model architecture stability).

    Args:
        model_class: The model class to instantiate.
        model_kwargs: Keyword arguments for model construction.
        X: Full input tensor.
        y: Full ground truth labels.
        n_folds: Number of folds.
        seed: Random seed for fold splitting.

    Returns:
        Dict with 'accuracies' and 'f1_scores' per fold.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_accuracies = []
    fold_f1s = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_test = X[test_idx]
        y_test = y[test_idx]

        model = model_class(**model_kwargs)
        model.eval()

        evaluator = DSTEREvaluator(model=model)
        results = evaluator.evaluate(X=X_test, y_true=y_test)

        fold_accuracies.append(results.accuracy)
        fold_f1s.append(results.macro_f1)

    return {
        "accuracies": fold_accuracies,
        "f1_scores": fold_f1s,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1": float(np.std(fold_f1s)),
    }
