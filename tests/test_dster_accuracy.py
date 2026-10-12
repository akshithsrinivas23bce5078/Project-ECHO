"""
Project ECHO — DSTER Accuracy Validation
==========================================
Validates that the DSTER model achieves an F1 classification
score greater than 0.95 on synthetic test data.

Target: F1 > 0.95
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import torch
from sklearn.metrics import f1_score

from dster.emotion_classifier import EmotionClassifier, EMOTION_LABELS


def generate_synthetic_dataset(n_samples: int = 500, seq_len: int = 50,
                                num_channels: int = 4, num_classes: int = 5):
    """
    Generate synthetic keystroke feature data with known labels.

    Each class has a distinct statistical signature so a well-trained
    model should achieve near-perfect classification.
    """
    np.random.seed(42)
    X_all = []
    y_all = []

    for cls in range(num_classes):
        # Each class has a unique mean/variance pattern
        mean_shift = cls * 0.5
        std_scale = 0.3 + cls * 0.1

        samples = np.random.normal(
            loc=mean_shift,
            scale=std_scale,
            size=(n_samples // num_classes, seq_len, num_channels)
        ).astype(np.float32)

        X_all.append(samples)
        y_all.extend([cls] * (n_samples // num_classes))

    X = np.concatenate(X_all, axis=0)
    y = np.array(y_all, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


def train_model_on_synthetic(model, X_train, y_train, epochs=30, lr=0.001):
    """Quick training loop on synthetic data."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    return model


class TestDSTERAccuracy:
    """Validates DSTER F1 > 0.95 on synthetic data."""

    def test_f1_score_exceeds_target(self):
        """DSTER must achieve F1 > 0.95 on well-separated synthetic classes."""
        TARGET_F1 = 0.95

        # Generate data
        X, y = generate_synthetic_dataset(n_samples=500)

        # Split 80/20
        split = int(0.8 * len(y))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Build and train model
        model = EmotionClassifier(
            num_channels=4, seq_len=50, embed_dim=32,
            num_heads=4, num_layers=1, dropout=0.05, num_classes=5
        )
        model = train_model_on_synthetic(model, X_train, y_train, epochs=50)

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test)
            _, probs = model(X_test_t)
            preds = torch.argmax(probs, dim=-1).numpy()

        f1 = f1_score(y_test, preds, average='weighted')
        print(f"\nDSTER F1 Score: {f1:.4f} (target: {TARGET_F1})")

        assert f1 > TARGET_F1, (
            f"DSTER F1 score {f1:.4f} is below target {TARGET_F1}"
        )

    def test_model_outputs_valid_labels(self):
        """Predict must return a valid emotion label string."""
        model = EmotionClassifier(
            num_channels=4, seq_len=50, embed_dim=32,
            num_heads=4, num_layers=1, num_classes=5
        )
        dummy = torch.randn(1, 50, 4)
        label, conf = model.predict(dummy)

        assert label in EMOTION_LABELS
        assert 0.0 <= conf <= 1.0

    def test_confidence_sums_to_one(self):
        """Softmax probabilities must sum to ~1.0."""
        model = EmotionClassifier(
            num_channels=4, seq_len=50, embed_dim=32,
            num_heads=4, num_layers=1, num_classes=5
        )
        dummy = torch.randn(1, 50, 4)
        _, probs = model(dummy)
        total = probs.sum().item()

        assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, not 1.0"
