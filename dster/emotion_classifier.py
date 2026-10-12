"""
Project ECHO — Emotion Classifier
===================================
Final classification head that maps the fused DSTER embedding
to an emotion label with a confidence score.

Output labels: frustrated, focused, fatigued, neutral, engaged

The classifier uses softmax probabilities as confidence scores.
If the maximum confidence is below the threshold (0.6), the
downstream adaptation layer will ignore the emotion signal entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from dster.dster_model import DSTERModel


# Default label set matching echo_config.yaml
EMOTION_LABELS = ["frustrated", "focused", "fatigued", "neutral", "engaged"]


class EmotionClassifier(nn.Module):
    """
    Full DSTER pipeline: keystroke features → emotion label + confidence.
    """

    def __init__(
        self,
        num_channels: int = 4,
        seq_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 5,
        labels: Optional[List[str]] = None,
    ):
        super().__init__()
        self.labels = labels or EMOTION_LABELS[:num_classes]
        self.num_classes = num_classes

        # DSTER backbone
        self.dster = DSTERModel(
            num_channels=num_channels,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.dster.output_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, num_channels) — raw keystroke features.

        Returns:
            logits: (batch, num_classes) — raw classification logits.
            confidence: (batch, num_classes) — softmax probabilities.
        """
        embedding = self.dster(x)
        logits = self.classifier(embedding)
        confidence = F.softmax(logits, dim=-1)
        return logits, confidence

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[str, float]:
        """
        Single-sample inference.

        Args:
            x: (1, seq_len, num_channels) or (seq_len, num_channels).

        Returns:
            Tuple of (emotion_label, confidence_score).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        self.eval()
        _, confidence = self.forward(x)
        probs = confidence.squeeze(0)

        max_idx = torch.argmax(probs).item()
        max_conf = probs[max_idx].item()
        label = self.labels[max_idx]

        return label, round(max_conf, 4)

    @torch.no_grad()
    def predict_batch(self, x: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Batch inference.

        Args:
            x: (batch, seq_len, num_channels).

        Returns:
            List of (emotion_label, confidence_score) per sample.
        """
        self.eval()
        _, confidence = self.forward(x)
        max_confs, max_indices = torch.max(confidence, dim=-1)

        results = []
        for idx, conf in zip(max_indices.tolist(), max_confs.tolist()):
            results.append((self.labels[idx], round(conf, 4)))

        return results
