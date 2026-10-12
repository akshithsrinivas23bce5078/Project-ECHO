"""
Project ECHO — Gaussian Range Encoding (GRE) Tests
=====================================================
Validates the GRE layer and its integration with the DSTER
dual-stream architecture (Phase 1).

Tests:
    1. GRE output shape matches expected dimensions
    2. DSTER forward pass with GRE produces valid embeddings
    3. Gradient flow through GRE layer
    4. Learnable parameters update during backward pass
    5. EmotionClassifier end-to-end with GRE
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import torch.nn as nn

from dster.gaussian_range_encoding import GaussianRangeEncoder
from dster.dster_model import DSTERModel
from dster.emotion_classifier import EmotionClassifier


class TestGaussianRangeEncoder:
    """Tests for the GRE layer."""

    def test_output_shape(self):
        """GRE output should be (*, embed_dim)."""
        gre = GaussianRangeEncoder(num_gaussians=16, embed_dim=64)
        x = torch.randn(32, 50, 1)  # (batch, seq, 1)
        out = gre(x)

        assert out.shape == (32, 50, 64)

    def test_single_value_shape(self):
        """GRE should handle single scalar input."""
        gre = GaussianRangeEncoder(num_gaussians=8, embed_dim=32)
        x = torch.tensor([[[1.5]]])  # (1, 1, 1)
        out = gre(x)

        assert out.shape == (1, 1, 32)

    def test_batch_processing(self):
        """GRE should process batches correctly."""
        gre = GaussianRangeEncoder(num_gaussians=16, embed_dim=64)
        x = torch.randn(8, 100, 1)
        out = gre(x)

        assert out.shape == (8, 100, 64)

    def test_learnable_parameters(self):
        """mu and sigma should be learnable Parameters by default."""
        gre = GaussianRangeEncoder(num_gaussians=16, embed_dim=64, learnable=True)

        assert isinstance(gre.mu, nn.Parameter)
        assert isinstance(gre.sigma, nn.Parameter)
        assert gre.mu.requires_grad is True
        assert gre.sigma.requires_grad is True

    def test_non_learnable_parameters(self):
        """When learnable=False, mu and sigma should be buffers."""
        gre = GaussianRangeEncoder(num_gaussians=16, embed_dim=64, learnable=False)

        assert not isinstance(gre.mu, nn.Parameter)
        assert not isinstance(gre.sigma, nn.Parameter)

    def test_gradient_flow(self):
        """Gradients should flow through the GRE layer."""
        gre = GaussianRangeEncoder(num_gaussians=16, embed_dim=64)
        x = torch.randn(4, 50, 1, requires_grad=True)

        out = gre(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert gre.mu.grad is not None
        assert gre.sigma.grad is not None

    def test_mu_init_range(self):
        """Gaussian centers should span the specified range."""
        gre = GaussianRangeEncoder(
            num_gaussians=10, embed_dim=32, mu_init_range=(0.0, 10.0)
        )
        assert gre.mu.data[0].item() == pytest.approx(0.0, abs=0.01)
        assert gre.mu.data[-1].item() == pytest.approx(10.0, abs=0.01)

    def test_extra_repr(self):
        """extra_repr should return informative string."""
        gre = GaussianRangeEncoder(num_gaussians=16, embed_dim=64)
        repr_str = gre.extra_repr()

        assert "num_gaussians=16" in repr_str
        assert "embed_dim=64" in repr_str


class TestDSTERWithGRE:
    """Tests for DSTER model with GRE integration."""

    def test_forward_pass_shape(self):
        """DSTER with GRE should produce correct output shape."""
        model = DSTERModel(
            num_channels=4, seq_len=50, embed_dim=64,
            num_heads=4, num_layers=2, num_gaussians=16,
        )
        x = torch.randn(8, 50, 4)
        out = model(x)

        assert out.shape == (8, 64)

    def test_single_sample(self):
        """DSTER should handle single-sample batch."""
        model = DSTERModel(
            num_channels=4, seq_len=50, embed_dim=64,
            num_heads=4, num_layers=1, num_gaussians=8,
        )
        x = torch.randn(1, 50, 4)
        out = model(x)

        assert out.shape == (1, 64)

    def test_gradient_flow_through_dster(self):
        """Gradients should flow through the full DSTER → GRE pipeline."""
        model = DSTERModel(
            num_channels=4, seq_len=50, embed_dim=64,
            num_heads=4, num_layers=1, num_gaussians=8,
        )
        x = torch.randn(4, 50, 4)

        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check that GRE parameters received gradients
        gre_temporal = model.stream_temporal.input_projection
        gre_channel = model.stream_channel.input_projection

        assert gre_temporal.mu.grad is not None
        assert gre_channel.mu.grad is not None


class TestEmotionClassifierWithGRE:
    """Tests for EmotionClassifier end-to-end with GRE."""

    def test_classify_output_shape(self):
        """EmotionClassifier should produce (batch, num_classes) logits."""
        classifier = EmotionClassifier(
            num_channels=4, seq_len=50, embed_dim=64,
            num_heads=4, num_layers=1, num_classes=5,
        )
        x = torch.randn(4, 50, 4)
        logits, confidence = classifier(x)

        assert logits.shape == (4, 5)
        assert confidence.shape == (4, 5)

    def test_predict_returns_label(self):
        """predict() should return (label, confidence) tuple."""
        classifier = EmotionClassifier(
            num_channels=4, seq_len=50, embed_dim=64,
            num_heads=4, num_layers=1, num_classes=5,
        )
        x = torch.randn(1, 50, 4)
        label, conf = classifier.predict(x)

        assert isinstance(label, str)
        assert label in ["frustrated", "focused", "fatigued", "neutral", "engaged"]
        assert 0.0 <= conf <= 1.0

    def test_predict_batch(self):
        """predict_batch() should return list of (label, conf) tuples."""
        classifier = EmotionClassifier(
            num_channels=4, seq_len=50, embed_dim=64,
            num_heads=4, num_layers=1, num_classes=5,
        )
        x = torch.randn(8, 50, 4)
        results = classifier.predict_batch(x)

        assert len(results) == 8
        for label, conf in results:
            assert isinstance(label, str)
            assert 0.0 <= conf <= 1.0
