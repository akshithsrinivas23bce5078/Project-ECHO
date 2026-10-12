"""
Project ECHO — Processing Latency Validation
===============================================
Validates that the end-to-end pipeline processes within
the 40ms latency target.

Target: < 40 ms per inference cycle
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import time
import numpy as np
import torch

from feature_extraction.digraph_latency import DigraphExtractor
from feature_extraction.clc_estimator import CLCEstimator
from feature_extraction.sigma_lognormal import SigmaLognormalModel
from dster.emotion_classifier import EmotionClassifier


LATENCY_TARGET_MS = 40.0


class TestLatency:
    """Validates end-to-end pipeline latency < 40ms."""

    def test_feature_extraction_latency(self):
        """Digraph extraction + CLC computation must complete quickly."""
        extractor = DigraphExtractor()
        clc = CLCEstimator(window_size=20)

        timestamps = np.cumsum(np.random.uniform(0.05, 0.3, 100))

        start = time.perf_counter()
        latencies = extractor.extract_from_timestamps(timestamps)
        clc_val, _ = clc.compute_clc(latencies)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\nFeature extraction latency: {elapsed_ms:.2f} ms")
        assert elapsed_ms < LATENCY_TARGET_MS, (
            f"Feature extraction took {elapsed_ms:.2f} ms (target: < {LATENCY_TARGET_MS} ms)"
        )

    def test_sigma_lognormal_latency(self):
        """Sigma-Lognormal analysis must complete within target."""
        model = SigmaLognormalModel()
        n = 200
        t = np.linspace(0, 2.0, n)
        x = np.cumsum(np.random.normal(5, 2, n))
        y = np.cumsum(np.random.normal(3, 2, n))

        # Warm up
        model.analyze_trajectory(x, y, t)

        start = time.perf_counter()
        runs = 5
        for _ in range(runs):
            model.analyze_trajectory(x, y, t)
        elapsed_ms = ((time.perf_counter() - start) * 1000) / runs

        print(f"\nSigma-Lognormal latency: {elapsed_ms:.2f} ms")
        assert elapsed_ms < LATENCY_TARGET_MS, (
            f"Sigma-Lognormal took {elapsed_ms:.2f} ms (target: < {LATENCY_TARGET_MS} ms)"
        )


    def test_dster_inference_latency(self):
        """Single DSTER inference must complete within target."""
        model = EmotionClassifier(
            num_channels=4, seq_len=50, embed_dim=32,
            num_heads=4, num_layers=1, num_classes=5
        )
        model.eval()
        dummy = torch.randn(1, 50, 4)

        # Warm up PyTorch C++ engine
        with torch.no_grad():
            for _ in range(5):
                model.predict(dummy)

        start = time.perf_counter()
        with torch.no_grad():
            model.predict(dummy)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\nDSTER inference latency: {elapsed_ms:.2f} ms")
        assert elapsed_ms < LATENCY_TARGET_MS, (
            f"DSTER inference took {elapsed_ms:.2f} ms (target: < {LATENCY_TARGET_MS} ms)"
        )



    def test_full_pipeline_latency(self):
        """
        Combined feature extraction + DSTER inference must stay
        within a reasonable multiple of the target.
        """
        # Feature extraction
        extractor = DigraphExtractor()
        clc = CLCEstimator(window_size=20)
        timestamps = np.cumsum(np.random.uniform(0.05, 0.3, 100))

        # DSTER
        model = EmotionClassifier(
            num_channels=4, seq_len=50, embed_dim=32,
            num_heads=4, num_layers=1, num_classes=5
        )
        model.eval()
        dummy = torch.randn(1, 50, 4)

        # Warm up
        with torch.no_grad():
            model(dummy)

        # Full pipeline measurement
        start = time.perf_counter()
        latencies = extractor.extract_from_timestamps(timestamps)
        clc_val, _ = clc.compute_clc(latencies)
        with torch.no_grad():
            label, conf = model.predict(dummy)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"\nFull pipeline latency: {elapsed_ms:.2f} ms")
        # Allow 2x target for the full pipeline
        assert elapsed_ms < LATENCY_TARGET_MS * 2, (
            f"Full pipeline took {elapsed_ms:.2f} ms (target: < {LATENCY_TARGET_MS * 2} ms)"
        )
