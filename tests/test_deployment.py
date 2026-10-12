"""
Project ECHO — Deployment Tests (Phase 5)
============================================
Tests for ONNX export, quantization, SSPRA auth, and edge runtime.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import torch

from deployment.quantization import ModelQuantizer, QuantizationResult
from deployment.sspra_authenticator import (
    SSPRAAuthenticator, BiometricProfile, AuthenticationResult,
)
from deployment.edge_runtime import (
    EdgeRuntime, RuntimeBackend, InferenceResult, RuntimeMetrics,
)
from dster.emotion_classifier import EmotionClassifier


@pytest.fixture
def small_classifier():
    """Create a small EmotionClassifier for testing."""
    return EmotionClassifier(
        num_channels=4, seq_len=50, embed_dim=32,
        num_heads=2, num_layers=1, num_classes=5,
    )


class TestModelQuantizer:
    """Tests for model quantization."""

    def test_dynamic_int8_quantization(self, small_classifier):
        quantizer = ModelQuantizer()
        q_model, result = quantizer.quantize_dynamic_int8(small_classifier)

        assert isinstance(result, QuantizationResult)
        assert result.method == "dynamic_int8"
        assert result.original_size_mb > 0
        assert result.compression_ratio >= 1.0

    def test_fp16_quantization(self, small_classifier):
        quantizer = ModelQuantizer()
        fp16_model, result = quantizer.quantize_fp16(small_classifier)

        assert result.method == "fp16"
        assert result.quantized_size_mb <= result.original_size_mb

    def test_benchmark_accuracy(self, small_classifier):
        quantizer = ModelQuantizer()
        q_model, _ = quantizer.quantize_dynamic_int8(small_classifier)

        result = quantizer.benchmark(
            small_classifier, q_model,
            input_shape=(1, 50, 4), n_runs=5,
        )

        assert result.accuracy_preserved > 0.8  # Should preserve most accuracy
        assert result.max_output_diff < 1.0  # Outputs shouldn't diverge wildly

    def test_model_size_computation(self, small_classifier):
        quantizer = ModelQuantizer()
        size = quantizer._model_size_mb(small_classifier)
        assert size > 0


class TestSSPRAAuthenticator:
    """Tests for behavioral biometric authentication."""

    def test_enrollment_workflow(self):
        auth = SSPRAAuthenticator(enrollment_threshold=10)
        auth.begin_enrollment("player1")

        # Add samples
        for i in range(15):
            n = auth.add_enrollment_sample(
                "player1", sigma=0.4 + np.random.normal(0, 0.02),
                mu=-0.8 + np.random.normal(0, 0.03),
                iki_mean=150 + np.random.normal(0, 10),
            )

        assert n == 15

        profile = auth.finalize_enrollment("player1")
        assert isinstance(profile, BiometricProfile)
        assert profile.enrollment_samples == 15
        assert auth.is_enrolled("player1")

    def test_authentication_matching_profile(self):
        auth = SSPRAAuthenticator(enrollment_threshold=10, auth_threshold=0.5)
        auth.begin_enrollment("player1")

        for _ in range(15):
            auth.add_enrollment_sample(
                "player1", sigma=0.40, mu=-0.80, iki_mean=150.0,
            )
        auth.finalize_enrollment("player1")

        # Authenticate with matching biometrics
        result = auth.authenticate("player1", sigma=0.40, mu=-0.80, iki_mean=150.0)
        assert isinstance(result, AuthenticationResult)
        assert result.confidence > 0.8
        assert result.sigma_match > 0.9

    def test_authentication_mismatched_profile(self):
        auth = SSPRAAuthenticator(enrollment_threshold=10, auth_threshold=0.7)
        auth.begin_enrollment("player1")

        for _ in range(15):
            auth.add_enrollment_sample("player1", sigma=0.40, mu=-0.80, iki_mean=150.0)
        auth.finalize_enrollment("player1")

        # Authenticate with very different biometrics
        result = auth.authenticate("player1", sigma=0.90, mu=-0.10, iki_mean=300.0)
        assert result.confidence < 0.5

    def test_unenrolled_user_fails(self):
        auth = SSPRAAuthenticator()
        result = auth.authenticate("unknown", sigma=0.4, mu=-0.8, iki_mean=150)
        assert result.authenticated is False
        assert result.confidence == 0.0

    def test_insufficient_enrollment_returns_none(self):
        auth = SSPRAAuthenticator(enrollment_threshold=30)
        auth.begin_enrollment("player1")

        for _ in range(10):
            auth.add_enrollment_sample("player1", sigma=0.4, mu=-0.8, iki_mean=150)

        profile = auth.finalize_enrollment("player1")
        assert profile is None

    def test_anomaly_detection(self):
        auth = SSPRAAuthenticator(
            enrollment_threshold=10, anomaly_sensitivity=2.0,
        )
        auth.begin_enrollment("player1")
        for _ in range(15):
            auth.add_enrollment_sample("player1", sigma=0.40, mu=-0.80, iki_mean=150.0)
        auth.finalize_enrollment("player1")

        # Build up stable history
        for _ in range(10):
            auth.authenticate("player1", sigma=0.40, mu=-0.80, iki_mean=150.0)

        # Sudden behavioral shift
        result = auth.authenticate("player1", sigma=0.90, mu=-0.10, iki_mean=400.0)
        assert result.anomaly_detected is True

    def test_rolling_score_smoothing(self):
        auth = SSPRAAuthenticator(enrollment_threshold=10, rolling_alpha=0.3)
        auth.begin_enrollment("player1")
        for _ in range(15):
            auth.add_enrollment_sample("player1", sigma=0.40, mu=-0.80, iki_mean=150.0)
        auth.finalize_enrollment("player1")

        scores = []
        for _ in range(10):
            result = auth.authenticate("player1", sigma=0.40, mu=-0.80, iki_mean=150.0)
            scores.append(result.rolling_score)

        # Rolling score should converge
        assert scores[-1] > scores[0]  # Should increase toward high confidence


class TestEdgeRuntime:
    """Tests for the unified edge inference runtime."""

    def test_pytorch_inference(self, small_classifier):
        runtime = EdgeRuntime(backend=RuntimeBackend.PYTORCH)
        runtime.load_pytorch(small_classifier)

        features = np.random.randn(1, 50, 4).astype(np.float32)
        result = runtime.infer(features)

        assert isinstance(result, InferenceResult)
        assert result.emotion_label in EdgeRuntime.DEFAULT_LABELS
        assert 0.0 <= result.emotion_confidence <= 1.0
        assert result.latency_ms > 0

    def test_metrics_tracking(self, small_classifier):
        runtime = EdgeRuntime(backend=RuntimeBackend.PYTORCH)
        runtime.load_pytorch(small_classifier)

        for _ in range(5):
            features = np.random.randn(1, 50, 4).astype(np.float32)
            runtime.infer(features)

        metrics = runtime.get_metrics()
        assert isinstance(metrics, RuntimeMetrics)
        assert metrics.total_inferences == 5
        assert metrics.mean_latency_ms > 0
        assert metrics.p99_latency_ms >= metrics.mean_latency_ms

    def test_warmup_clears_latencies(self, small_classifier):
        runtime = EdgeRuntime(backend=RuntimeBackend.PYTORCH)
        runtime.load_pytorch(small_classifier)

        runtime.warmup(input_shape=(1, 50, 4), n=3)
        metrics = runtime.get_metrics()
        assert metrics.total_inferences == 0  # Warmup cleared

    def test_no_model_raises(self):
        runtime = EdgeRuntime(backend=RuntimeBackend.PYTORCH)
        with pytest.raises(RuntimeError):
            runtime.infer(np.random.randn(1, 50, 4).astype(np.float32))

    def test_quantized_backend(self, small_classifier):
        quantizer = ModelQuantizer()
        q_model, _ = quantizer.quantize_dynamic_int8(small_classifier)

        runtime = EdgeRuntime(backend=RuntimeBackend.QUANTIZED)
        runtime.load_quantized(q_model)

        features = np.random.randn(1, 50, 4).astype(np.float32)
        result = runtime.infer(features)

        assert result.emotion_label in EdgeRuntime.DEFAULT_LABELS
        assert result.backend == "quantized"
