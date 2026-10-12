"""
Project ECHO — Tiered SNR Validation Tests
=============================================
Validates the multi-tier SNR classification system for
Sigma-Lognormal velocity reconstructions (Phase 1).

Tests:
    1. All four SNR tiers classify correctly
    2. Discard behavior for sub-15dB data
    3. Confidence weight assignment per tier
    4. Edge cases (perfect reconstruction, no signal)
    5. Integration with SigmaLognormalModel
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from feature_extraction.snr_validator import (
    SNRValidator, SNRResult, SNRQuality, CONFIDENCE_WEIGHTS,
)
from feature_extraction.sigma_lognormal import SigmaLognormalModel


class TestSNRValidator:
    """Tests for the tiered SNR validation protocol."""

    def test_excellent_tier(self):
        """SNR > 25 dB → EXCELLENT quality, weight 1.0."""
        validator = SNRValidator()
        result = validator.validate_snr_value(30.0)

        assert result.quality == SNRQuality.EXCELLENT
        assert result.confidence_weight == 1.0
        assert result.should_discard is False

    def test_good_tier(self):
        """SNR in [20, 25) dB → GOOD quality, weight 0.7."""
        validator = SNRValidator()
        result = validator.validate_snr_value(22.0)

        assert result.quality == SNRQuality.GOOD
        assert result.confidence_weight == 0.7
        assert result.should_discard is False

    def test_acceptable_tier(self):
        """SNR in [15, 20) dB → ACCEPTABLE quality, weight 0.3."""
        validator = SNRValidator()
        result = validator.validate_snr_value(17.5)

        assert result.quality == SNRQuality.ACCEPTABLE
        assert result.confidence_weight == 0.3
        assert result.should_discard is False

    def test_unsatisfactory_tier(self):
        """SNR < 15 dB → UNSATISFACTORY, weight 0.0, should discard."""
        validator = SNRValidator()
        result = validator.validate_snr_value(10.0)

        assert result.quality == SNRQuality.UNSATISFACTORY
        assert result.confidence_weight == 0.0
        assert result.should_discard is True

    def test_boundary_at_15db(self):
        """Exactly 15 dB should be ACCEPTABLE (not UNSATISFACTORY)."""
        validator = SNRValidator()
        result = validator.validate_snr_value(15.0)

        assert result.quality == SNRQuality.ACCEPTABLE
        assert result.should_discard is False

    def test_boundary_at_20db(self):
        """Exactly 20 dB should be GOOD."""
        validator = SNRValidator()
        result = validator.validate_snr_value(20.0)

        assert result.quality == SNRQuality.GOOD

    def test_boundary_at_25db(self):
        """Exactly 25 dB should be EXCELLENT."""
        validator = SNRValidator()
        result = validator.validate_snr_value(25.0)

        assert result.quality == SNRQuality.EXCELLENT

    def test_perfect_reconstruction(self):
        """Identical signals → 100 dB (perfect), EXCELLENT."""
        validator = SNRValidator()
        signal = np.random.randn(100)
        result = validator.validate(signal, signal.copy())

        assert result.snr_db == 100.0
        assert result.quality == SNRQuality.EXCELLENT

    def test_zero_signal(self):
        """Zero signal with noise → 0 dB, UNSATISFACTORY."""
        validator = SNRValidator()
        original = np.zeros(100)
        reconstructed = np.random.randn(100)
        result = validator.validate(original, reconstructed)

        assert result.snr_db == 0.0
        assert result.quality == SNRQuality.UNSATISFACTORY

    def test_custom_thresholds(self):
        """Custom thresholds should override defaults."""
        validator = SNRValidator(
            threshold_unsatisfactory=10.0,
            threshold_good=18.0,
            threshold_excellent=22.0,
        )
        result = validator.validate_snr_value(11.0)
        assert result.quality == SNRQuality.ACCEPTABLE

        result = validator.validate_snr_value(19.0)
        assert result.quality == SNRQuality.GOOD

    def test_invalid_thresholds_raises(self):
        """Non-increasing thresholds should raise ValueError."""
        with pytest.raises(ValueError):
            SNRValidator(
                threshold_unsatisfactory=25.0,
                threshold_good=20.0,
                threshold_excellent=15.0,
            )

    def test_confidence_weights_complete(self):
        """All SNRQuality values should have a confidence weight."""
        for quality in SNRQuality:
            assert quality in CONFIDENCE_WEIGHTS

    def test_snr_computation_accuracy(self):
        """SNR computation should match expected value for known signals."""
        validator = SNRValidator()
        # Signal with known SNR: signal = 1.0, noise = 0.1 → ~20 dB
        original = np.ones(100)
        noise = np.random.RandomState(42).normal(0, 0.1, 100)
        reconstructed = original + noise

        snr = validator.compute_snr(original, reconstructed)
        # SNR ≈ 10 * log10(100 / sum(noise²)) — should be near 20 dB
        assert 15 < snr < 30  # Reasonable range


class TestSNRIntegrationWithSigmaLognormal:
    """Tests that SigmaLognormalModel integrates tiered SNR correctly."""

    def test_analyze_trajectory_includes_snr_fields(self, synthetic_trajectory):
        """analyze_trajectory() should return tiered SNR fields."""
        x, y, t = synthetic_trajectory
        model = SigmaLognormalModel(min_strokes=3)

        result = model.analyze_trajectory(np.array(x), np.array(y), np.array(t))

        assert "snr_quality" in result
        assert "snr_confidence_weight" in result
        assert "discarded" in result
        assert result["snr_quality"] in ["unsatisfactory", "acceptable", "good", "excellent"]
        assert 0.0 <= result["snr_confidence_weight"] <= 1.0

    def test_high_snr_not_discarded(self, synthetic_trajectory):
        """High-quality trajectory should not be discarded."""
        x, y, t = synthetic_trajectory
        model = SigmaLognormalModel(min_strokes=3)

        result = model.analyze_trajectory(np.array(x), np.array(y), np.array(t))

        if result["snr_db"] > 15:
            assert result["discarded"] is False

    def test_strokes_include_theta_fields(self, synthetic_trajectory):
        """Stroke dicts should include theta_start and theta_end."""
        x, y, t = synthetic_trajectory
        model = SigmaLognormalModel(min_strokes=3)

        result = model.analyze_trajectory(np.array(x), np.array(y), np.array(t))

        if result["strokes"]:
            for s in result["strokes"]:
                assert "theta_start" in s
                assert "theta_end" in s

    def test_empty_trajectory_returns_unsatisfactory(self):
        """Empty trajectory should return UNSATISFACTORY quality."""
        model = SigmaLognormalModel()
        result = model.analyze_trajectory(
            np.array([0.0]), np.array([0.0]), np.array([0.0])
        )

        assert result["snr_quality"] == "unsatisfactory"
        assert result["discarded"] is True
        assert result["snr_confidence_weight"] == 0.0
