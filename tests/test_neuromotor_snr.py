"""
Project ECHO — Neuromotor SNR Validation
==========================================
Validates that the Sigma-Lognormal model achieves a
Signal-to-Noise Ratio strictly greater than 24 dB
on representative mouse trajectory data.

Target: SNR > 24 dB
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from feature_extraction.sigma_lognormal import SigmaLognormalModel


class TestNeuromotorSNR:
    """Validates Sigma-Lognormal SNR > 24 dB."""

    def test_snr_exceeds_target_on_smooth_trajectory(self, synthetic_trajectory):
        """
        A smooth, well-formed trajectory should reconstruct
        with SNR > 24 dB.
        """
        x, y, t = synthetic_trajectory
        model = SigmaLognormalModel(min_strokes=3)

        result = model.analyze_trajectory(
            np.array(x), np.array(y), np.array(t)
        )

        print(f"\nSNR: {result['snr_db']} dB (target: > 24.0 dB)")
        print(f"Strokes extracted: {result['stroke_count']}")

        assert result["snr_db"] > 24.0, (
            f"SNR {result['snr_db']} dB is below target 24.0 dB"
        )
        assert result["snr_pass"] is True

    def test_stroke_count_is_reasonable(self, synthetic_trajectory):
        """Extracted stroke count should be >= min_strokes."""
        x, y, t = synthetic_trajectory
        model = SigmaLognormalModel(min_strokes=3)

        result = model.analyze_trajectory(
            np.array(x), np.array(y), np.array(t)
        )

        assert result["stroke_count"] >= 3, (
            f"Only {result['stroke_count']} strokes extracted (min: 3)"
        )

    def test_velocity_profile_not_empty(self, synthetic_trajectory):
        """Velocity profile computation should produce output."""
        x, y, t = synthetic_trajectory
        model = SigmaLognormalModel()

        times, velocities = model.compute_velocity_profile(
            np.array(x), np.array(y), np.array(t)
        )

        assert len(times) > 0
        assert len(velocities) > 0
        assert len(times) == len(velocities)

    def test_degenerate_trajectory_handled(self):
        """Single-point or empty trajectory should not crash."""
        model = SigmaLognormalModel()

        result = model.analyze_trajectory(
            np.array([0]), np.array([0]), np.array([0.0])
        )

        assert result["stroke_count"] == 0
        assert result["snr_pass"] is False
