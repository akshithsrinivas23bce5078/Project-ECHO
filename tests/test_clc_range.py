"""
Project ECHO — CLC Range Validation
======================================
Validates that the CLC metric stays within [0.35, 0.55] for
tasks requiring genuine cognitive composition, and falls outside
that range for non-generative input (e.g., pasted text).

Target: CLC ∈ [0.35, 0.55] for genuine composition
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from feature_extraction.clc_estimator import CLCEstimator


class TestCLCRange:
    """Validates CLC metric behavior across input types."""

    def test_genuine_composition_is_nominal(self, synthetic_iki_nominal):
        """
        Genuine text composition (correlated complexity and IKI)
        should produce CLC in [0.35, 0.55].
        """
        iki, complexity = synthetic_iki_nominal
        estimator = CLCEstimator(window_size=20)

        clc, is_nominal = estimator.compute_clc(iki, complexity)

        print(f"\nGenuine composition CLC: {clc:.4f} (nominal: {is_nominal})")

        assert is_nominal, (
            f"CLC {clc:.4f} is outside nominal range [0.35, 0.55] "
            f"for genuine composition data"
        )
        assert 0.35 <= clc <= 0.55

    def test_pasted_text_is_not_nominal(self, synthetic_iki_pasted):
        """
        Pasted/non-generative text (uniform timing, random complexity)
        should produce CLC outside [0.35, 0.55].
        """
        iki, complexity = synthetic_iki_pasted
        estimator = CLCEstimator(window_size=20)

        clc, is_nominal = estimator.compute_clc(iki, complexity)

        print(f"\nPasted text CLC: {clc:.4f} (nominal: {is_nominal})")

        assert not is_nominal, (
            f"CLC {clc:.4f} falsely classified as nominal for pasted text"
        )

    def test_rolling_clc_stability(self, synthetic_iki_nominal):
        """
        Rolling CLC over genuine composition should be mostly nominal.
        At least 70% of windows should be in range.
        """
        iki, complexity = synthetic_iki_nominal
        estimator = CLCEstimator(window_size=20)

        results = estimator.compute_rolling_clc(iki, complexity, step=5)

        if len(results) == 0:
            pytest.skip("Not enough data for rolling CLC")

        nominal_count = sum(1 for _, is_nom in results if is_nom)
        ratio = nominal_count / len(results)

        print(f"\nRolling CLC: {nominal_count}/{len(results)} nominal ({ratio:.1%})")

        assert ratio >= 0.7, (
            f"Only {ratio:.1%} of rolling CLC windows are nominal (target: ≥ 70%)"
        )

    def test_insufficient_data_returns_zero(self):
        """CLC with insufficient data should return 0.0 and not nominal."""
        estimator = CLCEstimator(window_size=20)
        iki = np.array([100.0, 150.0])  # Only 2 samples, need 20

        clc, is_nominal = estimator.compute_clc(iki)

        assert clc == 0.0
        assert is_nominal is False

    def test_normalization_bounds(self):
        """Normalized cognitive load should be in [0, 1]."""
        estimator = CLCEstimator()

        for clc_raw in [-0.5, 0.0, 0.45, 1.0, 1.5]:
            load = estimator.normalize_clc_to_load(clc_raw)
            assert 0.0 <= load <= 1.0, (
                f"normalize_clc_to_load({clc_raw}) = {load}, out of [0, 1]"
            )
