"""
Project ECHO — Tiered SNR Validation Protocol
================================================
Implements a multi-tier Signal-to-Noise Ratio classification system
for Sigma-Lognormal velocity profile reconstructions.

The tiered approach replaces the previous binary pass/fail gate
(SNR > 24 dB) with a nuanced quality assessment that propagates
confidence weights to downstream estimator modules.

Tier Definitions (per empirical neuromotor literature):
    < 15 dB   → UNSATISFACTORY  — non-ballistic / jitter; DISCARD
    15–20 dB  → ACCEPTABLE      — usable with low confidence (0.3)
    20–25 dB  → GOOD            — standard quality (0.7)
    > 25 dB   → EXCELLENT       — pristine neuromotor data (1.0)

The confidence weight allows the POMDP / Active Inference layer to
assign higher belief precision to pristine neuromotor observations,
stabilizing the overall cognitive state estimation.

Reference:
    Plamondon, R. & Djioua, M. (2006). A multi-level representation
    paradigm for handwriting stroke generation.
"""

import enum
import numpy as np
from typing import NamedTuple


class SNRQuality(enum.Enum):
    """Tiered quality classification for velocity reconstructions."""
    UNSATISFACTORY = "unsatisfactory"
    ACCEPTABLE = "acceptable"
    GOOD = "good"
    EXCELLENT = "excellent"


class SNRResult(NamedTuple):
    """Result of a tiered SNR validation."""
    snr_db: float
    quality: SNRQuality
    confidence_weight: float
    should_discard: bool


# ---------------------------------------------------------------
# Default Thresholds (overridable via echo_config.yaml)
# ---------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "unsatisfactory": 15.0,
    "good": 20.0,
    "excellent": 25.0,
}

# Confidence weights per tier
CONFIDENCE_WEIGHTS = {
    SNRQuality.UNSATISFACTORY: 0.0,
    SNRQuality.ACCEPTABLE: 0.3,
    SNRQuality.GOOD: 0.7,
    SNRQuality.EXCELLENT: 1.0,
}


class SNRValidator:
    """
    Validates and classifies Sigma-Lognormal reconstruction quality
    using a multi-tier SNR thresholding protocol.

    Usage:
        validator = SNRValidator()
        result = validator.validate(original_velocity, reconstructed_velocity)
        if result.should_discard:
            # Skip this trajectory segment
            ...
        else:
            # Use result.confidence_weight for downstream weighting
            ...
    """

    def __init__(
        self,
        threshold_unsatisfactory: float = DEFAULT_THRESHOLDS["unsatisfactory"],
        threshold_good: float = DEFAULT_THRESHOLDS["good"],
        threshold_excellent: float = DEFAULT_THRESHOLDS["excellent"],
    ):
        """
        Args:
            threshold_unsatisfactory: SNR below this (dB) → UNSATISFACTORY.
            threshold_good: SNR at or above this (dB) → GOOD.
            threshold_excellent: SNR at or above this (dB) → EXCELLENT.
        """
        if not (threshold_unsatisfactory < threshold_good < threshold_excellent):
            raise ValueError(
                f"Thresholds must be strictly increasing: "
                f"{threshold_unsatisfactory} < {threshold_good} < {threshold_excellent}"
            )
        self._thresh_unsatisfactory = threshold_unsatisfactory
        self._thresh_good = threshold_good
        self._thresh_excellent = threshold_excellent

    def compute_snr(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> float:
        """
        Compute Signal-to-Noise Ratio (dB) between original and
        reconstructed velocity profiles.

        Args:
            original: 1-D array of original velocity values.
            reconstructed: 1-D array of reconstructed velocity values.

        Returns:
            SNR in decibels. Returns 100.0 for perfect reconstruction.
        """
        signal_power = np.sum(original ** 2)
        noise_power = np.sum((original - reconstructed) ** 2)

        if noise_power < 1e-12:
            return 100.0  # Perfect reconstruction

        if signal_power < 1e-12:
            return 0.0  # No signal

        snr = 10.0 * np.log10(signal_power / noise_power)
        return float(snr)

    def classify(self, snr_db: float) -> SNRQuality:
        """
        Classify an SNR value into a quality tier.

        Args:
            snr_db: Signal-to-noise ratio in decibels.

        Returns:
            SNRQuality enum value.
        """
        if snr_db < self._thresh_unsatisfactory:
            return SNRQuality.UNSATISFACTORY
        elif snr_db < self._thresh_good:
            return SNRQuality.ACCEPTABLE
        elif snr_db < self._thresh_excellent:
            return SNRQuality.GOOD
        else:
            return SNRQuality.EXCELLENT

    def validate(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> SNRResult:
        """
        Full validation pipeline: compute SNR → classify → assign weight.

        Args:
            original: 1-D array of original velocity values.
            reconstructed: 1-D array of reconstructed velocity values.

        Returns:
            SNRResult with snr_db, quality tier, confidence weight,
            and discard flag.
        """
        snr_db = self.compute_snr(original, reconstructed)
        quality = self.classify(snr_db)
        weight = CONFIDENCE_WEIGHTS[quality]
        should_discard = (quality == SNRQuality.UNSATISFACTORY)

        return SNRResult(
            snr_db=round(snr_db, 2),
            quality=quality,
            confidence_weight=weight,
            should_discard=should_discard,
        )

    def validate_snr_value(self, snr_db: float) -> SNRResult:
        """
        Classify a pre-computed SNR value (when raw signals aren't available).

        Args:
            snr_db: Pre-computed SNR in decibels.

        Returns:
            SNRResult with classification and confidence weight.
        """
        quality = self.classify(snr_db)
        weight = CONFIDENCE_WEIGHTS[quality]
        should_discard = (quality == SNRQuality.UNSATISFACTORY)

        return SNRResult(
            snr_db=round(snr_db, 2),
            quality=quality,
            confidence_weight=weight,
            should_discard=should_discard,
        )
