"""
Project ECHO — SSPRA Behavioral Biometric Authenticator
=========================================================
Phase 5: Edge Optimization & Deployment

Implements Sigma-lognormal Stroke Profile Replay Authentication
(SSPRA) — a behavioral biometric system that authenticates users
based on their unique neuromotor signatures extracted from typing
and mouse movement patterns.

The key insight: each person's neuromotor command system produces
unique σ and μ distributions that are as distinctive as fingerprints
but continuously verifiable during gameplay.

Authentication is passive and continuous — no explicit login required.
The system maintains a running authentication score that tracks
how well the current user's biometric profile matches the enrolled
profile.

Reference:
    Plamondon, R. & Srihari, S.N. (2000). On-Line and Off-Line
    Handwriting Recognition. IEEE TPAMI 22(1), 63-84.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class BiometricProfile:
    """Enrolled user biometric profile."""
    user_id: str
    sigma_mean: float          # Mean σ across enrollment
    sigma_std: float           # Std of σ
    mu_mean: float             # Mean μ across enrollment
    mu_std: float              # Std of μ
    iki_mean: float            # Mean inter-keystroke interval
    iki_std: float             # Std of IKI
    stroke_count_mean: float   # Mean strokes per segment
    enrollment_samples: int    # Number of enrollment samples
    created_at: float = 0.0


@dataclass
class AuthenticationResult:
    """Result of a continuous authentication check."""
    authenticated: bool
    confidence: float          # [0, 1] authentication confidence
    match_score: float         # Mahalanobis-like distance score
    sigma_match: float         # σ profile match [0, 1]
    mu_match: float            # μ profile match [0, 1]
    iki_match: float           # IKI profile match [0, 1]
    anomaly_detected: bool     # True if behavior suddenly changed
    rolling_score: float       # Exponentially weighted running score


class SSPRAAuthenticator:
    """
    Continuous behavioral biometric authentication via neuromotor signatures.

    Workflow:
        1. Enrollment: Collect σ, μ, IKI samples during initial gameplay
        2. Authentication: Continuously compare live biometrics to profile
        3. Anomaly detection: Flag sudden behavioral shifts

    Parameters:
        enrollment_threshold: Minimum samples needed for enrollment.
        auth_threshold: Minimum confidence to pass authentication.
        anomaly_sensitivity: How sensitive anomaly detection is (lower = more sensitive).
        rolling_alpha: Exponential smoothing for running auth score.
    """

    def __init__(
        self,
        enrollment_threshold: int = 30,
        auth_threshold: float = 0.6,
        anomaly_sensitivity: float = 3.0,
        rolling_alpha: float = 0.15,
    ):
        self.enrollment_threshold = enrollment_threshold
        self.auth_threshold = auth_threshold
        self.anomaly_sensitivity = anomaly_sensitivity
        self.rolling_alpha = rolling_alpha

        self._profiles: Dict[str, BiometricProfile] = {}
        self._enrollment_buffer: Dict[str, Dict[str, list]] = {}
        self._rolling_scores: Dict[str, float] = {}
        self._recent_scores: Dict[str, deque] = {}

    def begin_enrollment(self, user_id: str) -> None:
        """Start collecting biometric samples for a new user."""
        self._enrollment_buffer[user_id] = {
            "sigma": [], "mu": [], "iki": [], "stroke_count": [],
        }

    def add_enrollment_sample(
        self,
        user_id: str,
        sigma: float,
        mu: float,
        iki_mean: float,
        stroke_count: int = 3,
    ) -> int:
        """
        Add a biometric sample during enrollment.

        Returns:
            Number of samples collected so far.
        """
        if user_id not in self._enrollment_buffer:
            self.begin_enrollment(user_id)

        buf = self._enrollment_buffer[user_id]
        buf["sigma"].append(sigma)
        buf["mu"].append(mu)
        buf["iki"].append(iki_mean)
        buf["stroke_count"].append(stroke_count)

        return len(buf["sigma"])

    def finalize_enrollment(self, user_id: str) -> Optional[BiometricProfile]:
        """
        Finalize enrollment and create a biometric profile.

        Returns:
            BiometricProfile if enough samples, None otherwise.
        """
        if user_id not in self._enrollment_buffer:
            return None

        buf = self._enrollment_buffer[user_id]
        n = len(buf["sigma"])

        if n < self.enrollment_threshold:
            return None

        profile = BiometricProfile(
            user_id=user_id,
            sigma_mean=float(np.mean(buf["sigma"])),
            sigma_std=float(np.std(buf["sigma"]) + 1e-6),
            mu_mean=float(np.mean(buf["mu"])),
            mu_std=float(np.std(buf["mu"]) + 1e-6),
            iki_mean=float(np.mean(buf["iki"])),
            iki_std=float(np.std(buf["iki"]) + 1e-6),
            stroke_count_mean=float(np.mean(buf["stroke_count"])),
            enrollment_samples=n,
        )

        self._profiles[user_id] = profile
        self._rolling_scores[user_id] = 0.5
        self._recent_scores[user_id] = deque(maxlen=20)

        # Clean up enrollment buffer
        del self._enrollment_buffer[user_id]

        return profile

    def authenticate(
        self,
        user_id: str,
        sigma: float,
        mu: float,
        iki_mean: float,
    ) -> AuthenticationResult:
        """
        Perform continuous authentication check against enrolled profile.

        Args:
            user_id: The enrolled user to verify against.
            sigma: Current σ from IDeLog extraction.
            mu: Current μ from IDeLog extraction.
            iki_mean: Current mean IKI.

        Returns:
            AuthenticationResult with match scores and anomaly detection.
        """
        if user_id not in self._profiles:
            return AuthenticationResult(
                authenticated=False, confidence=0.0,
                match_score=0.0, sigma_match=0.0,
                mu_match=0.0, iki_match=0.0,
                anomaly_detected=False, rolling_score=0.0,
            )

        profile = self._profiles[user_id]

        # Compute per-feature Gaussian match scores
        sigma_match = self._gaussian_match(
            sigma, profile.sigma_mean, profile.sigma_std
        )
        mu_match = self._gaussian_match(
            mu, profile.mu_mean, profile.mu_std
        )
        iki_match = self._gaussian_match(
            iki_mean, profile.iki_mean, profile.iki_std
        )

        # Weighted composite score
        match_score = 0.4 * sigma_match + 0.35 * mu_match + 0.25 * iki_match
        confidence = match_score

        # Update rolling score
        prev_rolling = self._rolling_scores.get(user_id, 0.5)
        rolling = (
            self.rolling_alpha * confidence
            + (1 - self.rolling_alpha) * prev_rolling
        )
        self._rolling_scores[user_id] = rolling

        # Anomaly detection: sudden drop in match score
        self._recent_scores.setdefault(
            user_id, deque(maxlen=20)
        ).append(confidence)
        anomaly = self._detect_anomaly(user_id, confidence)

        authenticated = rolling >= self.auth_threshold and not anomaly

        return AuthenticationResult(
            authenticated=authenticated,
            confidence=round(confidence, 4),
            match_score=round(match_score, 4),
            sigma_match=round(sigma_match, 4),
            mu_match=round(mu_match, 4),
            iki_match=round(iki_match, 4),
            anomaly_detected=anomaly,
            rolling_score=round(rolling, 4),
        )

    def get_profile(self, user_id: str) -> Optional[BiometricProfile]:
        """Get enrolled profile for a user."""
        return self._profiles.get(user_id)

    def is_enrolled(self, user_id: str) -> bool:
        """Check if a user has a finalized biometric profile."""
        return user_id in self._profiles

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_match(value: float, mean: float, std: float) -> float:
        """
        Compute Gaussian match score: how likely this value is
        under the profile's distribution. Returns [0, 1].
        """
        z = abs(value - mean) / max(std, 1e-6)
        return float(np.exp(-0.5 * z ** 2))


    def _detect_anomaly(self, user_id: str, current_score: float) -> bool:
        """
        Detect anomaly via z-score of recent authentication scores.
        """
        scores = self._recent_scores.get(user_id, deque())
        if len(scores) < 5:
            return False

        arr = np.array(list(scores))
        mean = float(np.mean(arr[:-1]))  # Exclude current
        std = float(np.std(arr[:-1])) + 1e-6

        z = abs(current_score - mean) / std
        return bool(z > self.anomaly_sensitivity)
