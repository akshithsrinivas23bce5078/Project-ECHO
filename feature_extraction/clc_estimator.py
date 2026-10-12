"""
Project ECHO — Cognitive Load Correlation (CLC) Estimator
==========================================================
Computes the CLC metric as the Spearman rank correlation between
the local complexity of generated content and corresponding
inter-keystroke intervals.

Mathematical Definition:
    CLC = ρ_s(complexity_ranks, iki_ranks)

Where:
    - complexity_ranks: rank of local text complexity within a sliding window
    - iki_ranks: rank of inter-keystroke interval within the same window
    - ρ_s: Spearman rank correlation coefficient

Interpretation:
    - CLC ∈ [0.35, 0.55] → genuine cognitive composition (nominal)
    - CLC < 0.35 → possible pasted / non-generative text
    - CLC > 0.55 → unusually high deliberation or unfamiliar content

Reference: The CLC metric captures the intuition that harder-to-compose
text naturally produces longer pauses, yielding a positive correlation.
"""

import numpy as np
from scipy.stats import spearmanr
from typing import List, Tuple, Optional


class CLCEstimator:
    """
    Computes the Cognitive Load Correlation from keystroke dynamics.
    """

    # Nominal CLC range for genuine text composition
    NOMINAL_MIN = 0.35
    NOMINAL_MAX = 0.55

    def __init__(self, window_size: int = 20):
        """
        Args:
            window_size: Number of digraph samples per CLC computation window.
        """
        self.window_size = window_size

    def compute_clc(
        self,
        iki_values: np.ndarray,
        complexity_scores: Optional[np.ndarray] = None,
    ) -> Tuple[float, bool]:
        """
        Compute the CLC for a given window of keystroke data.

        Args:
            iki_values: 1-D array of inter-keystroke intervals (ms).
            complexity_scores: 1-D array of local text complexity scores,
                               same length as iki_values. If None, a synthetic
                               proxy is derived from IKI variance.

        Returns:
            Tuple of (clc_value, is_nominal) where is_nominal indicates
            whether the CLC falls within [0.35, 0.55].
        """
        if len(iki_values) < self.window_size:
            # Insufficient data — return neutral value with noise flag
            return 0.0, False

        # Trim to the most recent window
        iki_window = iki_values[-self.window_size:]

        if complexity_scores is not None:
            cx_window = complexity_scores[-self.window_size:]
        else:
            cx_window = self._synthetic_complexity(iki_window)

        # Spearman rank correlation
        rho, p_value = spearmanr(cx_window, iki_window)

        # Handle NaN (e.g., constant arrays)
        if np.isnan(rho):
            return 0.0, False

        clc = float(rho)
        is_nominal = self.NOMINAL_MIN <= clc <= self.NOMINAL_MAX

        return clc, is_nominal

    def compute_rolling_clc(
        self,
        iki_values: np.ndarray,
        complexity_scores: Optional[np.ndarray] = None,
        step: int = 5,
    ) -> List[Tuple[float, bool]]:
        """
        Compute CLC over a rolling window across the full IKI series.

        Args:
            iki_values: Full 1-D array of IKIs.
            complexity_scores: Optional parallel complexity scores.
            step: Number of samples to advance per window.

        Returns:
            List of (clc_value, is_nominal) tuples.
        """
        results = []
        n = len(iki_values)

        for start in range(0, n - self.window_size + 1, step):
            end = start + self.window_size
            iki_chunk = iki_values[start:end]
            cx_chunk = (
                complexity_scores[start:end]
                if complexity_scores is not None
                else None
            )
            clc, nominal = self.compute_clc(iki_chunk, cx_chunk)
            results.append((clc, nominal))

        return results

    def normalize_clc_to_load(self, clc_raw: float) -> float:
        """
        Map raw CLC (expected ~0.0–1.0 Spearman rho) to a normalized
        cognitive load estimate in [0.0, 1.0].

        Higher CLC → higher cognitive load (user is thinking harder,
        typing slower on complex content).
        """
        # Clamp to [0, 1] range and apply sigmoid-like scaling
        clamped = max(0.0, min(1.0, clc_raw))
        # Linear mapping centered on nominal midpoint (0.45)
        load = (clamped - 0.0) / 1.0
        return round(load, 4)

    @staticmethod
    def _synthetic_complexity(iki_window: np.ndarray) -> np.ndarray:
        """
        Generate a synthetic complexity proxy when true text complexity
        scores are not available.

        Uses local variance of IKI in a small sliding sub-window as a
        proxy for compositional difficulty.
        """
        n = len(iki_window)
        sub_window = max(3, n // 5)
        complexity = np.zeros(n)

        for i in range(n):
            start = max(0, i - sub_window // 2)
            end = min(n, i + sub_window // 2 + 1)
            complexity[i] = np.std(iki_window[start:end])

        return complexity

    # ------------------------------------------------------------------
    # Neuromotor Load Fusion (Phase 1)
    # ------------------------------------------------------------------

    def compute_neuromotor_load(
        self,
        sigma_series: np.ndarray,
        mu_series: np.ndarray,
        sigma_baseline: float = 0.40,
        mu_baseline: float = -0.80,
    ) -> dict:
        """
        Compute a neuromotor cognitive load estimate from IDeLog-extracted
        σ and μ parameters.

        The neuromotor load is derived from the deviation of current
        σ (log response time) and μ (log time delay) from their
        baseline values. Elevated σ indicates neuromuscular fatigue,
        while elevated μ indicates increased CNS processing delay.

        Args:
            sigma_series: 1-D array of σ values from recent IDeLog extractions.
            mu_series: 1-D array of μ values from recent IDeLog extractions.
            sigma_baseline: Expected baseline σ for a relaxed user.
            mu_baseline: Expected baseline μ for a relaxed user.

        Returns:
            Dictionary with:
                neuromotor_load: float in [0.0, 1.0]
                sigma_deviation: mean deviation of σ from baseline
                mu_deviation: mean deviation of μ from baseline
                sample_count: number of samples used
        """
        if len(sigma_series) == 0 or len(mu_series) == 0:
            return {
                "neuromotor_load": 0.0,
                "sigma_deviation": 0.0,
                "mu_deviation": 0.0,
                "sample_count": 0,
            }

        # Compute mean deviations from baseline
        sigma_dev = float(np.mean(np.abs(sigma_series - sigma_baseline)))
        mu_dev = float(np.mean(np.abs(mu_series - mu_baseline)))

        # Normalize to [0, 1] range using sigmoid-like scaling
        # σ deviation of ~0.5 → load ~0.7; μ deviation of ~1.0 → load ~0.7
        sigma_load = 1.0 / (1.0 + np.exp(-5.0 * (sigma_dev - 0.3)))
        mu_load = 1.0 / (1.0 + np.exp(-3.0 * (mu_dev - 0.5)))

        # Weighted combination (σ is more indicative of fatigue)
        neuromotor_load = float(0.6 * sigma_load + 0.4 * mu_load)
        neuromotor_load = max(0.0, min(1.0, neuromotor_load))

        return {
            "neuromotor_load": round(neuromotor_load, 4),
            "sigma_deviation": round(sigma_dev, 4),
            "mu_deviation": round(mu_dev, 4),
            "sample_count": int(min(len(sigma_series), len(mu_series))),
        }

    def compute_unified_load(
        self,
        clc_raw: float,
        neuromotor_load: float,
        clc_weight: float = 0.5,
        neuromotor_weight: float = 0.5,
    ) -> float:
        """
        Fuse CLC-derived cognitive load with neuromotor-derived load
        into a single unified cognitive load score.

        Args:
            clc_raw: Raw CLC value from compute_clc().
            neuromotor_load: Neuromotor load from compute_neuromotor_load().
            clc_weight: Weight for CLC component (default 0.5).
            neuromotor_weight: Weight for neuromotor component (default 0.5).

        Returns:
            Unified cognitive load estimate in [0.0, 1.0].
        """
        clc_load = self.normalize_clc_to_load(clc_raw)

        total_weight = clc_weight + neuromotor_weight
        if total_weight < 1e-6:
            return 0.0

        unified = (
            clc_weight * clc_load + neuromotor_weight * neuromotor_load
        ) / total_weight

        return round(max(0.0, min(1.0, unified)), 4)

