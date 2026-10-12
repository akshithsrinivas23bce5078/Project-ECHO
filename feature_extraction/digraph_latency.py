"""
Project ECHO — Digraph Latency Extraction
==========================================
Extracts digraph latencies (inter-keystroke intervals between consecutive
key-pairs) from raw keystroke event data.

Digraph latency is the foundational metric for both CLC estimation
and DSTER feature engineering.
"""

import numpy as np
from typing import List, Tuple, Optional


class DigraphExtractor:
    """
    Extracts digraph latency features from raw keystroke timing data.

    A digraph is a consecutive pair of keystrokes. The latency is the
    time interval (in milliseconds) between the press of the first key
    and the press of the second key.
    """

    def __init__(self, outlier_threshold_ms: float = 2000.0):
        """
        Args:
            outlier_threshold_ms: Digraphs with latency above this threshold
                                  are considered pauses (not typing) and are
                                  filtered out.
        """
        self.outlier_threshold_ms = outlier_threshold_ms

    def extract_from_pairs(
        self, digraph_pairs: List[Tuple[str, float]]
    ) -> np.ndarray:
        """
        Extract latency values from pre-computed digraph pairs.

        Args:
            digraph_pairs: List of (pair_label, latency_ms) tuples
                           from KeystrokeListener.get_digraph_latencies().

        Returns:
            1-D NumPy array of filtered latency values (ms).
        """
        if not digraph_pairs:
            return np.array([], dtype=np.float64)

        latencies = np.array([lat for _, lat in digraph_pairs], dtype=np.float64)
        return self._filter_outliers(latencies)

    def extract_from_timestamps(
        self, press_timestamps: List[float]
    ) -> np.ndarray:
        """
        Compute digraph latencies directly from a sequence of
        key-press timestamps (in seconds).

        Args:
            press_timestamps: Monotonically increasing press times.

        Returns:
            1-D NumPy array of inter-keystroke intervals (ms).
        """
        if len(press_timestamps) < 2:
            return np.array([], dtype=np.float64)

        ts = np.array(press_timestamps, dtype=np.float64)
        iki = np.diff(ts) * 1000.0  # Convert seconds → ms
        return self._filter_outliers(iki)

    def compute_statistics(self, latencies: np.ndarray) -> dict:
        """
        Compute summary statistics over a latency array.

        Returns:
            Dictionary with mean, std, median, p25, p75, and count.
        """
        if len(latencies) == 0:
            return {
                "mean_ms": 0.0, "std_ms": 0.0, "median_ms": 0.0,
                "p25_ms": 0.0, "p75_ms": 0.0, "count": 0
            }
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "median_ms": float(np.median(latencies)),
            "p25_ms": float(np.percentile(latencies, 25)),
            "p75_ms": float(np.percentile(latencies, 75)),
            "count": int(len(latencies)),
        }

    def _filter_outliers(self, latencies: np.ndarray) -> np.ndarray:
        """Remove pause-level outliers and negative values."""
        mask = (latencies > 0) & (latencies <= self.outlier_threshold_ms)
        return latencies[mask]
