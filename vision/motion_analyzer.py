"""
Project ECHO — Motion Analyzer
=================================
Computes motion energy from optical flow fields, detects
high-action vs. low-action game segments, and correlates
motion energy with cognitive load.

Uses Farneback dense optical flow for motion field estimation.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class MotionAnalyzer:
    """
    Analyzes motion patterns in game frame sequences.

    Computes:
        - Frame-level motion energy (mean optical flow magnitude)
        - Motion direction histograms
        - High/low action segment detection
        - Motion-cognitive load correlation
    """

    def __init__(
        self,
        high_action_threshold: float = 5.0,
        low_action_threshold: float = 1.0,
        flow_params: Optional[Dict] = None,
    ):
        """
        Args:
            high_action_threshold: Flow magnitude above this = high action.
            low_action_threshold: Flow magnitude below this = low action.
            flow_params: Farneback optical flow parameters.
        """
        self.high_action_threshold = high_action_threshold
        self.low_action_threshold = low_action_threshold
        self.flow_params = flow_params or {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0,
        }

    def compute_motion_energy(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> Dict:
        """
        Compute motion energy between two consecutive frames.

        Args:
            prev_frame: Previous BGR frame.
            curr_frame: Current BGR frame.

        Returns:
            Dict with motion metrics.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            self.flow_params["pyr_scale"],
            self.flow_params["levels"],
            self.flow_params["winsize"],
            self.flow_params["iterations"],
            self.flow_params["poly_n"],
            self.flow_params["poly_sigma"],
            self.flow_params["flags"],
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        # Motion energy metrics
        mean_magnitude = float(np.mean(magnitude))
        max_magnitude = float(np.max(magnitude))
        std_magnitude = float(np.std(magnitude))

        # Action classification
        if mean_magnitude > self.high_action_threshold:
            action_level = "high"
        elif mean_magnitude < self.low_action_threshold:
            action_level = "low"
        else:
            action_level = "medium"

        # Direction histogram (8 bins = 8 cardinal/ordinal directions)
        direction_hist = self._compute_direction_histogram(angle, magnitude)

        # Motion region: percentage of frame with significant motion
        motion_mask = magnitude > 1.0
        motion_coverage = float(np.sum(motion_mask) / motion_mask.size)

        return {
            "mean_magnitude": round(mean_magnitude, 4),
            "max_magnitude": round(max_magnitude, 4),
            "std_magnitude": round(std_magnitude, 4),
            "action_level": action_level,
            "direction_histogram": direction_hist.tolist(),
            "motion_coverage": round(motion_coverage, 4),
            "flow_field": flow,
            "magnitude_map": magnitude,
        }

    def analyze_sequence(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Analyze motion across a sequence of frames.

        Args:
            frames: List of BGR images.

        Returns:
            List of motion metric dicts (one per frame pair).
        """
        results = []
        for i in range(1, len(frames)):
            metrics = self.compute_motion_energy(frames[i - 1], frames[i])
            # Remove large arrays for sequence-level storage
            metrics.pop("flow_field", None)
            metrics.pop("magnitude_map", None)
            metrics["frame_index"] = i
            results.append(metrics)
        return results

    def detect_action_segments(
        self, motion_energies: List[float], min_segment_length: int = 5
    ) -> List[Dict]:
        """
        Detect contiguous high/low action segments.

        Args:
            motion_energies: List of mean motion magnitudes per frame.
            min_segment_length: Minimum frames for a segment.

        Returns:
            List of segment dicts with start, end, level, mean_energy.
        """
        segments = []
        if not motion_energies:
            return segments

        current_level = self._classify_energy(motion_energies[0])
        segment_start = 0

        for i in range(1, len(motion_energies)):
            level = self._classify_energy(motion_energies[i])
            if level != current_level:
                length = i - segment_start
                if length >= min_segment_length:
                    segment_energy = motion_energies[segment_start:i]
                    segments.append({
                        "start_frame": segment_start,
                        "end_frame": i,
                        "length": length,
                        "action_level": current_level,
                        "mean_energy": round(float(np.mean(segment_energy)), 4),
                    })
                current_level = level
                segment_start = i

        # Final segment
        length = len(motion_energies) - segment_start
        if length >= min_segment_length:
            segment_energy = motion_energies[segment_start:]
            segments.append({
                "start_frame": segment_start,
                "end_frame": len(motion_energies),
                "length": length,
                "action_level": current_level,
                "mean_energy": round(float(np.mean(segment_energy)), 4),
            })

        return segments

    def correlate_with_cognitive_load(
        self, motion_energies: np.ndarray, cognitive_loads: np.ndarray
    ) -> Dict:
        """
        Compute correlation between motion energy and cognitive load.

        Returns:
            Dict with pearson_r, spearman_rho, p_values.
        """
        from scipy.stats import pearsonr, spearmanr

        if len(motion_energies) < 3 or len(cognitive_loads) < 3:
            return {"pearson_r": 0.0, "spearman_rho": 0.0, "p_pearson": 1.0, "p_spearman": 1.0}

        # Align lengths
        n = min(len(motion_energies), len(cognitive_loads))
        me = np.array(motion_energies[:n], dtype=np.float64)
        cl = np.array(cognitive_loads[:n], dtype=np.float64)

        r, p_r = pearsonr(me, cl)
        rho, p_rho = spearmanr(me, cl)

        return {
            "pearson_r": round(float(r), 4),
            "spearman_rho": round(float(rho), 4),
            "p_pearson": round(float(p_r), 6),
            "p_spearman": round(float(p_rho), 6),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_direction_histogram(
        angle: np.ndarray, magnitude: np.ndarray, n_bins: int = 8
    ) -> np.ndarray:
        """Compute motion direction histogram weighted by magnitude."""
        bin_edges = np.linspace(0, 360, n_bins + 1)
        hist = np.zeros(n_bins, dtype=np.float64)

        for b in range(n_bins):
            mask = (angle >= bin_edges[b]) & (angle < bin_edges[b + 1])
            hist[b] = float(np.sum(magnitude[mask]))

        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    def _classify_energy(self, energy: float) -> str:
        if energy > self.high_action_threshold:
            return "high"
        elif energy < self.low_action_threshold:
            return "low"
        return "medium"
