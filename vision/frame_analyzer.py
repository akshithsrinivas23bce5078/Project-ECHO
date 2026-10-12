"""
Project ECHO — Frame Analyzer
================================
OpenCV-based game frame analysis for visual feature extraction.

Extracts visual complexity metrics, optical flow, saliency,
and scene change detection from game frames to complement
keyboard/mouse behavioral biometrics.

Features extracted per frame:
    - Visual complexity: edge density, color entropy, spatial frequency
    - Saliency map: spectral residual or fine-grained saliency
    - Scene change score: histogram correlation with previous frame
    - Optical flow: motion field magnitude and direction

Reference:
    Hou, X. & Zhang, L. (2007). Saliency Detection: A Spectral
    Residual Approach. CVPR.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple


class FrameAnalyzer:
    """
    Extracts visual features from game frames using OpenCV.

    Processes individual frames or frame sequences to produce
    metrics that correlate with cognitive load and attention.
    """

    def __init__(
        self,
        target_resolution: Tuple[int, int] = (640, 480),
        flow_params: Optional[Dict] = None,
        scene_change_threshold: float = 0.85,
    ):
        """
        Args:
            target_resolution: (width, height) to resize frames for analysis.
            flow_params: Farneback optical flow parameters.
            scene_change_threshold: Correlation threshold for scene change.
        """
        self.target_resolution = target_resolution
        self.scene_change_threshold = scene_change_threshold

        # Farneback optical flow defaults
        self.flow_params = flow_params or {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0,
        }

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Extract all visual features from a single frame.

        Args:
            frame: BGR image (H, W, 3) as numpy array.

        Returns:
            Dict with keys: visual_complexity, edge_density,
            color_entropy, spatial_frequency, scene_change_score,
            mean_flow_magnitude, saliency_peak.
        """
        # Resize for consistent analysis
        frame = cv2.resize(frame, self.target_resolution)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        features = {}

        # Visual complexity metrics
        features["edge_density"] = self._compute_edge_density(gray)
        features["color_entropy"] = self._compute_color_entropy(frame)
        features["spatial_frequency"] = self._compute_spatial_frequency(gray)
        features["visual_complexity"] = (
            0.4 * features["edge_density"]
            + 0.3 * features["color_entropy"]
            + 0.3 * features["spatial_frequency"]
        )

        # Scene change detection
        features["scene_change_score"] = self._compute_scene_change(gray)

        # Optical flow (requires previous frame)
        flow_mag = self._compute_optical_flow(gray)
        features["mean_flow_magnitude"] = float(np.mean(flow_mag)) if flow_mag is not None else 0.0
        features["max_flow_magnitude"] = float(np.max(flow_mag)) if flow_mag is not None else 0.0

        # Saliency
        saliency_map = self._compute_saliency(gray)
        features["saliency_peak"] = float(np.max(saliency_map))
        features["saliency_mean"] = float(np.mean(saliency_map))

        # Store for next frame
        self._prev_gray = gray.copy()

        return features

    def compute_batch_features(self, frames: list) -> list:
        """Analyze a sequence of frames."""
        self.reset()
        return [self.analyze_frame(f) for f in frames]

    def reset(self):
        """Reset internal state (previous frame, histogram)."""
        self._prev_gray = None
        self._prev_hist = None

    # ------------------------------------------------------------------
    # Feature computation methods
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_edge_density(gray: np.ndarray) -> float:
        """
        Compute edge density using Canny edge detection.
        Returns ratio of edge pixels to total pixels in [0, 1].
        """
        edges = cv2.Canny(gray, 50, 150)
        return float(np.count_nonzero(edges) / edges.size)

    @staticmethod
    def _compute_color_entropy(frame: np.ndarray) -> float:
        """
        Compute normalized color histogram entropy.
        Higher entropy = more complex color distribution.
        Returns value in [0, 1].
        """
        # Compute histogram for each channel
        entropy_sum = 0.0
        for ch in range(3):
            hist = cv2.calcHist([frame], [ch], None, [64], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-12)
            # Shannon entropy
            nonzero = hist[hist > 0]
            entropy_sum += float(-np.sum(nonzero * np.log2(nonzero)))

        # Normalize: max entropy for 64 bins = log2(64) = 6.0
        max_entropy = 6.0 * 3  # 3 channels
        return min(1.0, entropy_sum / max_entropy)

    @staticmethod
    def _compute_spatial_frequency(gray: np.ndarray) -> float:
        """
        Compute spatial frequency via Laplacian variance.
        Higher = more spatial detail/texture.
        Returns normalized value in [0, 1].
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(np.var(laplacian))
        # Normalize with sigmoid-like scaling
        return min(1.0, variance / 2000.0)

    def _compute_scene_change(self, gray: np.ndarray) -> float:
        """
        Detect scene changes via histogram correlation.
        Returns 1.0 - correlation (high = big change).
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if self._prev_hist is None:
            self._prev_hist = hist
            return 0.0

        correlation = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL)
        self._prev_hist = hist

        return float(1.0 - max(0.0, correlation))

    def _compute_optical_flow(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute dense optical flow using Farneback method.
        Returns magnitude array or None if no previous frame.
        """
        if self._prev_gray is None:
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            self.flow_params["pyr_scale"],
            self.flow_params["levels"],
            self.flow_params["winsize"],
            self.flow_params["iterations"],
            self.flow_params["poly_n"],
            self.flow_params["poly_sigma"],
            self.flow_params["flags"],
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return magnitude

    @staticmethod
    def _compute_saliency(gray: np.ndarray) -> np.ndarray:
        """
        Compute spectral residual saliency map.
        Returns normalized saliency in [0, 1].
        """
        # Spectral residual approach
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        magnitude = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
        log_magnitude = np.log(magnitude + 1e-12)

        # Smooth log-magnitude spectrum
        kernel_size = 3
        smooth_log = cv2.blur(log_magnitude, (kernel_size, kernel_size))

        # Spectral residual
        spectral_residual = log_magnitude - smooth_log

        # Reconstruct with residual amplitude
        phase = cv2.phase(dft[:, :, 0], dft[:, :, 1])
        exp_sr = np.exp(spectral_residual)
        real = exp_sr * np.cos(phase)
        imag = exp_sr * np.sin(phase)

        complex_sr = np.zeros_like(dft)
        complex_sr[:, :, 0] = real
        complex_sr[:, :, 1] = imag

        saliency = cv2.idft(complex_sr)
        saliency = cv2.magnitude(saliency[:, :, 0], saliency[:, :, 1])
        saliency = cv2.GaussianBlur(saliency, (51, 51), 0)

        # Normalize to [0, 1]
        if saliency.max() > 0:
            saliency = saliency / saliency.max()

        return saliency
