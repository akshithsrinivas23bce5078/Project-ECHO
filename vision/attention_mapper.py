"""
Project ECHO — Attention Mapper
==================================
Maps mouse trajectory onto game frames to produce attention heatmaps
using Gaussian Kernel Density Estimation.

The attention heatmap reveals which regions of the screen the player
focuses on, enabling correlation with game events and cognitive state.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


class AttentionMapper:
    """
    Generates attention heatmaps from mouse trajectory data.

    Uses Gaussian KDE to convert discrete (x, y) trajectory points
    into a smooth, continuous attention density map.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        kernel_bandwidth: float = 30.0,
        decay_factor: float = 0.95,
    ):
        """
        Args:
            resolution: (width, height) of the output heatmap.
            kernel_bandwidth: Gaussian kernel sigma for KDE.
            decay_factor: Temporal decay for older trajectory points.
                         1.0 = no decay, 0.0 = only latest point.
        """
        self.resolution = resolution
        self.kernel_bandwidth = kernel_bandwidth
        self.decay_factor = decay_factor

    def generate_heatmap(
        self,
        trajectory_x: np.ndarray,
        trajectory_y: np.ndarray,
        screen_resolution: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Generate an attention heatmap from trajectory points.

        Args:
            trajectory_x: 1-D array of x positions.
            trajectory_y: 1-D array of y positions.
            screen_resolution: Original screen resolution for normalization.

        Returns:
            2-D array (height, width) with values in [0, 1].
        """
        w, h = self.resolution
        heatmap = np.zeros((h, w), dtype=np.float64)

        if len(trajectory_x) == 0:
            return heatmap

        # Normalize coordinates to heatmap resolution
        xs = trajectory_x.copy().astype(np.float64)
        ys = trajectory_y.copy().astype(np.float64)

        if screen_resolution is not None:
            sw, sh = screen_resolution
            xs = xs / sw * w
            ys = ys / sh * h

        # Clamp to bounds
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)

        n = len(xs)

        # Temporal decay weights: newer points contribute more
        weights = np.array([
            self.decay_factor ** (n - 1 - i) for i in range(n)
        ])

        # Accumulate Gaussian contributions
        for i in range(n):
            cx, cy = int(xs[i]), int(ys[i])
            weight = weights[i]

            # Create local Gaussian kernel
            sigma = self.kernel_bandwidth
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Region of interest
            x_start = max(0, cx - kernel_size // 2)
            x_end = min(w, cx + kernel_size // 2 + 1)
            y_start = max(0, cy - kernel_size // 2)
            y_end = min(h, cy + kernel_size // 2 + 1)

            if x_end <= x_start or y_end <= y_start:
                continue

            # Generate Gaussian values for the ROI
            yy, xx = np.mgrid[y_start:y_end, x_start:x_end]
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            heatmap[y_start:y_end, x_start:x_end] += weight * gaussian

        # Normalize to [0, 1]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap /= max_val

        return heatmap

    def overlay_on_frame(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay a heatmap onto a game frame.

        Args:
            frame: BGR image.
            heatmap: 2-D array in [0, 1].
            alpha: Overlay transparency (0 = frame only, 1 = heatmap only).
            colormap: OpenCV colormap for the heatmap.

        Returns:
            BGR image with heatmap overlay.
        """
        # Resize heatmap to match frame
        h, w = frame.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, colormap)

        # Blend
        overlay = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
        return overlay

    def compute_attention_metrics(self, heatmap: np.ndarray) -> dict:
        """
        Compute summary metrics from an attention heatmap.

        Returns:
            Dict with: focus_area_ratio, entropy, peak_location,
            centroid, dispersion.
        """
        h, w = heatmap.shape

        # Focus area: ratio of pixels above 50% of max intensity
        threshold = 0.5
        focus_mask = heatmap > threshold
        focus_area_ratio = float(np.sum(focus_mask) / focus_mask.size)

        # Attention entropy (spatial distribution)
        flat = heatmap.flatten()
        flat_norm = flat / (flat.sum() + 1e-12)
        nonzero = flat_norm[flat_norm > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        max_entropy = np.log2(flat.size)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Peak and centroid
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        ys, xs = np.mgrid[0:h, 0:w]
        total = heatmap.sum() + 1e-12
        centroid_x = float(np.sum(xs * heatmap) / total)
        centroid_y = float(np.sum(ys * heatmap) / total)

        # Dispersion: weighted standard deviation of positions
        dx = xs - centroid_x
        dy = ys - centroid_y
        dispersion = float(np.sqrt(np.sum(heatmap * (dx ** 2 + dy ** 2)) / total))

        return {
            "focus_area_ratio": round(focus_area_ratio, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "peak_location": (int(peak_x), int(peak_y)),
            "centroid": (round(centroid_x, 1), round(centroid_y, 1)),
            "dispersion": round(dispersion, 2),
        }
