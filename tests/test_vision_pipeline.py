"""
Project ECHO — Vision Pipeline Tests
========================================
Validates OpenCV frame analysis, attention mapping,
and motion analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import cv2

from vision.frame_analyzer import FrameAnalyzer
from vision.attention_mapper import AttentionMapper
from vision.motion_analyzer import MotionAnalyzer


class TestFrameAnalyzer:
    """Validates OpenCV frame feature extraction."""

    def test_analyze_frame_returns_all_features(self, synthetic_frame):
        """analyze_frame must return all expected feature keys."""
        analyzer = FrameAnalyzer()
        features = analyzer.analyze_frame(synthetic_frame)

        expected_keys = [
            "edge_density", "color_entropy", "spatial_frequency",
            "visual_complexity", "scene_change_score",
            "mean_flow_magnitude", "max_flow_magnitude",
            "saliency_peak", "saliency_mean",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

    def test_feature_bounds(self, synthetic_frame):
        """All normalized features must be in [0, 1]."""
        analyzer = FrameAnalyzer()
        features = analyzer.analyze_frame(synthetic_frame)

        for key in ["edge_density", "color_entropy", "spatial_frequency", "visual_complexity"]:
            assert 0.0 <= features[key] <= 1.0, f"{key} = {features[key]} out of [0, 1]"

    def test_scene_change_on_identical_frames(self, synthetic_frame):
        """Two identical frames should have near-zero scene change."""
        analyzer = FrameAnalyzer()
        analyzer.analyze_frame(synthetic_frame)  # First frame
        features = analyzer.analyze_frame(synthetic_frame)  # Same frame

        assert features["scene_change_score"] < 0.05, (
            f"Scene change {features['scene_change_score']} too high for identical frames"
        )

    def test_scene_change_on_different_frames(self, synthetic_frame):
        """Very different frames should have high scene change."""
        analyzer = FrameAnalyzer()
        analyzer.analyze_frame(synthetic_frame)

        # Create a completely different frame
        different = np.full_like(synthetic_frame, 128)
        features = analyzer.analyze_frame(different)

        assert features["scene_change_score"] > 0.1, (
            f"Scene change {features['scene_change_score']} too low for different frames"
        )

    def test_optical_flow_first_frame_is_zero(self, synthetic_frame):
        """Optical flow should be zero for the first frame."""
        analyzer = FrameAnalyzer()
        features = analyzer.analyze_frame(synthetic_frame)
        assert features["mean_flow_magnitude"] == 0.0

    def test_optical_flow_detects_motion(self, synthetic_frame_pair):
        """Optical flow should detect motion between shifted frames."""
        frame1, frame2 = synthetic_frame_pair
        analyzer = FrameAnalyzer()
        analyzer.analyze_frame(frame1)
        features = analyzer.analyze_frame(frame2)
        assert features["mean_flow_magnitude"] > 0, "Flow should detect motion"

    def test_batch_features(self, synthetic_frame):
        """compute_batch_features should return a list of feature dicts."""
        analyzer = FrameAnalyzer()
        frames = [synthetic_frame, synthetic_frame.copy()]
        results = analyzer.compute_batch_features(frames)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_reset_clears_state(self, synthetic_frame):
        """After reset, first frame should have zero flow."""
        analyzer = FrameAnalyzer()
        analyzer.analyze_frame(synthetic_frame)
        analyzer.reset()
        features = analyzer.analyze_frame(synthetic_frame)
        assert features["mean_flow_magnitude"] == 0.0


class TestAttentionMapper:
    """Validates attention heatmap generation."""

    def test_heatmap_shape(self, synthetic_trajectory_points):
        """Heatmap must have the correct dimensions."""
        xs, ys = synthetic_trajectory_points
        mapper = AttentionMapper(resolution=(640, 480))
        heatmap = mapper.generate_heatmap(xs, ys)
        assert heatmap.shape == (480, 640)

    def test_heatmap_normalized(self, synthetic_trajectory_points):
        """Heatmap values must be in [0, 1]."""
        xs, ys = synthetic_trajectory_points
        mapper = AttentionMapper()
        heatmap = mapper.generate_heatmap(xs, ys)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-6

    def test_empty_trajectory_returns_zeros(self):
        """Empty trajectory should produce all-zero heatmap."""
        mapper = AttentionMapper()
        heatmap = mapper.generate_heatmap(np.array([]), np.array([]))
        assert np.all(heatmap == 0)

    def test_attention_metrics(self, synthetic_trajectory_points):
        """Attention metrics should have expected keys and bounds."""
        xs, ys = synthetic_trajectory_points
        mapper = AttentionMapper()
        heatmap = mapper.generate_heatmap(xs, ys)
        metrics = mapper.compute_attention_metrics(heatmap)

        assert "focus_area_ratio" in metrics
        assert "normalized_entropy" in metrics
        assert "peak_location" in metrics
        assert "centroid" in metrics
        assert "dispersion" in metrics
        assert 0.0 <= metrics["focus_area_ratio"] <= 1.0
        assert 0.0 <= metrics["normalized_entropy"] <= 1.0

    def test_overlay_on_frame(self, synthetic_frame, synthetic_trajectory_points):
        """Overlay must produce same-shape BGR image."""
        xs, ys = synthetic_trajectory_points
        mapper = AttentionMapper()
        heatmap = mapper.generate_heatmap(xs, ys)
        overlay = mapper.overlay_on_frame(synthetic_frame, heatmap)
        assert overlay.shape == synthetic_frame.shape


class TestMotionAnalyzer:
    """Validates motion energy computation."""

    def test_compute_motion_energy(self, synthetic_frame_pair):
        """Motion energy must return all expected keys."""
        frame1, frame2 = synthetic_frame_pair
        analyzer = MotionAnalyzer()
        metrics = analyzer.compute_motion_energy(frame1, frame2)

        for key in ["mean_magnitude", "max_magnitude", "action_level",
                     "direction_histogram", "motion_coverage"]:
            assert key in metrics, f"Missing key: {key}"

    def test_zero_motion_on_identical(self, synthetic_frame):
        """Identical frames should produce near-zero motion."""
        analyzer = MotionAnalyzer()
        metrics = analyzer.compute_motion_energy(synthetic_frame, synthetic_frame.copy())
        assert metrics["mean_magnitude"] < 0.5

    def test_action_level_classification(self, synthetic_frame_pair):
        """Action level must be one of high/medium/low."""
        frame1, frame2 = synthetic_frame_pair
        analyzer = MotionAnalyzer()
        metrics = analyzer.compute_motion_energy(frame1, frame2)
        assert metrics["action_level"] in ("high", "medium", "low")

    def test_direction_histogram_sums_to_one(self, synthetic_frame_pair):
        """Direction histogram should approximately sum to 1."""
        frame1, frame2 = synthetic_frame_pair
        analyzer = MotionAnalyzer()
        metrics = analyzer.compute_motion_energy(frame1, frame2)
        hist = metrics["direction_histogram"]
        total = sum(hist)
        if total > 0:
            assert abs(total - 1.0) < 0.01, f"Histogram sums to {total}"

    def test_segment_detection(self):
        """detect_action_segments must return valid segments."""
        analyzer = MotionAnalyzer()
        energies = [0.5] * 10 + [8.0] * 10 + [0.5] * 10
        segments = analyzer.detect_action_segments(energies, min_segment_length=5)
        assert len(segments) >= 2  # At least low-high or low-high-low
        levels = [s["action_level"] for s in segments]
        assert "high" in levels
        assert "low" in levels

    def test_correlate_with_cognitive_load(self):
        """Correlation should return valid statistics."""
        analyzer = MotionAnalyzer()
        me = np.random.normal(3, 1, 50)
        cl = np.random.normal(0.5, 0.2, 50)
        result = analyzer.correlate_with_cognitive_load(me, cl)
        assert "pearson_r" in result
        assert "spearman_rho" in result
        assert -1.0 <= result["pearson_r"] <= 1.0
