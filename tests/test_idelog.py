"""
Project ECHO — IDeLog Extractor Tests
========================================
Validates the IDeLog real-time neuromotor extraction algorithm
(Phase 1).

Tests:
    1. Virtual target point detection on synthetic trajectories
    2. 6-parameter extraction produces valid NeuromotorCommands
    3. Convergence within max iterations
    4. Degenerate / insufficient input handling
    5. SNR quality propagation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from feature_extraction.idelog_extractor import (
    IDeLogExtractor, NeuromotorCommand, TrajectorySegment, IDeLogResult,
)
from feature_extraction.snr_validator import SNRQuality


@pytest.fixture
def smooth_idelog_trajectory():
    """
    Generate a synthetic trajectory with 3 well-separated ballistic
    sub-movements for IDeLog testing.
    """
    n = 500
    t = np.linspace(0.01, 3.0, n)

    # 3 lognormal velocity strokes
    strokes = [
        (120.0, 0.10, -1.0, 0.35),
        (180.0, 1.10, -0.8, 0.35),
        (140.0, 2.10, -0.9, 0.35),
    ]

    vx_total = np.zeros_like(t)
    for D, t0, mu, sigma in strokes:
        valid = t > t0
        dt = t[valid] - t0
        log_dt = np.log(dt)
        exponent = -((log_dt - mu) ** 2) / (2.0 * sigma ** 2)
        denom = dt * sigma * np.sqrt(2.0 * np.pi)
        vx_total[valid] += D * np.exp(exponent) / denom

    dt_step = t[1] - t[0]
    xs = np.cumsum(vx_total) * dt_step
    ys = np.cumsum(vx_total * 0.3) * dt_step  # Slight y component

    return xs, ys, t


class TestIDeLogExtractor:
    """Tests for the IDeLog real-time extractor."""

    def test_extracts_neuromotor_commands(self, smooth_idelog_trajectory):
        """IDeLog should extract at least one NeuromotorCommand from a valid trajectory."""
        xs, ys, ts = smooth_idelog_trajectory
        extractor = IDeLogExtractor(max_iterations=30)

        result = extractor.process_raw(xs, ys, ts)

        assert isinstance(result, IDeLogResult)
        assert len(result.commands) > 0
        assert all(isinstance(c, NeuromotorCommand) for c in result.commands)

    def test_six_parameter_fields_valid(self, smooth_idelog_trajectory):
        """Each NeuromotorCommand should have valid 6-parameter values."""
        xs, ys, ts = smooth_idelog_trajectory
        extractor = IDeLogExtractor(max_iterations=20)

        result = extractor.process_raw(xs, ys, ts)

        for cmd in result.commands:
            assert cmd.D > 0, "Amplitude D must be positive"
            assert cmd.sigma > 0, "Sigma must be positive"
            assert -np.pi <= cmd.theta_start <= np.pi
            assert -np.pi <= cmd.theta_end <= np.pi

    def test_convergence_within_iterations(self, smooth_idelog_trajectory):
        """IDeLog should converge or exhaust max_iterations."""
        xs, ys, ts = smooth_idelog_trajectory
        extractor = IDeLogExtractor(max_iterations=50)

        result = extractor.process_raw(xs, ys, ts)

        assert result.iterations_used > 0
        assert result.iterations_used <= 50

    def test_virtual_targets_detected(self, smooth_idelog_trajectory):
        """Virtual target points should be detected between sub-movements."""
        xs, ys, ts = smooth_idelog_trajectory
        extractor = IDeLogExtractor(max_iterations=20)

        result = extractor.process_raw(xs, ys, ts)

        assert len(result.virtual_targets) >= 2

    def test_snr_quality_propagated(self, smooth_idelog_trajectory):
        """SNR result should be populated with quality classification."""
        xs, ys, ts = smooth_idelog_trajectory
        extractor = IDeLogExtractor(max_iterations=30)

        result = extractor.process_raw(xs, ys, ts)

        assert result.snr_result is not None
        assert result.snr_result.snr_db >= 0
        assert result.snr_result.quality in SNRQuality

    def test_degenerate_single_point(self):
        """Single-point trajectory should return empty result."""
        extractor = IDeLogExtractor()

        result = extractor.process_raw(
            np.array([0.0]), np.array([0.0]), np.array([0.0])
        )

        assert len(result.commands) == 0
        assert result.snr_result.should_discard is True

    def test_degenerate_too_few_points(self):
        """Trajectory with fewer than min_segment_points should return empty."""
        extractor = IDeLogExtractor(min_segment_points=20)

        result = extractor.process_raw(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.1, 0.2]),
        )

        assert len(result.commands) == 0
        assert result.snr_result.should_discard is True

    def test_segment_id_assignment(self, smooth_idelog_trajectory):
        """Each command should carry the correct segment_id."""
        xs, ys, ts = smooth_idelog_trajectory
        extractor = IDeLogExtractor(max_iterations=10)

        result = extractor.process_raw(xs, ys, ts)

        if result.commands:
            assert all(c.segment_id > 0 for c in result.commands)

    def test_trajectory_segment_dataclass(self):
        """TrajectorySegment should accept all fields correctly."""
        seg = TrajectorySegment(
            xs=np.array([0, 1, 2], dtype=np.float64),
            ys=np.array([0, 1, 2], dtype=np.float64),
            ts=np.array([0.0, 0.1, 0.2], dtype=np.float64),
            segment_id=42,
        )
        assert seg.segment_id == 42
        assert len(seg.xs) == 3

    def test_velocity_arrays_stored_in_result(self, smooth_idelog_trajectory):
        """Result should contain original and reconstructed velocity arrays."""
        xs, ys, ts = smooth_idelog_trajectory
        extractor = IDeLogExtractor(max_iterations=10)

        result = extractor.process_raw(xs, ys, ts)

        if result.commands:
            assert result.original_velocity is not None
            assert result.reconstructed_velocity is not None
            assert result.times is not None
            assert len(result.original_velocity) == len(result.reconstructed_velocity)
