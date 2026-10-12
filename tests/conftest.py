"""
Project ECHO — Shared Test Fixtures
=====================================
Provides reusable fixtures for the test suite including
synthetic keystroke data, POMDP payloads, model instances,
game sessions, and OpenCV test frames.
"""

import sys
import os
import pytest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def synthetic_iki_nominal():
    """
    Generate synthetic IKI data simulating genuine text composition.
    Expected to produce CLC in [0.35, 0.55] across all windows.
    """
    np.random.seed(42)
    n = 100
    c_block = np.linspace(0.1, 1.0, 20)
    r_block = c_block.copy()
    # Swap 3 pairs to achieve Spearman rank correlation = 0.3925 (within [0.35, 0.55])
    r_block[1], r_block[9] = r_block[9], r_block[1]
    r_block[2], r_block[14] = r_block[14], r_block[2]
    r_block[3], r_block[17] = r_block[17], r_block[3]

    complexity = np.tile(c_block, 5)
    iki = 150.0 + np.tile(r_block, 5) * 50.0
    return iki, complexity


@pytest.fixture
def synthetic_iki_pasted():
    """
    Generate synthetic IKI data simulating pasted (non-generative) text.
    Very uniform timing — expected CLC < 0.35.
    """
    np.random.seed(99)
    n = 100
    iki = np.random.normal(50.0, 2.0, n)
    complexity = np.random.uniform(0, 1, n)
    return iki, complexity


@pytest.fixture
def synthetic_trajectory():
    """
    Generate a synthetic mouse trajectory for Sigma-Lognormal testing.
    Synthesizes smooth lognormal strokes at 500 Hz sampling rate so velocity reconstruction exceeds 24 dB SNR.
    """
    n = 1000
    t = np.linspace(0.002, 2.0, n)

    # 3 Well-separated lognormal strokes: D, t0, mu, sigma
    strokes = [
        (100.0, 0.10, -1.0, 0.4),
        (150.0, 0.75, -0.8, 0.4),
        (120.0, 1.40, -0.9, 0.4),
    ]

    v_total = np.zeros_like(t)
    for D, t0, mu, sigma in strokes:
        valid = t > t0
        dt = t[valid] - t0
        log_dt = np.log(dt)
        exponent = -((log_dt - mu) ** 2) / (2.0 * sigma ** 2)
        denom = dt * sigma * np.sqrt(2.0 * np.pi)
        v_total[valid] += D * np.exp(exponent) / denom

    dt_step = t[1] - t[0]
    x = np.cumsum(v_total) * dt_step
    y = np.zeros_like(x)

    return x, y, t


@pytest.fixture
def high_trust_high_risk_payload():
    """Adaptation engine payload: high trust + high risk → counter automation bias."""
    return {
        "belief_state": {
            "trust_mean": 0.85,
            "trust_variance": 0.05,
            "cognitive_load": 0.4,
            "clc_raw": 0.45,
            "emotion_label": "focused",
            "emotion_confidence": 0.75,
        },
        "task_context": {
            "current_action": "deploy to production",
            "risk_level": "high",
        },
        "interaction_history": [],
    }


@pytest.fixture
def low_trust_low_risk_payload():
    """Adaptation engine payload: low trust + low risk → concise justification."""
    return {
        "belief_state": {
            "trust_mean": 0.2,
            "trust_variance": 0.03,
            "cognitive_load": 0.3,
            "clc_raw": 0.40,
            "emotion_label": "neutral",
            "emotion_confidence": 0.5,
        },
        "task_context": {
            "current_action": "format code",
            "risk_level": "low",
        },
        "interaction_history": [],
    }


@pytest.fixture
def high_cognitive_load_payload():
    """Adaptation engine payload: cognitive_load > 0.7 → simplify view."""
    return {
        "belief_state": {
            "trust_mean": 0.5,
            "trust_variance": 0.1,
            "cognitive_load": 0.85,
            "clc_raw": 0.50,
            "emotion_label": "fatigued",
            "emotion_confidence": 0.8,
        },
        "task_context": {
            "current_action": "review pull request",
            "risk_level": "med",
        },
        "interaction_history": [],
    }


# ============================================
# NEW: Game Framework Fixtures
# ============================================

@pytest.fixture
def game_registry():
    """Pre-initialized GameRegistry with all 5 genres."""
    from game_framework.game_registry import GameRegistry
    return GameRegistry()


@pytest.fixture
def fps_profile(game_registry):
    """FPS game profile."""
    return game_registry.get_profile("left_4_dead")


@pytest.fixture
def parameter_modulator():
    """Fresh ParameterModulator."""
    from game_framework.parameter_modulator import ParameterModulator
    return ParameterModulator(smoothing_factor=1.0)  # No smoothing for tests


@pytest.fixture
def session_simulator():
    """Pre-seeded SessionSimulator."""
    from game_framework.session_simulator import SessionSimulator
    return SessionSimulator(seed=42)


@pytest.fixture
def short_fps_session(session_simulator):
    """A short (60s) FPS session for quick tests."""
    return session_simulator.simulate("left_4_dead", duration_seconds=60)


# ============================================
# NEW: OpenCV Vision Fixtures
# ============================================

@pytest.fixture
def synthetic_frame():
    """Generate a synthetic BGR game frame (640x480)."""
    np.random.seed(42)
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # Add some structure: gradient + noise
    gradient = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
    frame[:, :, 0] = gradient  # Blue gradient
    return frame


@pytest.fixture
def synthetic_frame_pair():
    """Generate two consecutive synthetic frames with known motion."""
    np.random.seed(42)
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a white rectangle
    frame1[200:280, 200:350] = 255

    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    # Rectangle shifted 20 pixels right, 10 pixels down
    frame2[210:290, 220:370] = 255

    return frame1, frame2


@pytest.fixture
def synthetic_trajectory_points():
    """Generate synthetic mouse trajectory for attention mapping."""
    np.random.seed(42)
    n = 200
    # Simulate focused mouse movement around screen center
    xs = np.random.normal(320, 80, n).astype(np.float64)
    ys = np.random.normal(240, 60, n).astype(np.float64)
    return xs, ys


# ============================================
# Phase 1: Biometric Precision Fixtures
# ============================================

@pytest.fixture
def idelog_multi_stroke_trajectory():
    """
    Generate a multi-stroke synthetic trajectory for IDeLog testing.
    Contains 4 well-separated ballistic sub-movements with pauses.
    """
    np.random.seed(42)
    segments = []
    t_offset = 0.0

    for i in range(4):
        n_pts = 150
        t = np.linspace(t_offset + 0.01, t_offset + 0.8, n_pts)
        D, t0, mu, sigma = 100 + i * 30, t_offset + 0.02, -0.9 + i * 0.1, 0.35

        v = np.zeros_like(t)
        valid = t > t0
        dt = t[valid] - t0
        log_dt = np.log(dt)
        exponent = -((log_dt - mu) ** 2) / (2.0 * sigma ** 2)
        denom = dt * sigma * np.sqrt(2.0 * np.pi)
        v[valid] = D * np.exp(exponent) / denom

        dt_step = t[1] - t[0]
        x_seg = np.cumsum(v) * dt_step
        y_seg = np.cumsum(v * 0.2) * dt_step

        segments.append((x_seg, y_seg, t))
        t_offset = t[-1] + 0.3  # Pause between strokes

    xs = np.concatenate([s[0] for s in segments])
    ys = np.concatenate([s[1] for s in segments])
    ts = np.concatenate([s[2] for s in segments])

    return xs, ys, ts


@pytest.fixture
def neuromotor_sigma_mu_series():
    """
    Generate synthetic σ and μ time series for CLC neuromotor load testing.
    Simulates a session where fatigue gradually increases.
    """
    np.random.seed(42)
    n = 20
    # Baseline: σ=0.40, μ=-0.80
    # Gradual fatigue: σ drifts up, μ drifts up
    sigma_series = 0.40 + np.linspace(0, 0.4, n) + np.random.normal(0, 0.02, n)
    mu_series = -0.80 + np.linspace(0, 0.6, n) + np.random.normal(0, 0.03, n)

    return sigma_series, mu_series


@pytest.fixture
def snr_signal_pairs():
    """
    Generate signal pairs with known SNR tiers for testing.
    Returns dict mapping tier name to (original, reconstructed) arrays.
    """
    np.random.seed(42)
    n = 200
    original = np.sin(np.linspace(0, 4 * np.pi, n)) * 10

    pairs = {}

    # Excellent: very low noise (SNR > 25 dB)
    noise_excellent = np.random.normal(0, 0.05, n)
    pairs["excellent"] = (original, original + noise_excellent)

    # Good: moderate noise (SNR 20–25 dB)
    noise_good = np.random.normal(0, 0.3, n)
    pairs["good"] = (original, original + noise_good)

    # Acceptable: higher noise (SNR 15–20 dB)
    noise_acceptable = np.random.normal(0, 0.8, n)
    pairs["acceptable"] = (original, original + noise_acceptable)

    # Unsatisfactory: heavy noise (SNR < 15 dB)
    noise_unsatisfactory = np.random.normal(0, 3.0, n)
    pairs["unsatisfactory"] = (original, original + noise_unsatisfactory)

    return pairs

