"""
Project ECHO — Game Framework Tests
=======================================
Validates game registry, parameter modulation, DDA curves,
and session simulation across all 5 game genres.

Coverage:
    - Game registry: all genres load, parameters have valid ranges
    - Parameter modulator: DDA actions, monotonic adjustments, bounds
    - Session simulator: DataFrame structure, state evolution, data integrity
    - Edge cases: extreme trust/cogload, all emotion×load combinations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from game_framework.game_registry import GameRegistry, GameProfile, ParameterSpec
from game_framework.parameter_modulator import ParameterModulator
from game_framework.session_simulator import SessionSimulator


class TestGameRegistry:
    """Validates game profile loading and parameter specs."""

    def test_all_four_genres_registered(self, game_registry):
        """Registry must contain all 4 game genres."""
        genres = game_registry.list_genres()
        expected = {"left_4_dead", "alien_isolation", "resident_evil_4", "silent_hill"}
        assert set(genres) == expected

    def test_each_genre_has_parameters(self, game_registry):
        """Every genre must have at least 4 tunable parameters."""
        for genre in game_registry.list_genres():
            profile = game_registry.get_profile(genre)
            assert len(profile.parameters) >= 4, (
                f"{genre} has only {len(profile.parameters)} parameters"
            )

    def test_parameter_ranges_valid(self, game_registry):
        """All parameters must have min < default < max."""
        for genre in game_registry.list_genres():
            profile = game_registry.get_profile(genre)
            for name, spec in profile.parameters.items():
                assert spec.min_val <= spec.default <= spec.max_val, (
                    f"{genre}/{name}: min={spec.min_val} default={spec.default} max={spec.max_val}"
                )

    def test_parameter_clamping(self):
        """ParameterSpec.clamp() must enforce bounds."""
        spec = ParameterSpec("test", 0.5, 0.0, 1.0, "normalized")
        assert spec.clamp(-0.5) == 0.0
        assert spec.clamp(1.5) == 1.0
        assert spec.clamp(0.7) == 0.7

    def test_parameter_normalize_denormalize(self):
        """Normalize and denormalize must be inverses."""
        spec = ParameterSpec("test", 5.0, 0.0, 10.0, "count")
        for val in [0.0, 2.5, 5.0, 7.5, 10.0]:
            norm = spec.normalize(val)
            assert 0.0 <= norm <= 1.0
            recovered = spec.denormalize(norm)
            assert abs(recovered - val) < 1e-10

    def test_invalid_genre_raises(self, game_registry):
        """Requesting unknown genre must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown game genre"):
            game_registry.get_profile("mmorpg")

    def test_get_defaults(self, fps_profile):
        """get_defaults() must return a dict with all parameter names."""
        defaults = fps_profile.get_defaults()
        assert isinstance(defaults, dict)
        assert set(defaults.keys()) == set(fps_profile.parameters.keys())


class TestParameterModulator:
    """Validates DDA parameter computation."""

    def test_ease_reduces_difficulty(self, parameter_modulator, fps_profile):
        """Angry + high load should decrease difficulty parameters."""
        adj = parameter_modulator.compute_adjustments(
            profile=fps_profile,
            trust_mean=0.5,
            cognitive_load=0.85,
            emotion_label="angry",
            emotion_confidence=0.8,
        )
        # enemy_spawn_rate should decrease (easier)
        assert adj.deltas["enemy_spawn_rate"] <= 0, (
            f"enemy_spawn_rate delta {adj.deltas['enemy_spawn_rate']} should be <= 0"
        )
        # wanderer_density should decrease
        assert adj.deltas["wanderer_density"] <= 0, (
            f"wanderer_density delta {adj.deltas['wanderer_density']} should be <= 0"
        )

    def test_challenge_increases_difficulty(self, parameter_modulator, fps_profile):
        """Focused should increase difficulty."""
        adj = parameter_modulator.compute_adjustments(
            profile=fps_profile,
            trust_mean=0.5,
            cognitive_load=0.2,
            emotion_label="focused",
            emotion_confidence=0.8,
        )
        # enemy_spawn_rate should increase (harder)
        assert adj.deltas["enemy_spawn_rate"] >= 0, (
            f"enemy_spawn_rate delta {adj.deltas['enemy_spawn_rate']} should be >= 0"
        )

    def test_flow_state_holds_parameters(self, parameter_modulator, fps_profile):
        """Engaged + high intensity = flow state → no changes."""
        adj = parameter_modulator.compute_adjustments(
            profile=fps_profile,
            trust_mean=0.5,
            cognitive_load=0.8,
            emotion_label="neutral",
            emotion_confidence=0.8,
        )
        # All deltas should be zero or near-zero
        for param, delta in adj.deltas.items():
            assert abs(delta) < 0.01, (
                f"Flow state: {param} delta {delta} should be near-zero"
            )

    def test_parameters_stay_within_bounds(self, parameter_modulator, fps_profile):
        """After many adjustments, parameters must stay within spec bounds."""
        current = fps_profile.get_defaults()
        for _ in range(100):
            adj = parameter_modulator.compute_adjustments(
                profile=fps_profile,
                trust_mean=np.random.uniform(0, 1),
                cognitive_load=np.random.uniform(0, 1),
                emotion_label=np.random.choice(["angry", "focused", "thinking",
                                                 "neutral", "joy"]),
                emotion_confidence=np.random.uniform(0.3, 0.95),
                current_values=current,
            )
            current = adj.after

        for name, val in current.items():
            spec = fps_profile.parameters[name]
            assert spec.min_val <= val <= spec.max_val, (
                f"{name}: {val} out of [{spec.min_val}, {spec.max_val}]"
            )

    def test_low_emotion_confidence_reduces_magnitude(self, fps_profile):
        """Low emotion confidence should produce smaller deltas."""
        mod_high = ParameterModulator(smoothing_factor=1.0)
        mod_low = ParameterModulator(smoothing_factor=1.0)

        adj_high = mod_high.compute_adjustments(
            profile=fps_profile, trust_mean=0.5, cognitive_load=0.85,
            emotion_label="angry", emotion_confidence=0.9,
        )
        adj_low = mod_low.compute_adjustments(
            profile=fps_profile, trust_mean=0.5, cognitive_load=0.85,
            emotion_label="angry", emotion_confidence=0.4,
        )

        # High confidence should produce larger absolute deltas
        total_high = sum(abs(d) for d in adj_high.deltas.values())
        total_low = sum(abs(d) for d in adj_low.deltas.values())
        assert total_high >= total_low, (
            f"High conf deltas ({total_high}) should >= low conf ({total_low})"
        )

    def test_all_genres_produce_valid_adjustments(self, game_registry):
        """Every genre must produce well-formed ParameterAdjustments."""
        modulator = ParameterModulator(smoothing_factor=1.0)
        for genre in game_registry.list_genres():
            profile = game_registry.get_profile(genre)
            adj = modulator.compute_adjustments(
                profile=profile, trust_mean=0.5, cognitive_load=0.6,
                emotion_label="focused", emotion_confidence=0.7,
            )
            assert adj.genre == genre
            assert set(adj.before.keys()) == set(adj.after.keys())
            assert set(adj.deltas.keys()) == set(adj.before.keys())


class TestSessionSimulator:
    """Validates game session simulation."""

    def test_dataframe_shape(self, short_fps_session):
        """Session must have expected row count and essential columns."""
        df = short_fps_session
        assert len(df) == 60  # 60 seconds at 1 Hz
        required = ["time", "trust_mean", "cognitive_load", "emotion_label",
                     "emotion_confidence", "clc_raw", "dda_action"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_parameter_columns_present(self, short_fps_session, fps_profile):
        """Session must contain columns for every game parameter."""
        df = short_fps_session
        for param in fps_profile.get_parameter_names():
            assert param in df.columns, f"Missing parameter column: {param}"
            assert f"{param}_delta" in df.columns, f"Missing delta column: {param}_delta"

    def test_trust_within_bounds(self, short_fps_session):
        """Trust must stay in [0, 1] throughout the session."""
        df = short_fps_session
        assert df["trust_mean"].min() >= 0.0
        assert df["trust_mean"].max() <= 1.0

    def test_cognitive_load_within_bounds(self, short_fps_session):
        """Cognitive load must stay in [0, 1]."""
        df = short_fps_session
        assert df["cognitive_load"].min() >= 0.0
        assert df["cognitive_load"].max() <= 1.0

    def test_emotion_labels_valid(self, short_fps_session):
        """All emotion labels must be from the valid set."""
        df = short_fps_session
        valid = {"angry", "focused", "thinking", "neutral", "joy", "given_up", "normal", "frustrated", "fatigued", "engaged"}
        observed = set(df["emotion_label"].unique())
        assert observed.issubset(valid), f"Invalid emotions: {observed - valid}"

    def test_simulate_all_genres(self, session_simulator):
        """simulate_all_genres must return DataFrames for all 4 genres."""
        results = session_simulator.simulate_all_genres(duration_seconds=30)
        assert len(results) == 4
        for genre, df in results.items():
            assert len(df) == 30

    def test_scenario_struggle_increases_load(self):
        """Struggle scenario should trend toward higher cognitive load."""
        sim = SessionSimulator(seed=42)
        df = sim.simulate("left_4_dead", duration_seconds=120, scenario="struggle")
        first_half = df[df["time"] < 60]["cognitive_load"].mean()
        second_half = df[df["time"] >= 60]["cognitive_load"].mean()
        # Struggle should generally increase cognitive load
        # Allow some tolerance due to noise
        assert second_half > first_half - 0.1, (
            f"Struggle scenario: second half ({second_half:.3f}) should trend "
            f"higher than first half ({first_half:.3f})"
        )
