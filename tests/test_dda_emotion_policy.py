"""
Tests for the 5-State Emotion-Based DDA Policy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from game_framework.game_registry import GameRegistry
from game_framework.parameter_modulator import ParameterModulator

@pytest.fixture
def game_registry():
    return GameRegistry()

@pytest.fixture
def modulator():
    # Use smoothing_factor=1.0 for testing raw deltas without smoothing history
    return ParameterModulator(smoothing_factor=1.0)

def test_neutral_state_no_changes(game_registry, modulator):
    """Neutral state should produce exactly zero deltas for all parameters."""
    for genre in game_registry.list_genres():
        profile = game_registry.get_profile(genre)
        adj = modulator.compute_adjustments(
            profile=profile,
            trust_mean=0.5,
            cognitive_load=0.5,
            emotion_label="neutral",
            emotion_confidence=1.0
        )
        for param, delta in adj.deltas.items():
            assert abs(delta) < 1e-6, f"Neutral state produced non-zero delta for {param}"

def test_focused_increases_difficulty(game_registry, modulator):
    """Focused state should increase difficulty and decrease assistance."""
    for genre in game_registry.list_genres():
        profile = game_registry.get_profile(genre)
        adj = modulator.compute_adjustments(
            profile=profile,
            trust_mean=0.5,
            cognitive_load=0.5,
            emotion_label="focused",
            emotion_confidence=1.0
        )
        diff_increase = modulator.DIFFICULTY_INCREASE_PARAMS.get(genre, [])
        assist_params = modulator.ASSIST_PARAMS.get(genre, [])
        
        for param, delta in adj.deltas.items():
            if param in diff_increase:
                assert delta > 0, f"Focused should increase {param}"
            elif param in assist_params:
                assert delta < 0, f"Focused should decrease {param}"

def test_joy_increases_difficulty_gently(game_registry, modulator):
    """Joy state should increase difficulty, but less than focused."""
    for genre in game_registry.list_genres():
        profile = game_registry.get_profile(genre)
        
        adj_focused = modulator.compute_adjustments(
            profile=profile, trust_mean=0.5, cognitive_load=0.5,
            emotion_label="focused", emotion_confidence=1.0
        )
        adj_joy = modulator.compute_adjustments(
            profile=profile, trust_mean=0.5, cognitive_load=0.5,
            emotion_label="joy", emotion_confidence=1.0
        )
        
        diff_increase = modulator.DIFFICULTY_INCREASE_PARAMS.get(genre, [])
        for param in diff_increase:
            assert 0 < adj_joy.deltas[param] < adj_focused.deltas[param], \
                f"Joy delta {adj_joy.deltas[param]} for {param} should be positive but less than focused delta {adj_focused.deltas[param]}"

def test_angry_decreases_difficulty(game_registry, modulator):
    """Angry state should decrease difficulty and increase assistance."""
    for genre in game_registry.list_genres():
        profile = game_registry.get_profile(genre)
        adj = modulator.compute_adjustments(
            profile=profile,
            trust_mean=0.5,
            cognitive_load=0.5,
            emotion_label="angry",
            emotion_confidence=1.0
        )
        diff_increase = modulator.DIFFICULTY_INCREASE_PARAMS.get(genre, [])
        assist_params = modulator.ASSIST_PARAMS.get(genre, [])
        
        for param, delta in adj.deltas.items():
            if param in diff_increase:
                assert delta < 0, f"Angry should decrease {param}"
            elif param in assist_params:
                assert delta > 0, f"Angry should increase {param}"

def test_given_up_decreases_difficulty_strongly(game_registry, modulator):
    """Given_up state should decrease difficulty more strongly than angry."""
    for genre in game_registry.list_genres():
        profile = game_registry.get_profile(genre)
        
        adj_angry = modulator.compute_adjustments(
            profile=profile, trust_mean=0.5, cognitive_load=0.5,
            emotion_label="angry", emotion_confidence=1.0
        )
        adj_given_up = modulator.compute_adjustments(
            profile=profile, trust_mean=0.5, cognitive_load=0.5,
            emotion_label="given_up", emotion_confidence=1.0
        )
        
        diff_increase = modulator.DIFFICULTY_INCREASE_PARAMS.get(genre, [])
        assist_params = modulator.ASSIST_PARAMS.get(genre, [])
        
        for param in diff_increase:
            assert adj_given_up.deltas[param] < adj_angry.deltas[param] < 0, \
                f"Given_up delta {adj_given_up.deltas[param]} should be more negative than angry {adj_angry.deltas[param]} for {param}"
                
        for param in assist_params:
            assert adj_given_up.deltas[param] > adj_angry.deltas[param] > 0, \
                f"Given_up assistance {adj_given_up.deltas[param]} should be higher than angry {adj_angry.deltas[param]} for {param}"

def test_thinking_state_split_behavior(game_registry, modulator):
    """Thinking state should decrease combat, increase puzzle, hold specified."""
    for genre in game_registry.list_genres():
        profile = game_registry.get_profile(genre)
        adj = modulator.compute_adjustments(
            profile=profile,
            trust_mean=0.5,
            cognitive_load=0.5,
            emotion_label="thinking",
            emotion_confidence=1.0
        )
        
        diff_increase = modulator.DIFFICULTY_INCREASE_PARAMS.get(genre, [])
        assist_params = modulator.ASSIST_PARAMS.get(genre, [])
        thinking_increase = modulator.THINKING_INCREASE_PARAMS.get(genre, [])
        thinking_hold = modulator.THINKING_HOLD_PARAMS.get(genre, [])
        
        for param, delta in adj.deltas.items():
            if param in thinking_hold:
                assert abs(delta) < 1e-6, f"Thinking should hold {param}"
            elif param in thinking_increase:
                assert delta > 0, f"Thinking should increase cognitive param {param}"
            elif param in assist_params:
                assert delta > 0, f"Thinking should slightly increase assist param {param}"
            elif param in diff_increase:
                assert delta < 0, f"Thinking should decrease combat param {param}"

def test_legacy_emotion_mapping(game_registry, modulator):
    """Legacy emotion labels should map to the new 5-state policy."""
    profile = game_registry.get_profile("left_4_dead")
    
    adj_frustrated = modulator.compute_adjustments(
        profile=profile, trust_mean=0.5, cognitive_load=0.5,
        emotion_label="frustrated", emotion_confidence=1.0
    )
    adj_angry = modulator.compute_adjustments(
        profile=profile, trust_mean=0.5, cognitive_load=0.5,
        emotion_label="angry", emotion_confidence=1.0
    )
    assert adj_frustrated.deltas == adj_angry.deltas
    
    adj_normal = modulator.compute_adjustments(
        profile=profile, trust_mean=0.5, cognitive_load=0.5,
        emotion_label="normal", emotion_confidence=1.0
    )
    adj_neutral = modulator.compute_adjustments(
        profile=profile, trust_mean=0.5, cognitive_load=0.5,
        emotion_label="neutral", emotion_confidence=1.0
    )
    assert adj_normal.deltas == adj_neutral.deltas

def test_smoothing_reduces_delta_magnitude(game_registry):
    """Smoothing factor < 1.0 should result in smaller deltas than raw."""
    profile = game_registry.get_profile("left_4_dead")
    
    mod_raw = ParameterModulator(smoothing_factor=1.0)
    mod_smooth = ParameterModulator(smoothing_factor=0.3)
    
    adj_raw = mod_raw.compute_adjustments(
        profile=profile, trust_mean=0.5, cognitive_load=0.5,
        emotion_label="focused", emotion_confidence=1.0
    )
    
    adj_smooth = mod_smooth.compute_adjustments(
        profile=profile, trust_mean=0.5, cognitive_load=0.5,
        emotion_label="focused", emotion_confidence=1.0
    )
    
    for param in profile.get_parameter_names():
        if adj_raw.deltas[param] != 0:
            assert abs(adj_smooth.deltas[param]) < abs(adj_raw.deltas[param]), \
                f"Smoothed delta {adj_smooth.deltas[param]} should be smaller than raw {adj_raw.deltas[param]}"
