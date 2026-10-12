"""
Project ECHO — Parameter Modulator
=====================================
Computes Dynamic Difficulty Adjustment (DDA) deltas for game
parameters based on the current belief state (trust, cognitive
load, emotion).

Emotion State Model (5 Active States + Neutral):
    - Neutral   → no parameter changes (hold)
    - Focused   → moderate challenge ramp  (+10% per cycle)
    - Joy       → gentle challenge ramp    (+5% per cycle)
    - Angry     → moderate difficulty reduction + moderate assistance (-10% per cycle)
    - Thinking  → reduce combat pressure, preserve/increase puzzle challenge (-5% combat, +3% puzzle)
    - Given-Up  → significant difficulty reduction + maximum assistance (-20% per cycle)

HCI Justification:
    The 5-state model provides granular control over DDA responses:
    - Focused/Joy produce positive ramps of different magnitudes to
      maintain flow without over-challenging (Csikszentmihalyi, 1990).
    - Angry applies moderate relief to prevent frustration spirals.
    - Thinking creates a "thinking space" by reducing combat pressure
      while preserving cognitive challenge (puzzles, decision-making).
    - Given-Up applies the strongest ease as the player has fully
      disengaged, preventing frustration/death loops.
    - Neutral produces zero deltas, keeping parameters stable when
      the player's state is indeterminate.

All deltas are bounded by ParameterSpec min/max constraints.
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from game_framework.game_registry import GameProfile, ParameterSpec


@dataclass
class ParameterAdjustment:
    """Result of a single parameter modulation step."""
    timestamp: float
    genre: str
    before: Dict[str, float]
    after: Dict[str, float]
    deltas: Dict[str, float]
    reason: str
    emotion: str
    cognitive_load: float
    trust_mean: float


class ParameterModulator:
    """
    Computes game parameter adjustments from belief states.

    Uses a set of DDA curves tuned per genre to produce smooth,
    bounded parameter changes that respond to the player's
    inferred psychological state.
    """

    # ------------------------------------------------------------------
    # Per-genre parameter classification
    # ------------------------------------------------------------------

    # Parameters whose value INCREASES when the game gets harder
    DIFFICULTY_INCREASE_PARAMS = {
        "left_4_dead": [
            "enemy_spawn_rate", "horde_frequency", "health_item_scarcity",
            "special_infected_cap", "wanderer_density",
        ],
        "alien_isolation": [
            "alien_menace_gauge", "tension_budget", "alien_search_radius",
            "noise_sensitivity", "vent_drop_probability",
        ],
        "resident_evil_4": [
            "enemy_accuracy", "action_score", "enemy_aggression",
            "damage_taken_mult",
        ],
        "silent_hill": [
            "world_decay_speed", "monster_type_bias", "puzzle_obfuscation",
            "static_interference",
        ],
    }

    # Parameters whose value INCREASES when the player needs help
    ASSIST_PARAMS = {
        "left_4_dead": [],
        "alien_isolation": [],
        "resident_evil_4": ["item_drop_quality"],
        "silent_hill": ["environment_lighting"],
    }

    # Parameters that should INCREASE during the "thinking" state
    # (cognitive challenge preserved while combat pressure drops)
    THINKING_INCREASE_PARAMS = {
        "left_4_dead": [],
        "alien_isolation": [],
        "resident_evil_4": ["action_score"],
        "silent_hill": ["puzzle_obfuscation"],
    }

    # Parameters that should remain UNCHANGED during the "thinking" state
    THINKING_HOLD_PARAMS = {
        "left_4_dead": [],
        "alien_isolation": [],
        "resident_evil_4": ["item_drop_quality"],
        "silent_hill": ["monster_type_bias"],
    }

    ENV_MUSIC = ["music_intensity"]
    ENV_LIGHT = ["lighting_brightness"]

    # ------------------------------------------------------------------
    # Base delta magnitudes per emotion state (fraction of param range)
    # ------------------------------------------------------------------
    STATE_DELTAS = {
        "neutral":  0.00,   # No changes
        "focused":  0.10,   # Moderate difficulty increase
        "joy":      0.05,   # Small difficulty increase
        "angry":   -0.10,   # Moderate difficulty decrease
        "thinking":-0.05,   # Moderate combat reduction
        "given_up":-0.20,   # Large difficulty decrease
    }

    # Assist parameter multipliers (positive = increase assistance)
    # These are applied in the OPPOSITE direction to difficulty
    ASSIST_DELTAS = {
        "neutral":  0.00,
        "focused": -0.05,   # Slight reduction in assistance
        "joy":     -0.025,  # Very slight reduction
        "angry":    0.10,   # Moderate assistance increase
        "thinking": 0.02,   # Slight assistance increase
        "given_up": 0.20,   # Maximum assistance increase
    }

    # Special delta for thinking-state cognitive params
    THINKING_COGNITIVE_DELTA = 0.03  # Slight increase for puzzle challenge

    # ------------------------------------------------------------------
    # Reason strings per state
    # ------------------------------------------------------------------
    STATE_REASONS = {
        "neutral":  "State is NEUTRAL -> Parameters held steady",
        "focused":  "State is FOCUSED -> Player is engaged, ramping up challenge!",
        "joy":      "State is JOY -> Player is enjoying, gentle challenge increase",
        "angry":    "Player is ANGRY (frustrated) -> Reducing difficulty, increasing assistance",
        "thinking": "Player is THINKING (hand on chin) -> Reducing combat pressure, preserving puzzle challenge",
        "given_up": "Player has GIVEN UP (no input 30s+) -> Dramatic ease + maximum assistance",
    }

    def __init__(self, smoothing_factor: float = 0.3):
        """
        Args:
            smoothing_factor: Exponential smoothing alpha for delta changes.
                             Lower = slower adaptation. Range [0.05, 1.0].
        """
        self.smoothing_factor = max(0.05, min(1.0, smoothing_factor))
        self._prev_deltas: Dict[str, Dict[str, float]] = {}

    def compute_adjustments(
        self,
        profile: GameProfile,
        trust_mean: float,
        cognitive_load: float,
        emotion_label: str,
        emotion_confidence: float,
        current_values: Optional[Dict[str, float]] = None,
        timestamp: float = 0.0,
    ) -> ParameterAdjustment:
        """
        Compute parameter deltas for a given belief state.

        Args:
            profile: The game genre profile.
            trust_mean: POMDP trust mean [0, 1].
            cognitive_load: Estimated cognitive load [0, 1].
            emotion_label: One of: neutral, focused, joy, angry, thinking, given_up.
            emotion_confidence: Confidence [0, 1].
            current_values: Current parameter values (defaults if None).
            timestamp: Current session time.

        Returns:
            ParameterAdjustment with before/after values and reasoning.
        """
        # NaN / Inf Safeguards (Fault Tolerance)
        if math.isnan(trust_mean) or math.isinf(trust_mean):
            trust_mean = 0.5
        if math.isnan(cognitive_load) or math.isinf(cognitive_load):
            cognitive_load = 0.5
        if math.isnan(emotion_confidence) or math.isinf(emotion_confidence):
            emotion_confidence = 0.0
            emotion_label = "neutral"

        if current_values is None:
            current_values = profile.get_defaults()

        genre = profile.genre

        # Normalize the emotion label for backward compatibility
        normalized_emotion = self._normalize_emotion(emotion_label)

        # Get the reason string
        reason = self.STATE_REASONS.get(
            normalized_emotion,
            f"State is {normalized_emotion.upper()} -> Default behavior",
        )

        # Compute raw deltas using the per-state policy
        raw_deltas = self._compute_raw_deltas(
            genre, profile, normalized_emotion, cognitive_load,
            trust_mean, emotion_confidence
        )

        # Apply exponential smoothing
        smoothed_deltas = self._smooth_deltas(genre, raw_deltas)

        # Apply deltas and clamp
        new_values = {}
        final_deltas = {}
        for param_name, spec in profile.parameters.items():
            old_val = current_values.get(param_name, spec.default)
            delta = smoothed_deltas.get(param_name, 0.0)
            
            # NaN/Inf check on delta
            if math.isnan(delta) or math.isinf(delta):
                delta = 0.0

            # Rate Limiting (Lipschitz Continuity Guarantee)
            param_range = spec.max_val - spec.min_val
            max_delta = 0.20 * param_range
            delta = max(-max_delta, min(max_delta, delta))

            new_val = spec.clamp(old_val + delta)
            new_values[param_name] = round(new_val, 4)
            final_deltas[param_name] = round(new_val - old_val, 4)

        return ParameterAdjustment(
            timestamp=timestamp,
            genre=genre,
            before=dict(current_values),
            after=new_values,
            deltas=final_deltas,
            reason=reason,
            emotion=normalized_emotion,
            cognitive_load=cognitive_load,
            trust_mean=trust_mean,
        )

    def _normalize_emotion(self, emotion_label: str) -> str:
        """
        Normalize emotion labels for backward compatibility.

        Maps legacy labels (normal, frustrated) to the canonical
        5-state model while passing through valid labels unchanged.
        """
        EMOTION_MAP = {
            # Canonical states (pass through)
            "neutral":  "neutral",
            "focused":  "focused",
            "joy":      "joy",
            "angry":    "angry",
            "thinking": "thinking",
            "given_up": "given_up",
            # Legacy / backward-compatible aliases
            "normal":     "neutral",
            "frustrated": "angry",
            "engaged":    "focused",
            "fatigued":   "thinking",
        }
        return EMOTION_MAP.get(emotion_label, "neutral")

    def _compute_raw_deltas(
        self,
        genre: str,
        profile: GameProfile,
        emotion: str,
        cognitive_load: float,
        trust_mean: float,
        emotion_confidence: float,
    ) -> Dict[str, float]:
        """
        Compute unsmoothed deltas based on the emotion state.

        Each parameter's delta is determined by:
        1. Its classification (difficulty / assist / thinking-cognitive / hold)
        2. The current emotion state's base magnitude
        3. Scaling by emotion confidence
        4. Scaling by the parameter's range (max - min)
        """
        deltas = {name: 0.0 for name in profile.parameters}

        # Neutral state: all deltas are zero
        if emotion == "neutral":
            return deltas

        # Get base magnitudes for this emotion
        diff_magnitude = self.STATE_DELTAS.get(emotion, 0.0)
        assist_magnitude = self.ASSIST_DELTAS.get(emotion, 0.0)

        # Scale by emotion confidence (minimum 0.5 to prevent zero-out)
        confidence_scale = max(0.5, emotion_confidence)
        diff_magnitude *= confidence_scale
        assist_magnitude *= confidence_scale

        # Get parameter classification lists for this genre
        diff_increase = self.DIFFICULTY_INCREASE_PARAMS.get(genre, [])
        assist_params = self.ASSIST_PARAMS.get(genre, [])
        thinking_increase = self.THINKING_INCREASE_PARAMS.get(genre, [])
        thinking_hold = self.THINKING_HOLD_PARAMS.get(genre, [])

        for param_name, spec in profile.parameters.items():
            param_range = spec.max_val - spec.min_val

            if emotion == "thinking":
                # Special handling for thinking state:
                # - Combat/pressure params: decrease
                # - Puzzle/cognitive params: increase
                # - Hold params: no change
                # - Assist params: slight increase
                if param_name in thinking_hold:
                    deltas[param_name] = 0.0
                elif param_name in thinking_increase:
                    # Increase cognitive challenge
                    deltas[param_name] = self.THINKING_COGNITIVE_DELTA * confidence_scale * param_range
                elif param_name in assist_params:
                    # Slight assistance increase
                    deltas[param_name] = assist_magnitude * param_range
                elif param_name in diff_increase:
                    # Decrease combat pressure
                    deltas[param_name] = diff_magnitude * param_range
                else:
                    # Unclassified params: apply gentle ease
                    deltas[param_name] = diff_magnitude * param_range * 0.5
            else:
                # Standard handling for focused/joy/angry/given_up
                if param_name in diff_increase:
                    # Difficulty params: positive magnitude = harder
                    deltas[param_name] = diff_magnitude * param_range
                elif param_name in assist_params:
                    # Assist params: positive assist_magnitude = more help
                    deltas[param_name] = assist_magnitude * param_range
                else:
                    # Unclassified params: apply half the difficulty delta
                    deltas[param_name] = diff_magnitude * param_range * 0.5

        return deltas

    def _smooth_deltas(
        self, genre: str, raw_deltas: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply exponential moving average smoothing to deltas."""
        if genre not in self._prev_deltas:
            self._prev_deltas[genre] = {k: 0.0 for k in raw_deltas}

        smoothed = {}
        alpha = self.smoothing_factor
        for param, raw in raw_deltas.items():
            prev = self._prev_deltas[genre].get(param, 0.0)
            smoothed[param] = alpha * raw + (1 - alpha) * prev

        self._prev_deltas[genre] = smoothed
        return smoothed

    def reset(self, genre: Optional[str] = None):
        """Reset smoothing state for one or all genres."""
        if genre:
            self._prev_deltas.pop(genre, None)
        else:
            self._prev_deltas.clear()
