"""
Project ECHO — Game Session Simulator
========================================
Simulates a complete game session by generating synthetic player
telemetry (trust, cognitive load, emotion) that evolves over time,
and records all DDA parameter adjustments at each timestep.

Produces a Pandas DataFrame (SessionLog) suitable for analysis
and visualization.

Usage:
    simulator = SessionSimulator()
    df = simulator.simulate("fps", duration_seconds=1800)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from game_framework.game_registry import GameRegistry, GameProfile
from game_framework.parameter_modulator import ParameterModulator, ParameterAdjustment


# Emotion transition probabilities conditioned on cognitive load
# Maps (cognitive_load_bucket) → transition weights for each emotion
EMOTION_TRANSITIONS = {
    "high_load": {
        "frustrated": 0.35, "focused": 0.25, "fatigued": 0.20,
        "neutral": 0.10, "engaged": 0.10,
    },
    "medium_load": {
        "frustrated": 0.10, "focused": 0.30, "fatigued": 0.10,
        "neutral": 0.20, "engaged": 0.30,
    },
    "low_load": {
        "frustrated": 0.05, "focused": 0.15, "fatigued": 0.25,
        "neutral": 0.35, "engaged": 0.20,
    },
}

EMOTION_LABELS = ["frustrated", "focused", "fatigued", "neutral", "engaged"]


class SessionSimulator:
    """
    Generates synthetic game sessions with evolving player states.

    Produces a DataFrame with columns:
        - time: session timestamp (seconds)
        - trust_mean, trust_variance, cognitive_load, clc_raw
        - emotion_label, emotion_confidence
        - <param_name>: current value for each game parameter
        - <param_name>_delta: adjustment delta for each parameter
        - dda_action: the DDA action class (ease/hold/challenge)
    """

    def __init__(self, registry: Optional[GameRegistry] = None,
                 seed: int = 42):
        self.registry = registry or GameRegistry()
        self.modulator = ParameterModulator(smoothing_factor=0.3)
        self.rng = np.random.RandomState(seed)

    def simulate(
        self,
        genre: str,
        duration_seconds: int = 1800,
        sample_rate_hz: float = 1.0,
        scenario: str = "natural",
    ) -> pd.DataFrame:
        """
        Simulate a game session.

        Args:
            genre: Game genre key (e.g., "fps", "puzzle").
            duration_seconds: Session length in seconds.
            sample_rate_hz: Belief updates per second.
            scenario: Simulation scenario:
                - "natural": organic state evolution
                - "struggle": player gradually struggles
                - "mastery": player gradually masters
                - "volatile": rapid state changes

        Returns:
            DataFrame with all telemetry and parameter values over time.
        """
        profile = self.registry.get_profile(genre)
        dt = 1.0 / sample_rate_hz
        n_steps = int(duration_seconds * sample_rate_hz)

        # Initialize state
        trust = 0.5
        cognitive_load = 0.4
        clc_raw = 0.45
        emotion_idx = 3  # neutral
        current_params = profile.get_defaults()

        records = []

        for step in range(n_steps):
            t = step * dt

            # Evolve player state
            trust, cognitive_load, clc_raw, emotion_idx = self._evolve_state(
                trust, cognitive_load, clc_raw, emotion_idx, t,
                duration_seconds, scenario
            )

            emotion_label = EMOTION_LABELS[emotion_idx]
            emotion_confidence = self._generate_emotion_confidence(emotion_idx)
            trust_variance = 0.05 + 0.15 * (1.0 - trust)  # Higher variance at low trust

            # Compute DDA adjustments
            adjustment = self.modulator.compute_adjustments(
                profile=profile,
                trust_mean=trust,
                cognitive_load=cognitive_load,
                emotion_label=emotion_label,
                emotion_confidence=emotion_confidence,
                current_values=current_params,
                timestamp=t,
            )

            # Build record
            record = {
                "time": round(t, 2),
                "trust_mean": round(trust, 4),
                "trust_variance": round(trust_variance, 4),
                "cognitive_load": round(cognitive_load, 4),
                "clc_raw": round(clc_raw, 4),
                "emotion_label": emotion_label,
                "emotion_confidence": round(emotion_confidence, 4),
                "dda_action": adjustment.reason.split("→")[0].strip()
                if "→" in adjustment.reason else adjustment.reason,
            }

            # Add parameter values and deltas
            for param_name in profile.get_parameter_names():
                record[param_name] = adjustment.after.get(param_name, 0.0)
                record[f"{param_name}_delta"] = adjustment.deltas.get(param_name, 0.0)

            records.append(record)
            current_params = adjustment.after

        self.modulator.reset(genre)
        return pd.DataFrame(records)

    def simulate_all_genres(
        self, duration_seconds: int = 1800, **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Simulate sessions for all registered game genres."""
        results = {}
        for genre in self.registry.list_genres():
            results[genre] = self.simulate(genre, duration_seconds, **kwargs)
        return results

    # ------------------------------------------------------------------
    # State evolution
    # ------------------------------------------------------------------

    def _evolve_state(
        self, trust: float, cogload: float, clc: float,
        emotion_idx: int, t: float, total_t: float, scenario: str,
    ):
        """Evolve player state by one timestep using scenario-driven dynamics."""
        progress = t / total_t  # 0 → 1 over session

        if scenario == "struggle":
            # Trust decays, cognitive load increases
            trust_drift = -0.0005 * progress
            cogload_drift = 0.0008 * progress
        elif scenario == "mastery":
            # Trust grows, cognitive load decreases
            trust_drift = 0.0006 * progress
            cogload_drift = -0.0005 * progress
        elif scenario == "volatile":
            # Rapid oscillations
            trust_drift = 0.003 * np.sin(2 * np.pi * t / 120)
            cogload_drift = 0.003 * np.cos(2 * np.pi * t / 90)
        else:  # natural
            # Gentle random walk with mean-reversion
            trust_drift = 0.001 * (0.5 - trust)
            cogload_drift = 0.001 * (0.45 - cogload)

        # Add noise
        trust_noise = self.rng.normal(0, 0.015)
        cogload_noise = self.rng.normal(0, 0.02)

        trust = np.clip(trust + trust_drift + trust_noise, 0.05, 0.95)
        cogload = np.clip(cogload + cogload_drift + cogload_noise, 0.05, 0.95)

        # CLC tracks cognitive load with noise
        clc = np.clip(0.3 * clc + 0.7 * (0.35 + cogload * 0.3) + self.rng.normal(0, 0.03), 0.1, 0.9)

        # Emotion transition
        if self.rng.random() < 0.08:  # 8% chance of emotion change per step
            emotion_idx = self._sample_emotion(cogload)

        return trust, cogload, clc, emotion_idx

    def _sample_emotion(self, cognitive_load: float) -> int:
        """Sample an emotion label based on cognitive load."""
        if cognitive_load > 0.65:
            bucket = "high_load"
        elif cognitive_load > 0.35:
            bucket = "medium_load"
        else:
            bucket = "low_load"

        weights = EMOTION_TRANSITIONS[bucket]
        probs = [weights[label] for label in EMOTION_LABELS]
        return int(self.rng.choice(len(EMOTION_LABELS), p=probs))

    def _generate_emotion_confidence(self, emotion_idx: int) -> float:
        """Generate a plausible DSTER confidence score."""
        # Base confidence depends on emotion (some are harder to detect)
        base = {0: 0.65, 1: 0.75, 2: 0.60, 3: 0.55, 4: 0.70}[emotion_idx]
        noise = self.rng.normal(0, 0.1)
        return float(np.clip(base + noise, 0.3, 0.95))
