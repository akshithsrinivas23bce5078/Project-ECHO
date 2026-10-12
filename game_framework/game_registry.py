"""
Project ECHO — Game Registry
===============================
Defines game profiles and their tunable parameter sets for
Dynamic Difficulty Adjustment (DDA).

Each game genre has a distinct set of parameters that the
adaptation engine can modulate based on the player's
inferred trust, cognitive load, and emotional state.

Supported genres:
    - FPS (First-Person Shooter)
    - Puzzle
    - Strategy (Real-Time Strategy)
    - Racing
    - Survival Horror
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ParameterSpec:
    """Specification for a single tunable game parameter."""
    name: str
    default: float
    min_val: float
    max_val: float
    unit: str

    def clamp(self, value: float) -> float:
        """Clamp a value to the parameter's valid range."""
        return max(self.min_val, min(self.max_val, value))

    def normalize(self, value: float) -> float:
        """Normalize a value to [0, 1] within the parameter range."""
        span = self.max_val - self.min_val
        if span < 1e-12:
            return 0.5
        return (value - self.min_val) / span

    def denormalize(self, norm_value: float) -> float:
        """Convert a [0, 1] value back to the parameter's range."""
        return self.min_val + norm_value * (self.max_val - self.min_val)


@dataclass
class GameProfile:
    """
    Complete profile for a game genre, including its display name
    and all tunable parameters with their ranges.
    """
    genre: str
    display_name: str
    parameters: Dict[str, ParameterSpec] = field(default_factory=dict)

    def get_defaults(self) -> Dict[str, float]:
        """Return default values for all parameters."""
        return {name: spec.default for name, spec in self.parameters.items()}

    def get_parameter_names(self) -> List[str]:
        """Return sorted list of parameter names."""
        return sorted(self.parameters.keys())

    def validate_values(self, values: Dict[str, float]) -> Dict[str, float]:
        """Clamp all values to their valid ranges."""
        clamped = {}
        for name, val in values.items():
            if name in self.parameters:
                clamped[name] = self.parameters[name].clamp(val)
        return clamped


class GameRegistry:
    """
    Central registry of all supported game genres and their profiles.

    Can be loaded from echo_config.yaml or constructed programmatically.
    """

    def __init__(self):
        self._profiles: Dict[str, GameProfile] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load game profiles from echo_config.yaml."""
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'echo_config.yaml'
        )
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            games_config = config.get('games', {}).get('profiles', {})
            for genre, genre_data in games_config.items():
                profile = GameProfile(
                    genre=genre,
                    display_name=genre_data.get('display_name', genre),
                )
                for param_name, param_data in genre_data.get('parameters', {}).items():
                    profile.parameters[param_name] = ParameterSpec(
                        name=param_name,
                        default=float(param_data['default']),
                        min_val=float(param_data['min']),
                        max_val=float(param_data['max']),
                        unit=param_data.get('unit', 'normalized'),
                    )
                self._profiles[genre] = profile
        else:
            self._build_hardcoded_profiles()

    def _build_hardcoded_profiles(self):
        """Fallback: build profiles programmatically if config not found."""
        # 1. Left 4 Dead
        l4d = GameProfile(genre="left_4_dead", display_name="Left 4 Dead (AI Director)")
        l4d.parameters = {
            "enemy_spawn_rate": ParameterSpec("enemy_spawn_rate", 0.5, 0.1, 1.0, "normalized"),
            "horde_frequency": ParameterSpec("horde_frequency", 0.5, 0.1, 1.0, "normalized"),
            "health_item_scarcity": ParameterSpec("health_item_scarcity", 0.5, 0.2, 0.9, "normalized"),
            "special_infected_cap": ParameterSpec("special_infected_cap", 3.0, 1.0, 6.0, "count"),
            "wanderer_density": ParameterSpec("wanderer_density", 0.5, 0.1, 1.0, "normalized"),
        }
        self._profiles["left_4_dead"] = l4d

        # 2. Alien: Isolation
        alien = GameProfile(genre="alien_isolation", display_name="Alien: Isolation (Menace Gauge)")
        alien.parameters = {
            "alien_menace_gauge": ParameterSpec("alien_menace_gauge", 0.5, 0.0, 1.0, "normalized"),
            "tension_budget": ParameterSpec("tension_budget", 1.0, 0.2, 2.0, "multiplier"),
            "alien_search_radius": ParameterSpec("alien_search_radius", 0.5, 0.1, 1.0, "normalized"),
            "vent_drop_probability": ParameterSpec("vent_drop_probability", 0.3, 0.05, 0.8, "probability"),
            "noise_sensitivity": ParameterSpec("noise_sensitivity", 0.5, 0.1, 1.0, "normalized"),
        }
        self._profiles["alien_isolation"] = alien

        # 3. Resident Evil 4
        re4 = GameProfile(genre="resident_evil_4", display_name="Resident Evil 4 (Action Scoring)")
        re4.parameters = {
            "enemy_accuracy": ParameterSpec("enemy_accuracy", 0.6, 0.1, 1.0, "normalized"),
            "action_score": ParameterSpec("action_score", 5.0, 1.0, 10.0, "level"),
            "enemy_aggression": ParameterSpec("enemy_aggression", 0.5, 0.2, 1.0, "normalized"),
            "item_drop_quality": ParameterSpec("item_drop_quality", 0.5, 0.1, 1.0, "normalized"),
            "damage_taken_mult": ParameterSpec("damage_taken_mult", 1.0, 0.5, 2.0, "multiplier"),
        }
        self._profiles["resident_evil_4"] = re4

        # 4. Silent Hill: Shattered Memories
        sh = GameProfile(genre="silent_hill", display_name="Silent Hill: Shattered Memories")
        sh.parameters = {
            "world_decay_speed": ParameterSpec("world_decay_speed", 0.5, 0.1, 1.0, "normalized"),
            "monster_type_bias": ParameterSpec("monster_type_bias", 0.5, 0.0, 1.0, "normalized"),
            "puzzle_obfuscation": ParameterSpec("puzzle_obfuscation", 0.5, 0.1, 1.0, "normalized"),
            "environment_lighting": ParameterSpec("environment_lighting", 0.5, 0.1, 1.0, "normalized"),
            "static_interference": ParameterSpec("static_interference", 0.5, 0.0, 1.0, "normalized"),
        }
        self._profiles["silent_hill"] = sh

    def get_profile(self, genre: str) -> GameProfile:
        """Get a game profile by genre key."""
        if genre not in self._profiles:
            raise ValueError(
                f"Unknown game genre '{genre}'. "
                f"Available: {list(self._profiles.keys())}"
            )
        return self._profiles[genre]

    def list_genres(self) -> List[str]:
        """Return all registered genre keys."""
        return list(self._profiles.keys())

    def list_profiles(self) -> List[GameProfile]:
        """Return all registered profiles."""
        return list(self._profiles.values())

    def get_all_parameter_names(self) -> Dict[str, List[str]]:
        """Return parameter names per genre."""
        return {
            genre: profile.get_parameter_names()
            for genre, profile in self._profiles.items()
        }
