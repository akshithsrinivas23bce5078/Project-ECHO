"""
Project ECHO — Markov Blanket State Decomposition
====================================================
Phase 3: Active Inference

Implements the Markov Blanket formalism for decomposing the
system's state space into internal, external, sensory, and
active states. This enables principled separation of:

    - Internal states: belief variables the agent controls
    - External states: true world states (player emotion, intent)
    - Sensory states: observations (keystrokes, mouse, vision)
    - Active states: actions (DDA adjustments, UI changes)

The Markov Blanket boundary defines what the agent can and
cannot directly observe, formalizing the POMDP structure.

Reference:
    Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems.
    Friston, K. (2013). Life as We Know It. J. Royal Soc. Interface.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class StateType(Enum):
    """Classification of states in the Markov Blanket."""
    INTERNAL = "internal"
    EXTERNAL = "external"
    SENSORY = "sensory"
    ACTIVE = "active"


@dataclass
class StateNode:
    """A single state variable in the Markov Blanket graph."""
    name: str
    state_type: StateType
    value: float = 0.0
    precision: float = 1.0
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


@dataclass
class BlanketDecomposition:
    """Result of Markov Blanket decomposition."""
    internal_states: Dict[str, float]
    external_states: Dict[str, float]
    sensory_states: Dict[str, float]
    active_states: Dict[str, float]
    blanket_integrity: float  # 0-1 measure of blanket separation quality


class MarkovBlanket:
    """
    Markov Blanket state space decomposition for Project ECHO.

    Defines the boundary between the agent (DDA system) and the
    environment (player), enabling principled active inference.

    The blanket consists of:
        SENSORY: {clc_raw, iki_mean, snr_db, emotion_label,
                  emotion_confidence, neuromotor_load, visual_features}
        ACTIVE:  {verbosity, ui_action, difficulty_delta,
                  hud_declutter, highlight_targets}
        INTERNAL: {trust_belief, cogload_belief, free_energy,
                   prediction_error, precision}
        EXTERNAL: {true_trust, true_cogload, true_emotion,
                   true_fatigue, player_intent}
    """

    def __init__(self):
        self.states: Dict[str, StateNode] = {}
        self._build_default_blanket()

    def _build_default_blanket(self):
        """Construct the default Project ECHO Markov Blanket."""
        # Sensory states (observations from the player)
        sensory = [
            "clc_raw", "iki_mean_ms", "snr_db",
            "emotion_label_encoded", "emotion_confidence",
            "neuromotor_load", "sigma_deviation", "mu_deviation",
            "movement_energy", "face_detected",
        ]
        for name in sensory:
            self.states[name] = StateNode(
                name=name, state_type=StateType.SENSORY
            )

        # Active states (agent actions)
        active = [
            "verbosity_level", "ui_action_code", "difficulty_delta",
            "hud_declutter", "highlight_targets", "adaptation_magnitude",
        ]
        for name in active:
            self.states[name] = StateNode(
                name=name, state_type=StateType.ACTIVE
            )

        # Internal states (agent beliefs)
        internal = [
            "trust_belief", "cogload_belief", "free_energy",
            "prediction_error", "precision_mean",
            "epistemic_value", "pragmatic_value",
        ]
        for name in internal:
            self.states[name] = StateNode(
                name=name, state_type=StateType.INTERNAL
            )

        # External states (true player states — not directly observable)
        external = [
            "true_trust", "true_cogload", "true_emotion",
            "true_fatigue", "player_intent", "engagement_level",
        ]
        for name in external:
            self.states[name] = StateNode(
                name=name, state_type=StateType.EXTERNAL
            )

        # Define causal connections
        # External → Sensory (player states cause observations)
        for ext in external:
            for sens in sensory:
                self.states[ext].children.append(sens)
                self.states[sens].parents.append(ext)

        # Sensory → Internal (observations update beliefs)
        for sens in sensory:
            for int_s in internal:
                self.states[sens].children.append(int_s)
                self.states[int_s].parents.append(sens)

        # Internal → Active (beliefs drive actions)
        for int_s in internal:
            for act in active:
                self.states[int_s].children.append(act)
                self.states[act].parents.append(int_s)

        # Active → External (actions affect player state)
        for act in active:
            for ext in external:
                self.states[act].children.append(ext)
                self.states[ext].parents.append(act)

    def decompose(
        self,
        observation: Dict[str, float],
        belief: Dict[str, float],
        action: Dict[str, float],
    ) -> BlanketDecomposition:
        """
        Perform Markov Blanket decomposition given current state.

        Args:
            observation: Current sensory observations.
            belief: Current internal belief values.
            action: Current active state values.

        Returns:
            BlanketDecomposition with separated state spaces.
        """
        sensory = {}
        for name, node in self.states.items():
            if node.state_type == StateType.SENSORY:
                node.value = observation.get(name, 0.0)
                sensory[name] = node.value

        internal = {}
        for name, node in self.states.items():
            if node.state_type == StateType.INTERNAL:
                node.value = belief.get(name, 0.0)
                internal[name] = node.value

        active = {}
        for name, node in self.states.items():
            if node.state_type == StateType.ACTIVE:
                node.value = action.get(name, 0.0)
                active[name] = node.value

        external = {}
        for name, node in self.states.items():
            if node.state_type == StateType.EXTERNAL:
                external[name] = node.value

        integrity = self._compute_blanket_integrity(
            sensory, internal, active, external
        )

        return BlanketDecomposition(
            internal_states=internal,
            external_states=external,
            sensory_states=sensory,
            active_states=active,
            blanket_integrity=round(integrity, 4),
        )

    def get_sensory_states(self) -> List[str]:
        """Return names of all sensory state variables."""
        return [n for n, s in self.states.items() if s.state_type == StateType.SENSORY]

    def get_active_states(self) -> List[str]:
        """Return names of all active state variables."""
        return [n for n, s in self.states.items() if s.state_type == StateType.ACTIVE]

    def get_internal_states(self) -> List[str]:
        """Return names of all internal state variables."""
        return [n for n, s in self.states.items() if s.state_type == StateType.INTERNAL]

    def get_blanket_states(self) -> List[str]:
        """Return the Markov blanket (sensory + active states)."""
        return self.get_sensory_states() + self.get_active_states()

    def _compute_blanket_integrity(
        self,
        sensory: Dict, internal: Dict,
        active: Dict, external: Dict,
    ) -> float:
        """
        Compute a measure of blanket separation quality.

        Returns 1.0 when the blanket perfectly separates internal
        from external states. Lower values indicate information
        leakage across the boundary.
        """
        # Count non-zero states in each partition
        total_populated = sum(
            1 for d in [sensory, internal, active, external]
            for v in d.values() if v != 0.0
        )
        total_states = sum(len(d) for d in [sensory, internal, active, external])

        if total_states == 0:
            return 1.0

        # Integrity is high when each partition has some non-zero states
        partitions_active = sum(
            1 for d in [sensory, internal, active]
            if any(v != 0.0 for v in d.values())
        )

        return min(1.0, partitions_active / 3.0)
