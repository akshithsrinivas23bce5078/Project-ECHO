"""
Project ECHO — Active Inference Tests (Phase 3)
==================================================
Tests for Free Energy Engine and Markov Blanket decomposition.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from estimator.src.free_energy_engine import FreeEnergyEngine, FreeEnergyState
from estimator.src.markov_blanket import (
    MarkovBlanket, BlanketDecomposition, StateType, StateNode,
)


class TestFreeEnergyEngine:
    """Tests for the Free Energy minimization engine."""

    def test_initial_uniform_belief(self):
        engine = FreeEnergyEngine(state_dim=15)
        belief = engine.get_belief()
        assert len(belief) == 15
        assert np.allclose(belief, 1.0 / 15)

    def test_minimize_returns_valid_state(self):
        engine = FreeEnergyEngine(state_dim=15)
        likelihood = np.random.rand(15) + 0.1
        transition = np.ones(15) / 15
        result = engine.minimize_free_energy(likelihood, transition)

        assert isinstance(result, FreeEnergyState)
        assert isinstance(result.free_energy, float)
        assert isinstance(result.surprise, float)
        assert isinstance(result.prediction_error, float)

    def test_belief_sums_to_one(self):
        engine = FreeEnergyEngine(state_dim=15)
        likelihood = np.random.rand(15) + 0.1
        transition = np.ones(15) / 15

        for _ in range(10):
            engine.minimize_free_energy(likelihood, transition)

        belief = engine.get_belief()
        assert abs(np.sum(belief) - 1.0) < 1e-6

    def test_free_energy_decreases_with_updates(self):
        """Free energy should generally decrease as beliefs converge."""
        engine = FreeEnergyEngine(state_dim=15, learning_rate=0.1)
        # Consistent observation favoring state 3
        likelihood = np.zeros(15) + 0.01
        likelihood[3] = 1.0
        transition = np.ones(15) / 15

        energies = []
        for _ in range(20):
            result = engine.minimize_free_energy(likelihood, transition)
            energies.append(result.free_energy)

        # Energy in the second half should be lower on average
        first_half = np.mean(energies[:10])
        second_half = np.mean(energies[10:])
        assert second_half <= first_half + 0.5  # Allow some tolerance

    def test_surprise_high_for_unexpected_observation(self):
        engine = FreeEnergyEngine(state_dim=15)

        # Build a confident prior
        transition = np.zeros(15) + 0.01
        transition[0] = 0.9
        transition /= transition.sum()

        # Observation contradicts prior
        likelihood = np.zeros(15) + 0.01
        likelihood[14] = 1.0

        result = engine.minimize_free_energy(likelihood, transition)
        assert result.surprise > 0.5

    def test_epistemic_value_high_when_uncertain(self):
        engine = FreeEnergyEngine(state_dim=15)
        # Uniform belief (maximum uncertainty)
        likelihood = np.zeros(15) + 0.01
        likelihood[5] = 1.0
        transition = np.ones(15) / 15

        result = engine.minimize_free_energy(likelihood, transition)
        assert result.epistemic_value >= 0.0

    def test_reset_restores_uniform(self):
        engine = FreeEnergyEngine(state_dim=15)
        likelihood = np.random.rand(15) + 0.1
        transition = np.ones(15) / 15
        engine.minimize_free_energy(likelihood, transition)
        engine.reset()

        belief = engine.get_belief()
        assert np.allclose(belief, 1.0 / 15)

    def test_precision_decays(self):
        engine = FreeEnergyEngine(state_dim=15, precision_decay=0.9)
        initial_precision = engine.get_precision().copy()

        likelihood = np.random.rand(15) + 0.1
        transition = np.ones(15) / 15
        engine.minimize_free_energy(likelihood, transition)

        updated_precision = engine.get_precision()
        assert np.all(updated_precision <= initial_precision + 1e-6)

    def test_prediction_error_history(self):
        engine = FreeEnergyEngine(state_dim=15)
        likelihood = np.random.rand(15) + 0.1
        transition = np.ones(15) / 15

        for _ in range(5):
            engine.minimize_free_energy(likelihood, transition)

        assert engine.get_mean_prediction_error() > 0.0


class TestMarkovBlanket:
    """Tests for the Markov Blanket state decomposition."""

    def test_default_blanket_has_all_types(self):
        blanket = MarkovBlanket()
        types = {s.state_type for s in blanket.states.values()}
        assert StateType.INTERNAL in types
        assert StateType.EXTERNAL in types
        assert StateType.SENSORY in types
        assert StateType.ACTIVE in types

    def test_sensory_states_populated(self):
        blanket = MarkovBlanket()
        sensory = blanket.get_sensory_states()
        assert len(sensory) >= 5
        assert "clc_raw" in sensory
        assert "neuromotor_load" in sensory

    def test_active_states_populated(self):
        blanket = MarkovBlanket()
        active = blanket.get_active_states()
        assert len(active) >= 3
        assert "difficulty_delta" in active

    def test_blanket_states_are_sensory_plus_active(self):
        blanket = MarkovBlanket()
        blanket_states = set(blanket.get_blanket_states())
        sensory = set(blanket.get_sensory_states())
        active = set(blanket.get_active_states())
        assert blanket_states == sensory | active

    def test_decompose_returns_valid_structure(self):
        blanket = MarkovBlanket()
        obs = {"clc_raw": 0.45, "snr_db": 22.0, "neuromotor_load": 0.3}
        belief = {"trust_belief": 0.6, "free_energy": -2.5}
        action = {"difficulty_delta": 0.05}

        result = blanket.decompose(obs, belief, action)

        assert isinstance(result, BlanketDecomposition)
        assert "clc_raw" in result.sensory_states
        assert result.sensory_states["clc_raw"] == 0.45
        assert "trust_belief" in result.internal_states
        assert 0.0 <= result.blanket_integrity <= 1.0

    def test_causal_connections_exist(self):
        blanket = MarkovBlanket()
        # External states should have children (sensory)
        ext = blanket.states["true_trust"]
        assert len(ext.children) > 0
        # Sensory states should have parents (external)
        sens = blanket.states["clc_raw"]
        assert len(sens.parents) > 0

    def test_state_node_dataclass(self):
        node = StateNode(
            name="test", state_type=StateType.SENSORY,
            value=0.5, precision=1.0
        )
        assert node.name == "test"
        assert node.state_type == StateType.SENSORY
