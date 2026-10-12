"""
Project ECHO — POMDP Convergence Tests
=========================================
Validates Bayesian belief state convergence, transition model
properties, and observation model correctness.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

from estimator.src.pomdp_model import (
    POMDPModel, NUM_STATES, NUM_TRUST, NUM_COGLOAD,
    TRUST_LEVELS, COGLOAD_LEVELS, state_index, state_from_index,
)


class TestTransitionModel:
    """Validates transition matrix properties."""

    def test_transition_rows_sum_to_one(self):
        """Each row of T must sum to 1.0 (valid probability distribution)."""
        pomdp = POMDPModel()
        row_sums = pomdp.transition.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_transition_non_negative(self):
        """All transition probabilities must be non-negative."""
        pomdp = POMDPModel()
        assert np.all(pomdp.transition >= 0), "Negative transition probabilities found"

    def test_transition_shape(self):
        """Transition matrix must be (15, 15)."""
        pomdp = POMDPModel()
        assert pomdp.transition.shape == (NUM_STATES, NUM_STATES)

    def test_self_transition_is_highest(self):
        """The self-transition probability should be the highest per row."""
        pomdp = POMDPModel()
        for s in range(NUM_STATES):
            assert pomdp.transition[s, s] == pomdp.transition[s].max(), (
                f"State {s}: self-transition is not maximum"
            )


class TestBeliefState:
    """Validates belief state initialization and updates."""

    def test_initial_belief_is_uniform(self):
        """Initial belief must be uniform over all 15 states."""
        pomdp = POMDPModel()
        expected = np.ones(NUM_STATES) / NUM_STATES
        np.testing.assert_allclose(pomdp.belief, expected)

    def test_belief_sums_to_one(self):
        """Belief state must always sum to 1.0."""
        pomdp = POMDPModel()
        assert abs(pomdp.belief.sum() - 1.0) < 1e-10

    def test_belief_sums_to_one_after_update(self):
        """Belief must sum to 1.0 after updates."""
        pomdp = POMDPModel()
        for _ in range(20):
            pomdp.update_belief({
                "clc_raw": np.random.uniform(0.2, 0.7),
                "iki_mean_ms": np.random.uniform(100, 400),
                "snr_db": np.random.uniform(15, 35),
            })
            assert abs(pomdp.belief.sum() - 1.0) < 1e-10, (
                f"Belief sums to {pomdp.belief.sum()}"
            )

    def test_belief_non_negative(self):
        """Belief probabilities must always be non-negative."""
        pomdp = POMDPModel()
        for _ in range(20):
            pomdp.update_belief({
                "clc_raw": np.random.uniform(0.2, 0.7),
                "iki_mean_ms": np.random.uniform(100, 400),
                "snr_db": np.random.uniform(15, 35),
            })
            assert np.all(pomdp.belief >= 0), "Negative belief values found"

    def test_reset_restores_uniform(self):
        """Reset must restore uniform belief."""
        pomdp = POMDPModel()
        pomdp.update_belief({"clc_raw": 0.6, "iki_mean_ms": 300, "snr_db": 20})
        pomdp.reset()
        expected = np.ones(NUM_STATES) / NUM_STATES
        np.testing.assert_allclose(pomdp.belief, expected)


class TestBeliefConvergence:
    """Validates that consistent observations drive belief convergence."""

    def test_high_load_observations_increase_cogload_belief(self):
        """
        Consistent high-cognitive-load observations should increase
        the marginal probability of high cognitive load states.
        """
        pomdp = POMDPModel()
        high_load_obs = {"clc_raw": 0.60, "iki_mean_ms": 350.0, "snr_db": 24.0}

        for _ in range(50):
            pomdp.update_belief(high_load_obs)

        summary = pomdp.get_belief_summary()
        assert summary["cognitive_load"] > 0.6, (
            f"Cognitive load {summary['cognitive_load']} should be > 0.6 "
            "after 50 high-load observations"
        )

    def test_low_load_observations_decrease_cogload_belief(self):
        """
        Consistent low-load observations should decrease cognitive load.
        """
        pomdp = POMDPModel()
        low_load_obs = {"clc_raw": 0.30, "iki_mean_ms": 120.0, "snr_db": 28.0}

        for _ in range(50):
            pomdp.update_belief(low_load_obs)

        summary = pomdp.get_belief_summary()
        assert summary["cognitive_load"] < 0.4, (
            f"Cognitive load {summary['cognitive_load']} should be < 0.4 "
            "after 50 low-load observations"
        )

    def test_high_trust_observations_increase_trust(self):
        """
        Observations consistent with high trust should raise trust_mean.
        """
        pomdp = POMDPModel()
        # High SNR and moderate other metrics suggest high trust + smooth motor control
        high_trust_obs = {"clc_raw": 0.45, "iki_mean_ms": 200.0, "snr_db": 30.0}

        for _ in range(50):
            pomdp.update_belief(high_trust_obs)

        summary = pomdp.get_belief_summary()
        assert summary["trust_mean"] > 0.5, (
            f"Trust mean {summary['trust_mean']} should be > 0.5 "
            "after high-trust observations"
        )


class TestStateMapping:
    """Validates state index/label mapping functions."""

    def test_state_index_roundtrip(self):
        """state_index and state_from_index must be inverses."""
        for t in range(NUM_TRUST):
            for c in range(NUM_COGLOAD):
                idx = state_index(t, c)
                trust_label, cogload_label = state_from_index(idx)
                assert trust_label == TRUST_LEVELS[t]
                assert cogload_label == COGLOAD_LEVELS[c]

    def test_total_states(self):
        """Total states must equal 15."""
        assert NUM_STATES == 15
        assert NUM_TRUST * NUM_COGLOAD == 15


class TestObservationModel:
    """Validates observation likelihood computation."""

    def test_likelihood_non_negative(self):
        """Observation likelihood must be non-negative for all states."""
        pomdp = POMDPModel()
        obs = {"clc_raw": 0.45, "iki_mean_ms": 200.0, "snr_db": 24.0}
        likelihood = pomdp._observation_likelihood(obs)
        assert np.all(likelihood >= 0)

    def test_likelihood_not_all_zero(self):
        """Likelihood should not be all zeros for reasonable observations."""
        pomdp = POMDPModel()
        obs = {"clc_raw": 0.45, "iki_mean_ms": 200.0, "snr_db": 24.0}
        likelihood = pomdp._observation_likelihood(obs)
        assert np.sum(likelihood) > 0

    def test_gaussian_pdf_correctness(self):
        """Gaussian PDF must produce known value at mean."""
        val = POMDPModel._gaussian_pdf(0.0, 0.0, 1.0)
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert abs(val - expected) < 1e-10
