"""
Project ECHO — Free Energy Engine
=====================================
Phase 3: Active Inference

Implements the Free Energy Principle (FEP) for autonomous belief
optimization. The agent minimizes variational free energy F by:

    F = D_KL[q(s) || p(s)] - E_q[ln p(o|s)]

Where:
    q(s) = approximate posterior (current belief)
    p(s) = prior (generative model prediction)
    p(o|s) = likelihood of observations given state

This replaces brute-force Bayesian updates with a gradient-descent
approach on the free energy functional, enabling the system to
actively seek information (epistemic action) and reduce surprise.

Reference:
    Friston, K. (2010). The Free-Energy Principle: A Unified Brain
    Theory? Nature Reviews Neuroscience 11(2), 127-138.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FreeEnergyState:
    """Snapshot of the free energy computation."""
    free_energy: float = 0.0
    surprise: float = 0.0          # -ln p(o) — sensory surprise
    complexity: float = 0.0        # D_KL[q || p] — divergence from prior
    accuracy: float = 0.0          # E_q[ln p(o|s)] — expected log-likelihood
    prediction_error: float = 0.0  # Difference between predicted and actual
    epistemic_value: float = 0.0   # Information gain from action
    pragmatic_value: float = 0.0   # Goal-directed utility


class FreeEnergyEngine:
    """
    Variational Free Energy minimization for active inference.

    Instead of passive Bayesian filtering (observe → update),
    this engine actively minimizes free energy by:
        1. Predicting expected observations from the generative model
        2. Computing prediction error (surprise)
        3. Updating beliefs to minimize divergence
        4. Computing epistemic value for action selection

    Parameters:
        state_dim: Number of latent states.
        learning_rate: Step size for free energy gradient descent.
        precision_decay: How quickly precision (inverse variance) decays.
        prior_strength: Weight of the prior in the free energy bound.
    """

    def __init__(
        self,
        state_dim: int = 15,
        learning_rate: float = 0.05,
        precision_decay: float = 0.98,
        prior_strength: float = 0.3,
    ):
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.precision_decay = precision_decay
        self.prior_strength = prior_strength

        # Belief (approximate posterior q)
        self.q_belief = np.ones(state_dim) / state_dim

        # Prior (generative model prediction p)
        self.p_prior = np.ones(state_dim) / state_dim

        # Precision (inverse variance) — tracks certainty
        self.precision = np.ones(state_dim)

        # History for temporal smoothing
        self._prediction_errors: list = []
        self._max_history = 50

    def minimize_free_energy(
        self,
        observation_likelihood: np.ndarray,
        transition_prediction: np.ndarray,
    ) -> FreeEnergyState:
        """
        Perform one step of variational free energy minimization.

        Args:
            observation_likelihood: P(o|s) for each state. Shape (state_dim,).
            transition_prediction: Predicted state distribution from
                                   transition model. Shape (state_dim,).

        Returns:
            FreeEnergyState with decomposed free energy components.
        """
        # Update prior from transition model
        self.p_prior = self.prior_strength * transition_prediction + \
                       (1 - self.prior_strength) * self.p_prior
        self.p_prior = self._normalize(self.p_prior)

        # Compute free energy components
        # Accuracy: E_q[ln p(o|s)]
        safe_likelihood = np.maximum(observation_likelihood, 1e-12)
        accuracy = float(np.dot(self.q_belief, np.log(safe_likelihood)))

        # Complexity: D_KL[q(s) || p(s)]
        complexity = float(self._kl_divergence(self.q_belief, self.p_prior))

        # Free energy: F = complexity - accuracy
        free_energy = complexity - accuracy

        # Surprise: -ln p(o) ≈ -ln Σ_s p(o|s)p(s)
        marginal_likelihood = np.dot(observation_likelihood, self.p_prior)
        surprise = float(-np.log(max(marginal_likelihood, 1e-12)))

        # Prediction error
        predicted_obs = np.dot(self.q_belief, safe_likelihood)
        actual_obs = np.mean(observation_likelihood)
        prediction_error = float(abs(predicted_obs - actual_obs))

        self._prediction_errors.append(prediction_error)
        if len(self._prediction_errors) > self._max_history:
            self._prediction_errors.pop(0)

        # Gradient descent on free energy to update q
        # ∂F/∂q ∝ ln(q/p) - ln(p(o|s))
        log_q = np.log(np.maximum(self.q_belief, 1e-12))
        log_p = np.log(np.maximum(self.p_prior, 1e-12))
        log_like = np.log(safe_likelihood)

        gradient = log_q - log_p - log_like
        gradient *= self.precision  # Precision-weighted gradient

        # Update belief
        self.q_belief = self.q_belief - self.learning_rate * gradient
        self.q_belief = np.maximum(self.q_belief, 1e-12)
        self.q_belief = self._normalize(self.q_belief)

        # Update precision (adaptive learning rate)
        self.precision *= self.precision_decay
        self.precision = np.maximum(self.precision, 0.1)

        # Epistemic value: expected information gain
        epistemic_value = self._compute_epistemic_value(
            observation_likelihood, transition_prediction
        )

        # Pragmatic value: goal-directed utility (prefer low cognitive load)
        pragmatic_value = self._compute_pragmatic_value()

        return FreeEnergyState(
            free_energy=round(free_energy, 6),
            surprise=round(surprise, 6),
            complexity=round(complexity, 6),
            accuracy=round(accuracy, 6),
            prediction_error=round(prediction_error, 6),
            epistemic_value=round(epistemic_value, 6),
            pragmatic_value=round(pragmatic_value, 6),
        )

    def get_belief(self) -> np.ndarray:
        """Return current approximate posterior q(s)."""
        return self.q_belief.copy()

    def get_precision(self) -> np.ndarray:
        """Return current precision weights."""
        return self.precision.copy()

    def get_mean_prediction_error(self) -> float:
        """Return the rolling mean prediction error."""
        if not self._prediction_errors:
            return 0.0
        return float(np.mean(self._prediction_errors))

    def reset(self):
        """Reset to uniform belief and default precision."""
        self.q_belief = np.ones(self.state_dim) / self.state_dim
        self.p_prior = np.ones(self.state_dim) / self.state_dim
        self.precision = np.ones(self.state_dim)
        self._prediction_errors.clear()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_epistemic_value(
        self, likelihood: np.ndarray, prediction: np.ndarray
    ) -> float:
        """
        Epistemic value = expected information gain.
        Higher when the agent is uncertain and observations are informative.
        """
        # Entropy of current belief
        entropy_q = self._entropy(self.q_belief)

        # Expected posterior entropy (approximate)
        posterior = likelihood * self.q_belief
        total = np.sum(posterior)
        if total > 1e-12:
            posterior /= total
        expected_entropy = self._entropy(posterior)

        return float(max(0.0, entropy_q - expected_entropy))

    def _compute_pragmatic_value(self) -> float:
        """
        Pragmatic value = alignment with preferred states.
        Prefers states with medium trust and low-medium cognitive load.
        """
        # Preferred state distribution (medium trust, low-medium load)
        preferred = np.zeros(self.state_dim)
        # Assuming 5 trust × 3 cogload layout
        for t in range(min(5, self.state_dim // 3)):
            for c in range(min(3, self.state_dim)):
                idx = t * 3 + c
                if idx < self.state_dim:
                    # Prefer medium trust (t=2), low load (c=0)
                    trust_pref = np.exp(-0.5 * (t - 2) ** 2)
                    load_pref = np.exp(-0.8 * c ** 2)
                    preferred[idx] = trust_pref * load_pref

        preferred = self._normalize(preferred)

        # Pragmatic value = -D_KL[q || preferred]
        return float(-self._kl_divergence(self.q_belief, preferred))

    @staticmethod
    def _kl_divergence(q: np.ndarray, p: np.ndarray) -> float:
        """KL(q || p) with numerical safety."""
        q_safe = np.maximum(q, 1e-12)
        p_safe = np.maximum(p, 1e-12)
        return float(np.sum(q_safe * np.log(q_safe / p_safe)))

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        """Shannon entropy H(p)."""
        p_safe = np.maximum(p, 1e-12)
        return float(-np.sum(p_safe * np.log(p_safe)))

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """Normalize to sum to 1."""
        total = np.sum(x)
        if total > 1e-12:
            return x / total
        return np.ones_like(x) / len(x)
