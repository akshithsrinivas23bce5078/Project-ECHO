"""
Project ECHO — Bayesian POMDP Trust/Cognitive-Load Model
==========================================================
Models human-agent interaction as a Partially Observable Markov
Decision Process (POMDP), where latent human trust and cognitive
load are inferred through real-time behavioral biometrics.

State Space:
    S = (trust_level, cognitive_load_level)
    - trust_level ∈ {very_low, low, medium, high, very_high}  (5 states)
    - cognitive_load_level ∈ {low, medium, high}               (3 states)
    - Total: 15 composite states

Observation Space:
    O = (clc_raw, emotion_label, emotion_confidence, iki_mean, snr_db)

Actions:
    A = (verbosity_level, ui_action)
    These are the adaptation layer's outputs, not task actions.

The belief state b(s) is a probability distribution over all 15 states,
continuously updated via Bayesian inference:
    b'(s') = η * O(o|s',a) * Σ_s T(s'|s,a) * b(s)

Phase 3: Active Inference integration via optional Free Energy Engine.
"""

import numpy as np
import math
from typing import Dict, Tuple, Optional
from estimator.src.free_energy_engine import FreeEnergyEngine, FreeEnergyState


# State indices
TRUST_LEVELS = ["very_low", "low", "medium", "high", "very_high"]
COGLOAD_LEVELS = ["low", "medium", "high"]

NUM_TRUST = len(TRUST_LEVELS)
NUM_COGLOAD = len(COGLOAD_LEVELS)
NUM_STATES = NUM_TRUST * NUM_COGLOAD  # 15


def state_index(trust_idx: int, cogload_idx: int) -> int:
    """Map (trust, cogload) pair to flat state index."""
    return trust_idx * NUM_COGLOAD + cogload_idx


def state_from_index(idx: int) -> Tuple[str, str]:
    """Map flat state index to (trust_level, cogload_level)."""
    t = idx // NUM_COGLOAD
    c = idx % NUM_COGLOAD
    return TRUST_LEVELS[t], COGLOAD_LEVELS[c]


class POMDPModel:
    """
    POMDP for inferring latent trust and cognitive load.

    Maintains:
        - Transition model T(s'|s, a): how states evolve
        - Observation model O(o|s): how observations relate to states
        - Belief state b(s): current probability distribution
    """

    def __init__(self, enable_active_inference: bool = False):
        # Initialize uniform belief state
        self.belief = np.ones(NUM_STATES) / NUM_STATES

        self.enable_active_inference = enable_active_inference
        self._free_energy_engine = None
        self._last_fe_state = None
        if enable_active_inference:
            self._free_energy_engine = FreeEnergyEngine(state_dim=NUM_STATES)

        # Transition model: slight tendency toward state persistence
        # T[s][s'] = probability of transitioning from s to s'
        self.transition = self._build_transition_model()

        # Observation model parameters (Gaussian means/stds per state)
        self.obs_params = self._build_observation_model()

    def get_belief_summary(self) -> Dict:
        """
        Summarize the current belief state into trust_mean, trust_variance,
        and cognitive_load for the downstream adaptation engine.
        """
        # Marginal trust distribution
        trust_marginal = np.zeros(NUM_TRUST)
        for t in range(NUM_TRUST):
            for c in range(NUM_COGLOAD):
                trust_marginal[t] += self.belief[state_index(t, c)]

        # Trust mean: weighted average of trust indices normalized to [0, 1]
        trust_values = np.array([i / (NUM_TRUST - 1) for i in range(NUM_TRUST)])
        trust_mean = float(np.dot(trust_marginal, trust_values))
        trust_variance = float(np.dot(trust_marginal, (trust_values - trust_mean) ** 2))

        # Marginal cognitive load distribution
        cogload_marginal = np.zeros(NUM_COGLOAD)
        for c in range(NUM_COGLOAD):
            for t in range(NUM_TRUST):
                cogload_marginal[c] += self.belief[state_index(t, c)]

        cogload_values = np.array([i / (NUM_COGLOAD - 1) for i in range(NUM_COGLOAD)])
        cognitive_load = float(np.dot(cogload_marginal, cogload_values))

        summary = {
            "trust_mean": round(trust_mean, 4),
            "trust_variance": round(trust_variance, 4),
            "cognitive_load": round(cognitive_load, 4),
        }

        # Phase 3: Add free energy metrics if active inference is enabled
        if self._last_fe_state is not None:
            summary["free_energy"] = self._last_fe_state.free_energy
            summary["surprise"] = self._last_fe_state.surprise
            summary["prediction_error"] = self._last_fe_state.prediction_error
            summary["epistemic_value"] = self._last_fe_state.epistemic_value
            summary["pragmatic_value"] = self._last_fe_state.pragmatic_value

        return summary

    def update_belief(self, observation: Dict) -> Dict:
        """
        Bayesian belief update given a new observation.

        b'(s') = η * O(o|s') * Σ_s T(s'|s) * b(s)

        Args:
            observation: Dict with keys clc_raw, iki_mean_ms, snr_db.

        Returns:
            Updated belief summary.
        """
        # Step 1: State prediction (transition)
        predicted = self.transition.T @ self.belief  # (NUM_STATES,)

        # Step 2: Observation likelihood
        likelihood = self._observation_likelihood(observation)

        # Step 3: Bayes update
        updated = predicted * likelihood
        total = np.sum(updated)
        if total > 1e-12:
            updated /= total
        else:
            # Laplace Smoothing: inject small uniform noise to prevent structural collapse
            # while retaining the predicted prior.
            noise = np.ones(NUM_STATES) * 1e-5
            updated = predicted + noise
            updated /= np.sum(updated)

        self.belief = updated

        # Phase 3: Run free energy minimization in parallel
        if self._free_energy_engine is not None:
            self._last_fe_state = self._free_energy_engine.minimize_free_energy(
                observation_likelihood=likelihood,
                transition_prediction=predicted,
            )

        return self.get_belief_summary()

    def reset(self):
        """Reset belief to uniform."""
        if self._free_energy_engine is not None:
            self._free_energy_engine.reset()
            self._last_fe_state = None
        self.belief = np.ones(NUM_STATES) / NUM_STATES

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_transition_model(self) -> np.ndarray:
        """
        Build transition matrix T[s, s'].
        States tend to persist with slight diffusion to neighbors.
        """
        T = np.zeros((NUM_STATES, NUM_STATES))

        for s in range(NUM_STATES):
            t_s, c_s = s // NUM_COGLOAD, s % NUM_COGLOAD

            for sp in range(NUM_STATES):
                t_sp, c_sp = sp // NUM_COGLOAD, sp % NUM_COGLOAD

                # Probability based on distance in trust and cogload axes
                t_dist = abs(t_sp - t_s)
                c_dist = abs(c_sp - c_s)

                # Exponential decay from current state
                T[s, sp] = np.exp(-1.5 * t_dist) * np.exp(-1.5 * c_dist)

            # Normalize rows
            T[s, :] /= np.sum(T[s, :])

        return T

    def _build_observation_model(self) -> Dict:
        """
        Define expected observation statistics for each state.
        Maps state → expected (clc_raw, iki_mean_ms, snr_db).
        """
        params = {}
        for s in range(NUM_STATES):
            trust_level, cogload_level = state_from_index(s)

            # CLC: higher cognitive load → higher CLC
            clc_mean = {"low": 0.30, "medium": 0.45, "high": 0.60}[cogload_level]
            clc_std = 0.08

            # IKI: higher cognitive load → slower typing
            iki_mean = {"low": 120.0, "medium": 200.0, "high": 350.0}[cogload_level]
            iki_std = 50.0

            # SNR: higher trust generally correlates with smoother motor control
            snr_mean = {
                "very_low": 18.0, "low": 20.0, "medium": 24.0,
                "high": 27.0, "very_high": 30.0
            }[trust_level]
            snr_std = 3.0

            params[s] = {
                "clc": (clc_mean, clc_std),
                "iki": (iki_mean, iki_std),
                "snr": (snr_mean, snr_std),
            }

        return params

    def _observation_likelihood(self, obs: Dict) -> np.ndarray:
        """
        Compute P(observation | state) for all states.
        Uses Gaussian likelihood for each observation dimension.
        """
        likelihood = np.ones(NUM_STATES)

        clc = obs.get("clc_raw", 0.45)
        iki = obs.get("iki_mean_ms", 200.0)
        snr = obs.get("snr_db", 24.0)
        
        # Fault Tolerance: Check for NaN/Inf/None
        if clc is None or math.isnan(clc) or math.isinf(clc): clc = 0.45
        if iki is None or math.isnan(iki) or math.isinf(iki): iki = 200.0
        if snr is None or math.isnan(snr) or math.isinf(snr): snr = 24.0

        # Fault Tolerance: Bound extreme outliers to prevent PDF underflow
        clc = max(0.0, min(1.0, clc))
        iki = max(0.0, min(2000.0, iki))
        snr = max(0.0, min(100.0, snr))

        for s in range(NUM_STATES):
            p = self.obs_params[s]

            # Gaussian likelihood per dimension
            l_clc = self._gaussian_pdf(clc, *p["clc"])
            l_iki = self._gaussian_pdf(iki, *p["iki"])
            l_snr = self._gaussian_pdf(snr, *p["snr"])

            likelihood[s] = l_clc * l_iki * l_snr

        return likelihood

    @staticmethod
    def _gaussian_pdf(x: float, mean: float, std: float) -> float:
        """Univariate Gaussian PDF."""
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )
