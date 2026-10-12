"""
Project ECHO — Bayesian Belief Updater
========================================
Orchestrates the continuous loop of:
    1. Collecting observations from input_capture and feature_extraction
    2. Running Bayesian belief update via the POMDP model
    3. Fusing DSTER emotion output with POMDP trust/cogload
    4. Emitting the unified belief_state for the adaptation middleware

Phase 3: Integrates neuromotor state and free energy metrics.

This is the glue between the raw signal pipeline and the POMDP.
"""

import time
import numpy as np
from typing import Dict, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from estimator.src.pomdp_model import POMDPModel


class BeliefUpdater:
    """
    Fuses observations from CLC, DSTER, and neuromotor analysis
    into a single belief_state payload for the adaptation engine.

    Phase 3: Accepts neuromotor state (IDeLog σ/μ) and propagates
    free energy metrics when active inference is enabled.
    """

    def __init__(self, enable_active_inference: bool = False):
        self.pomdp = POMDPModel(enable_active_inference=enable_active_inference)
        self._last_update_time: float = 0.0
        self._update_count: int = 0

    def update(
        self,
        clc_raw: float,
        clc_is_nominal: bool,
        iki_mean_ms: float,
        snr_db: float,
        emotion_label: str,
        emotion_confidence: float,
        neuromotor_load: float = 0.0,
        sigma_mean: float = 0.40,
        mu_mean: float = -0.80,
    ) -> Dict:
        """
        Perform a single belief update cycle.

        Args:
            clc_raw: Spearman rho from CLC estimator.
            clc_is_nominal: Whether CLC is in [0.35, 0.55].
            iki_mean_ms: Mean inter-keystroke interval (ms).
            snr_db: Sigma-Lognormal reconstruction SNR (dB).
            emotion_label: DSTER output label.
            emotion_confidence: DSTER softmax confidence.
            neuromotor_load: Phase 1 unified neuromotor load [0, 1].
            sigma_mean: Phase 1 mean σ from IDeLog.
            mu_mean: Phase 1 mean μ from IDeLog.

        Returns:
            Complete belief_state dict matching the middleware input contract.
        """
        now = time.perf_counter()

        # Build observation vector for POMDP
        observation = {
            "clc_raw": clc_raw,
            "iki_mean_ms": iki_mean_ms,
            "snr_db": snr_db,
        }

        # Bayesian update (+ optional free energy minimization)
        summary = self.pomdp.update_belief(observation)

        # Normalize cognitive_load from CLC if CLC is valid
        if clc_is_nominal:
            cognitive_load = self._fuse_cognitive_load(
                summary["cognitive_load"], clc_raw, neuromotor_load
            )
        else:
            cognitive_load = summary["cognitive_load"]

        # Compose the belief_state payload
        belief_state = {
            "trust_mean": summary["trust_mean"],
            "trust_variance": summary["trust_variance"],
            "cognitive_load": round(cognitive_load, 4),
            "clc_raw": round(clc_raw, 4),
            "emotion_label": emotion_label,
            "emotion_confidence": round(emotion_confidence, 4),
            "neuromotor_load": round(neuromotor_load, 4),
            "sigma_mean": round(sigma_mean, 4),
            "mu_mean": round(mu_mean, 4),
        }

        # Phase 3: Add free energy metrics if available
        if "free_energy" in summary:
            belief_state["free_energy"] = summary["free_energy"]
            belief_state["prediction_error"] = summary["prediction_error"]
            belief_state["surprise"] = summary["surprise"]
            belief_state["epistemic_value"] = summary["epistemic_value"]

        self._last_update_time = now
        self._update_count += 1

        return belief_state

    def get_update_stats(self) -> Dict:
        """Return statistics about the update loop."""
        return {
            "total_updates": self._update_count,
            "last_update_time": self._last_update_time,
        }

    def reset(self):
        """Reset the POMDP and counters."""
        self.pomdp.reset()
        self._update_count = 0
        self._last_update_time = 0.0

    @staticmethod
    def _fuse_cognitive_load(
        pomdp_load: float, clc_raw: float, neuromotor_load: float = 0.0,
    ) -> float:
        """
        Fuse the POMDP-estimated cognitive load with CLC and neuromotor signals.

        Phase 3: Three-way fusion:
            50% POMDP (Bayesian posterior)
            30% CLC (keystroke-derived)
            20% neuromotor (mouse kinematics-derived)
        """
        clc_load = max(0.0, min(1.0, clc_raw))
        neuro_load = max(0.0, min(1.0, neuromotor_load))

        fused = 0.5 * pomdp_load + 0.3 * clc_load + 0.2 * neuro_load
        return fused

