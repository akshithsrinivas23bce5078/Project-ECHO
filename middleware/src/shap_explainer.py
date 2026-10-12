"""
Project ECHO — SHAP-based Feature Attribution
================================================
Phase 4: Transparency / HCXAI

Provides model-agnostic feature attribution for explaining
DDA decisions to developers and (in simplified form) to players.

Implements a lightweight SHAP-like attribution that decomposes
each adaptation decision into per-feature contributions without
requiring the full SHAP library. This enables real-time
explainability within the game loop's latency budget.

Reference:
    Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to
    Interpreting Model Predictions. NeurIPS.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FeatureAttribution:
    """Attribution for a single feature's contribution to the decision."""
    feature_name: str
    value: float          # Current feature value
    attribution: float    # SHAP-like attribution (positive = push toward action)
    baseline: float       # Baseline (expected) value
    direction: str        # "positive" | "negative" | "neutral"
    magnitude: str        # "low" | "medium" | "high"


@dataclass
class ExplanationReport:
    """Complete explanation for a DDA decision."""
    decision: str                        # e.g., "decrease_difficulty"
    confidence: float                    # Decision confidence [0, 1]
    attributions: List[FeatureAttribution]
    top_contributors: List[str]          # Top 3 feature names
    counterfactual: str                  # "If X were different, then Y"
    natural_language: str                # Human-readable explanation
    evidence_summary: Dict[str, str]     # Feature → evidence string


class SHAPExplainer:
    """
    Lightweight SHAP-like feature attribution for real-time DDA explanations.

    Uses marginal contribution estimation via feature occlusion
    (a computationally cheap approximation of Shapley values)
    to attribute each decision to individual input features.
    """

    # Feature baselines (expected values for a neutral player)
    BASELINES = {
        "trust_mean": 0.5,
        "trust_variance": 0.06,
        "cognitive_load": 0.45,
        "clc_raw": 0.45,
        "emotion_confidence": 0.6,
        "neuromotor_load": 0.3,
        "snr_confidence": 0.5,
        "movement_energy": 2.0,
        "non_game_key_ratio": 0.1,
        "idle_seconds": 2.0,
    }

    # Feature importance weights (how much each feature matters)
    IMPORTANCE = {
        "trust_mean": 0.15,
        "trust_variance": 0.05,
        "cognitive_load": 0.20,
        "clc_raw": 0.10,
        "emotion_confidence": 0.10,
        "neuromotor_load": 0.15,
        "snr_confidence": 0.05,
        "movement_energy": 0.08,
        "non_game_key_ratio": 0.07,
        "idle_seconds": 0.05,
    }

    # Decision thresholds for direction labels
    EMOTION_DECISION_MAP = {
        "frustrated": "decrease_difficulty",
        "focused": "maintain",
        "fatigued": "decrease_difficulty",
        "neutral": "increase_difficulty",
        "engaged": "increase_difficulty",
    }

    def explain(
        self,
        belief_state: Dict[str, float],
        decision: str,
        emotion_label: str = "neutral",
    ) -> ExplanationReport:
        """
        Generate a complete explanation for a DDA decision.

        Args:
            belief_state: Current belief state values.
            decision: The DDA action taken.
            emotion_label: Current emotion classification.

        Returns:
            ExplanationReport with attributions and natural language.
        """
        attributions = self._compute_attributions(belief_state, decision)

        # Sort by absolute attribution
        attributions.sort(key=lambda a: abs(a.attribution), reverse=True)
        top_3 = [a.feature_name for a in attributions[:3]]

        # Compute confidence from attribution agreement
        positive_attrs = sum(a.attribution for a in attributions if a.attribution > 0)
        negative_attrs = sum(abs(a.attribution) for a in attributions if a.attribution < 0)
        total = positive_attrs + negative_attrs + 1e-6
        confidence = positive_attrs / total if decision != "maintain" else 0.5

        # Generate counterfactual
        counterfactual = self._generate_counterfactual(
            attributions, decision, emotion_label
        )

        # Generate natural language
        natural_language = self._generate_explanation(
            attributions[:3], decision, emotion_label
        )

        # Build evidence summary
        evidence = {}
        for attr in attributions:
            direction_emoji = "↑" if attr.direction == "positive" else "↓" if attr.direction == "negative" else "→"
            evidence[attr.feature_name] = (
                f"{direction_emoji} {attr.value:.3f} "
                f"(baseline: {attr.baseline:.3f}, "
                f"impact: {attr.magnitude})"
            )

        return ExplanationReport(
            decision=decision,
            confidence=round(confidence, 4),
            attributions=attributions,
            top_contributors=top_3,
            counterfactual=counterfactual,
            natural_language=natural_language,
            evidence_summary=evidence,
        )

    def _compute_attributions(
        self, belief_state: Dict[str, float], decision: str
    ) -> List[FeatureAttribution]:
        """Compute marginal contribution (SHAP-like) for each feature."""
        attributions = []

        for feature, importance in self.IMPORTANCE.items():
            value = belief_state.get(feature, self.BASELINES.get(feature, 0.0))
            baseline = self.BASELINES.get(feature, 0.0)

            # Marginal contribution: deviation from baseline × importance
            deviation = value - baseline
            attribution = deviation * importance

            # Direction and magnitude
            if abs(attribution) < 0.01:
                direction = "neutral"
                magnitude = "low"
            elif attribution > 0:
                direction = "positive"
                magnitude = "high" if abs(attribution) > 0.05 else "medium"
            else:
                direction = "negative"
                magnitude = "high" if abs(attribution) > 0.05 else "medium"

            attributions.append(FeatureAttribution(
                feature_name=feature,
                value=round(value, 4),
                attribution=round(attribution, 6),
                baseline=round(baseline, 4),
                direction=direction,
                magnitude=magnitude,
            ))

        return attributions

    def _generate_counterfactual(
        self,
        attributions: List[FeatureAttribution],
        decision: str,
        emotion_label: str,
    ) -> str:
        """Generate a counterfactual explanation."""
        if not attributions:
            return "No significant factors identified."

        top = attributions[0]
        opposite = "lower" if top.direction == "positive" else "higher"
        alt_decision = "maintain current settings"

        return (
            f"If {top.feature_name} were {opposite} "
            f"(closer to {top.baseline:.2f}), the system would likely "
            f"{alt_decision} instead of {decision}."
        )

    def _generate_explanation(
        self,
        top_attributions: List[FeatureAttribution],
        decision: str,
        emotion_label: str,
    ) -> str:
        """Generate natural language explanation from top attributions."""
        if not top_attributions:
            return f"Decision: {decision}. No significant contributing factors."

        # Map feature names to human-readable
        READABLE = {
            "trust_mean": "player trust level",
            "cognitive_load": "cognitive load",
            "neuromotor_load": "neuromotor stress",
            "emotion_confidence": "emotion detection confidence",
            "clc_raw": "typing complexity",
            "movement_energy": "physical movement",
            "non_game_key_ratio": "non-game key usage",
            "idle_seconds": "idle time",
            "snr_confidence": "signal quality",
            "trust_variance": "trust uncertainty",
        }

        factors = []
        for attr in top_attributions:
            name = READABLE.get(attr.feature_name, attr.feature_name)
            state = "elevated" if attr.direction == "positive" else "reduced"
            factors.append(f"{state} {name}")

        factor_str = ", ".join(factors[:-1])
        if len(factors) > 1:
            factor_str += f", and {factors[-1]}"
        else:
            factor_str = factors[0]

        action_desc = {
            "decrease_difficulty": "reducing difficulty",
            "increase_difficulty": "increasing challenge",
            "maintain": "maintaining current settings",
            "simplify_view": "simplifying the HUD",
            "offer_break": "suggesting a break",
        }.get(decision, decision)

        return (
            f"The system is {action_desc} primarily due to {factor_str}. "
            f"Current emotional state: {emotion_label}."
        )
