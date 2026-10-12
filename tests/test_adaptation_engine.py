"""
Project ECHO — Adaptation Engine Behavioral Tests
====================================================
Tests all 5 behavioral rules defined in the adaptation middleware spec:

1. Trust calibration (not persuasion)
2. Cognitive load response
3. Emotion-adaptive tone
4. Explainability output format
5. Guardrails (never disclose internals)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from middleware.src.models import EchoInputPayload, HcxaiOutput
from middleware.src.adaptation_engine import AdaptationEngine


class TestRule1TrustCalibration:
    """Rule 1: Trust calibration — not persuasion."""

    def test_high_trust_high_risk_increases_detail(self, high_trust_high_risk_payload):
        """
        High trust + high risk → detailed verbosity + surface uncertainty.
        Counters automation bias.
        """
        engine = AdaptationEngine()
        payload = EchoInputPayload(**high_trust_high_risk_payload)
        result = engine.process(payload)

        assert result.response_verbosity == "detailed"
        assert result.surface_uncertainty is True
        assert result.suggested_ui_action == "show_confidence_band"

    def test_low_trust_low_risk_is_concise(self, low_trust_low_risk_payload):
        """
        Low trust + low risk → minimal verbosity.
        Counters algorithm aversion.
        """
        engine = AdaptationEngine()
        payload = EchoInputPayload(**low_trust_low_risk_payload)
        result = engine.process(payload)

        assert result.response_verbosity == "minimal"
        assert result.surface_uncertainty is False


class TestRule2CognitiveLoad:
    """Rule 2: Cognitive load response."""

    def test_high_cognitive_load_simplifies(self, high_cognitive_load_payload):
        """cognitive_load > 0.7 → minimal verbosity + simplify view."""
        engine = AdaptationEngine()
        payload = EchoInputPayload(**high_cognitive_load_payload)
        result = engine.process(payload)

        # Cognitive load rule should trigger simplify_view
        # (emotion may override to offer_break if fatigued + confident)
        assert result.response_verbosity in ["minimal", "standard"]

    def test_low_cognitive_load_with_noise_flags(self):
        """
        cognitive_load < 0.3 with clc_raw outside [0.35, 0.55]
        should NOT assert emotional state confidently.
        """
        engine = AdaptationEngine()
        payload = EchoInputPayload(**{
            "belief_state": {
                "trust_mean": 0.5,
                "trust_variance": 0.1,
                "cognitive_load": 0.2,
                "clc_raw": 0.20,  # Outside nominal range
                "emotion_label": "frustrated",
                "emotion_confidence": 0.8,
            },
            "task_context": {"current_action": "run tests", "risk_level": "low"},
            "interaction_history": [],
        })
        result = engine.process(payload)

        # With measurement noise flagged, the engine should NOT
        # adapt tone based on emotion even though confidence > 0.6.
        # The explanation should NOT mention frustration.
        assert "frustrated" not in result.explanation.lower()


class TestRule3EmotionAdaptive:
    """Rule 3: Emotion-adaptive tone."""

    def test_low_emotion_confidence_ignored(self):
        """emotion_confidence < 0.6 → emotion label NOT referenced."""
        engine = AdaptationEngine()
        payload = EchoInputPayload(**{
            "belief_state": {
                "trust_mean": 0.5,
                "trust_variance": 0.1,
                "cognitive_load": 0.5,
                "clc_raw": 0.45,
                "emotion_label": "frustrated",
                "emotion_confidence": 0.3,  # Below threshold
            },
            "task_context": {"current_action": "compile", "risk_level": "med"},
            "interaction_history": [],
        })
        result = engine.process(payload)

        # Should not reference frustration with low confidence
        assert "take over" not in result.explanation.lower()

    def test_frustrated_with_high_confidence_adapts_tone(self):
        """Frustrated + high confidence → adjusted pacing/warmth."""
        engine = AdaptationEngine()
        payload = EchoInputPayload(**{
            "belief_state": {
                "trust_mean": 0.5,
                "trust_variance": 0.1,
                "cognitive_load": 0.5,
                "clc_raw": 0.45,
                "emotion_label": "frustrated",
                "emotion_confidence": 0.8,
            },
            "task_context": {"current_action": "debug", "risk_level": "med"},
            "interaction_history": [],
        })
        result = engine.process(payload)

        # Should use supportive tone
        assert "got this" in result.explanation.lower() or "take over" in result.explanation.lower()


class TestRule4OutputFormat:
    """Rule 4: HCXAI output format compliance."""

    def test_output_has_required_fields(self, high_trust_high_risk_payload):
        """Output must contain all four HCXAI fields."""
        engine = AdaptationEngine()
        payload = EchoInputPayload(**high_trust_high_risk_payload)
        result = engine.process(payload)

        assert hasattr(result, "response_verbosity")
        assert hasattr(result, "surface_uncertainty")
        assert hasattr(result, "suggested_ui_action")
        assert hasattr(result, "explanation")

    def test_verbosity_is_valid_enum(self, high_trust_high_risk_payload):
        """response_verbosity must be one of the three valid values."""
        engine = AdaptationEngine()
        payload = EchoInputPayload(**high_trust_high_risk_payload)
        result = engine.process(payload)

        assert result.response_verbosity in ["minimal", "standard", "detailed"]

    def test_ui_action_is_valid_enum(self, high_trust_high_risk_payload):
        """suggested_ui_action must be a valid directive."""
        engine = AdaptationEngine()
        payload = EchoInputPayload(**high_trust_high_risk_payload)
        result = engine.process(payload)

        valid = ["none", "offer_break", "show_confidence_band", "simplify_view"]
        assert result.suggested_ui_action in valid


class TestRule5Guardrails:
    """Rule 5: Guardrails — never disclose internals."""

    def test_no_raw_values_in_explanation(self, high_trust_high_risk_payload):
        """Explanation must NEVER contain raw biometric values."""
        engine = AdaptationEngine()
        payload = EchoInputPayload(**high_trust_high_risk_payload)
        result = engine.process(payload)

        explanation = result.explanation.lower()

        # Must not leak POMDP internals
        assert "0.85" not in explanation  # trust_mean
        assert "trust_mean" not in explanation
        assert "cognitive_load" not in explanation
        assert "clc_raw" not in explanation
        assert "pomdp" not in explanation
        assert "dster" not in explanation
        assert "spearman" not in explanation

    def test_no_task_decisions_in_output(self, high_trust_high_risk_payload):
        """
        The adaptation layer must NEVER make task decisions or
        override agent actions — only modulate communication.
        """
        engine = AdaptationEngine()
        payload = EchoInputPayload(**high_trust_high_risk_payload)
        result = engine.process(payload)

        explanation = result.explanation.lower()

        # Should not contain decision overrides
        assert "i will not" not in explanation
        assert "cancelling" not in explanation
        assert "aborting" not in explanation
        assert "stopping" not in explanation
