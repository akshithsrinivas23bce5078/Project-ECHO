"""
Project ECHO — HCXAI Output Formatter
=======================================
Formats the adaptation engine's output into the final Human-Centered
Conversational Explainable AI (HCXAI) JSON specification.

Adds Gemini-style explanation templating with:
    - Contextual preambles based on risk level
    - Dynamic phrasing based on verbosity
    - Guardrail enforcement (no raw biometrics disclosed)

Phase 4: Feature attribution and evidence summaries from SHAP
         explainer integrated into the output.

This module sits between the AdaptationEngine and the API response,
ensuring all user-facing text follows HCXAI principles.
"""

from typing import Dict, List, Optional


class HCXAIFormatter:
    """
    Templates and formats the HCXAI response for the React frontend.

    GUARDRAILS:
        - NEVER disclose raw biometric values, DSTER scores, or POMDP
          internals to the end user.
        - NEVER make task decisions or override agent actions.
        - Only modulate HOW the agent communicates.
    """

    # Contextual preambles by risk level
    RISK_PREAMBLES = {
        "low": "",
        "med": "I want to be transparent about this step. ",
        "high": "This is a critical action — please take a moment to review. ",
    }

    # Verbosity templates
    TEMPLATES = {
        "minimal": {
            "prefix": "",
            "suffix": "",
            "max_sentences": 2,
        },
        "standard": {
            "prefix": "",
            "suffix": " Let me know if you'd like more detail.",
            "max_sentences": 4,
        },
        "detailed": {
            "prefix": "Here's a detailed breakdown: ",
            "suffix": " I've surfaced the key uncertainties for your review.",
            "max_sentences": 8,
        },
    }

    def format(
        self,
        verbosity: str,
        surface_uncertainty: bool,
        ui_action: str,
        explanation: str,
        risk_level: str = "low",
        evidence_summary: Optional[Dict[str, str]] = None,
        top_contributors: Optional[List[str]] = None,
        counterfactual: Optional[str] = None,
    ) -> Dict:
        """
        Format the adaptation engine output into the final HCXAI response.

        Args:
            verbosity: "minimal" | "standard" | "detailed"
            surface_uncertainty: Whether to show confidence bands.
            ui_action: Suggested UI action directive.
            explanation: Raw explanation text from the engine.
            risk_level: "low" | "med" | "high"
            evidence_summary: Phase 4 SHAP evidence per feature.
            top_contributors: Phase 4 top contributing features.
            counterfactual: Phase 4 counterfactual explanation.

        Returns:
            HCXAI-compliant JSON dict.
        """
        template = self.TEMPLATES.get(verbosity, self.TEMPLATES["standard"])
        preamble = self.RISK_PREAMBLES.get(risk_level, "")

        # Build the formatted explanation
        formatted = f"{preamble}{template['prefix']}{explanation}{template['suffix']}"

        # Trim to max sentences for verbosity level
        formatted = self._trim_sentences(formatted, template["max_sentences"])

        # Add uncertainty language if flagged
        if surface_uncertainty and "uncertainty" not in formatted.lower():
            formatted += (
                " Note: there is some uncertainty in the current projections."
            )

        response = {
            "response_verbosity": verbosity,
            "surface_uncertainty": surface_uncertainty,
            "suggested_ui_action": ui_action,
            "explanation": formatted.strip(),
        }

        # Phase 4: Add transparency metadata if available
        if evidence_summary and verbosity == "detailed":
            # Filter evidence to only show human-readable entries
            response["evidence_summary"] = {
                self._humanize_feature(k): v
                for k, v in (evidence_summary or {}).items()
            }

        if top_contributors:
            response["decision_factors"] = [
                self._humanize_feature(f) for f in top_contributors
            ]

        if counterfactual and verbosity == "detailed":
            response["counterfactual"] = counterfactual

        return response

    @staticmethod
    def _trim_sentences(text: str, max_sentences: int) -> str:
        """Trim text to at most max_sentences sentences."""
        sentences = text.replace("! ", ". ").split(". ")
        if len(sentences) <= max_sentences:
            return text
        trimmed = ". ".join(sentences[:max_sentences])
        if not trimmed.endswith("."):
            trimmed += "."
        return trimmed

    @staticmethod
    def _humanize_feature(feature: str) -> str:
        """Convert internal feature names to human-readable labels."""
        HUMAN_NAMES = {
            "trust_mean": "Trust Level",
            "cognitive_load": "Cognitive Load",
            "neuromotor_load": "Physical Stress",
            "emotion_confidence": "Emotion Confidence",
            "clc_raw": "Typing Complexity",
            "movement_energy": "Movement Energy",
            "snr_confidence": "Signal Quality",
            "trust_variance": "Trust Uncertainty",
            "non_game_key_ratio": "Non-Game Key Usage",
            "idle_seconds": "Idle Time",
        }
        return HUMAN_NAMES.get(feature, feature.replace("_", " ").title())

