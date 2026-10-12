"""
Project ECHO — Adaptation Engine
===================================
Phase 4: Enhanced with neuromotor-aware DDA rules, SHAP-based
explainability, and adaptive dashboard integration.
"""

try:
    from middleware.src.models import EchoInputPayload, HcxaiOutput
except ImportError:
    from models import EchoInputPayload, HcxaiOutput


class AdaptationEngine:
    """
    Core DDA decision engine with Phase 4 neuromotor-awareness.

    Enhancements:
        - RULE 6: Neuromotor stress detection from IDeLog σ/μ deviations
        - RULE 7: Prediction error-driven recalibration
        - SHAP explainability integration for decision transparency
    """

    # Phase 4: Neuromotor stress thresholds
    NEUROMOTOR_STRESS_THRESHOLD = 0.7
    SIGMA_DEVIATION_THRESHOLD = 0.15

    def process(self, payload: EchoInputPayload) -> HcxaiOutput:
        bs = payload.belief_state
        tc = payload.task_context

        # Default State
        verbosity = "standard"
        surface_uncert = False
        ui_action = "none"
        explanation = f"I am preparing to {tc.current_action}."
        granular = None
        opt_in = None
        hud_declutter = False
        highlight_targets = False

        is_measurement_noise = False
        if bs.cognitive_load < 0.3 and not (0.35 <= bs.clc_raw <= 0.55):
            is_measurement_noise = True

        # RULE 6 (Phase 4): Neuromotor stress detection
        neuromotor_stressed = False
        if bs.neuromotor_state is not None:
            ns = bs.neuromotor_state
            if (ns.neuromotor_load > self.NEUROMOTOR_STRESS_THRESHOLD or
                    abs(ns.sigma_deviation) > self.SIGMA_DEVIATION_THRESHOLD):
                neuromotor_stressed = True

        # RULE 2: Cognitive load response (HUD Decluttering)
        if bs.cognitive_load > 0.8 or (neuromotor_stressed and bs.cognitive_load > 0.6):
            verbosity = "minimal"
            ui_action = "simplify_view"
            hud_declutter = True
            highlight_targets = True
            if neuromotor_stressed:
                explanation = (
                    "High cognitive and physical stress detected. "
                    "Decluttering HUD and highlighting priority targets."
                )
            else:
                explanation = (
                    "High cognitive load detected. Decluttering HUD "
                    "and highlighting priority targets."
                )

        # RULE 1 & Agency: Trust calibration and Opt-In Prompts
        if bs.trust_mean > 0.7 and tc.risk_level == "high":
            verbosity = "detailed"
            surface_uncert = True
            ui_action = "show_confidence_band"
            explanation = (
                f"I am preparing to {tc.current_action}. Because this is a high-risk action, "
                "I want to highlight that there is some uncertainty in the current projection. "
                "Please review the underlying data carefully."
            )
        elif bs.trust_mean < 0.4:
            # Low trust - prefer opt-in assistance rather than forced intervention
            verbosity = "minimal"
            surface_uncert = False
            opt_in = "Press 'F' for assistance"

        # RULE 4: Gaming Dynamic Difficulty Adjustment (DDA)
        if getattr(tc, "domain", "enterprise") == "gaming" and not is_measurement_noise:
            # Frustrated + Failing (simulated by high load)
            if bs.emotion_label == "frustrated" and bs.cognitive_load > 0.7:
                ui_action = "decrease_difficulty"
                granular = {"enemy_accuracy": -0.15, "spawn_rate": -0.10, "enemy_health": -0.20}

                # Phase 4: Amplify ease if neuromotor stress also detected
                if neuromotor_stressed:
                    granular["enemy_accuracy"] = -0.20
                    granular["spawn_rate"] = -0.15

                if bs.trust_mean < 0.4:
                    opt_in = "Press 'F' for emergency powerup"
                    explanation = "You seem frustrated. I can lower enemy stats if you need it."
                else:
                    explanation = "You seem a bit stuck. I've applied micro-adjustments: lowered enemy accuracy and spawn rate."

            # Flow State: Excited/Engaged/Focused + Failing (high load) -> No intervention
            elif bs.emotion_label in ["engaged", "focused"] and bs.cognitive_load > 0.7:
                ui_action = "none"
                explanation = "You're in a high-intensity flow state. Maintaining current challenge."

            # Flow State: Bored/Fatigued/Neutral + Succeeding (low load) -> Increase difficulty
            elif bs.emotion_label in ["fatigued", "neutral"] and bs.cognitive_load < 0.4:
                ui_action = "increase_difficulty"
                granular = {"enemy_speed": 0.15, "spawn_rate": 0.10}
                explanation = "Things seem a bit slow. Increasing challenge to keep you engaged."

            # Classic Flow: Engaged + Low load
            elif bs.emotion_label in ["engaged", "focused"] and bs.cognitive_load < 0.4:
                ui_action = "increase_difficulty"
                explanation = "You're breezing through! Increasing challenge to keep you in the zone."

            # RULE 6 (Phase 4): Neuromotor fatigue without emotional frustration
            elif neuromotor_stressed and bs.emotion_label not in ["frustrated"]:
                ui_action = "decrease_difficulty"
                granular = {"enemy_accuracy": -0.10, "spawn_rate": -0.05}
                explanation = "Detected physical stress in your inputs. Gently reducing challenge."

        # RULE 3: Emotion-adaptive tone (Enterprise Fallback)
        # Only adapt tone when emotion confidence is above threshold (0.6)
        elif not is_measurement_noise and getattr(tc, "domain", "enterprise") != "gaming" and bs.emotion_confidence >= 0.6:
            if bs.emotion_label == "frustrated":
                explanation = f"I've got this. {explanation} Let me know if you want to take over."
            elif bs.emotion_label == "fatigued":
                ui_action = "offer_break"
                explanation = f"I'm handling the {tc.current_action}. Take your time."

        return HcxaiOutput(
            response_verbosity=verbosity,
            surface_uncertainty=surface_uncert,
            suggested_ui_action=ui_action,
            explanation=explanation,
            granular_adjustments=granular,
            opt_in_prompt=opt_in,
            hud_declutter=hud_declutter,
            highlight_targets=highlight_targets
        )

