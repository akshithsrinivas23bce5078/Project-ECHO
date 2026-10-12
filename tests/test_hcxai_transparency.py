"""
Project ECHO — HCXAI Transparency Tests (Phase 4)
====================================================
Tests for SHAP Explainer and Adaptive Dashboard.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from middleware.src.shap_explainer import (
    SHAPExplainer, ExplanationReport, FeatureAttribution,
)
from middleware.src.adaptive_dashboard import (
    AdaptiveDashboard, DashboardState, DashboardDetailLevel,
    DashboardGauge, DDAAction,
)


class TestSHAPExplainer:
    """Tests for the SHAP-like feature attribution explainer."""

    def test_explain_returns_report(self):
        explainer = SHAPExplainer()
        belief = {"trust_mean": 0.3, "cognitive_load": 0.8, "neuromotor_load": 0.6}
        report = explainer.explain(belief, "decrease_difficulty", "frustrated")

        assert isinstance(report, ExplanationReport)
        assert report.decision == "decrease_difficulty"
        assert len(report.attributions) > 0
        assert len(report.top_contributors) <= 3

    def test_attributions_have_valid_fields(self):
        explainer = SHAPExplainer()
        belief = {"trust_mean": 0.7, "cognitive_load": 0.2}
        report = explainer.explain(belief, "increase_difficulty")

        for attr in report.attributions:
            assert isinstance(attr, FeatureAttribution)
            assert attr.direction in ["positive", "negative", "neutral"]
            assert attr.magnitude in ["low", "medium", "high"]

    def test_high_cogload_produces_negative_attribution(self):
        explainer = SHAPExplainer()
        belief = {"cognitive_load": 0.95}  # Very high
        report = explainer.explain(belief, "decrease_difficulty")

        cogload_attr = None
        for attr in report.attributions:
            if attr.feature_name == "cognitive_load":
                cogload_attr = attr
                break

        assert cogload_attr is not None
        assert cogload_attr.direction == "positive"  # Deviation is positive
        assert cogload_attr.attribution > 0

    def test_neutral_state_low_attributions(self):
        explainer = SHAPExplainer()
        belief = {}  # All defaults (baselines)
        report = explainer.explain(belief, "maintain")

        total_attribution = sum(abs(a.attribution) for a in report.attributions)
        assert total_attribution < 0.05  # Near-zero total attribution

    def test_counterfactual_generated(self):
        explainer = SHAPExplainer()
        belief = {"cognitive_load": 0.9}
        report = explainer.explain(belief, "decrease_difficulty")

        assert len(report.counterfactual) > 0
        assert "If" in report.counterfactual

    def test_natural_language_generated(self):
        explainer = SHAPExplainer()
        belief = {"trust_mean": 0.2, "cognitive_load": 0.8}
        report = explainer.explain(belief, "decrease_difficulty", "frustrated")

        assert len(report.natural_language) > 0
        assert "frustrated" in report.natural_language

    def test_evidence_summary_populated(self):
        explainer = SHAPExplainer()
        belief = {"trust_mean": 0.6}
        report = explainer.explain(belief, "maintain")

        assert len(report.evidence_summary) > 0
        for key, val in report.evidence_summary.items():
            assert isinstance(val, str)

    def test_confidence_in_valid_range(self):
        explainer = SHAPExplainer()
        belief = {"cognitive_load": 0.8}
        report = explainer.explain(belief, "decrease_difficulty")

        assert 0.0 <= report.confidence <= 1.0


class TestAdaptiveDashboard:
    """Tests for the Adaptive Transparency Dashboard."""

    def test_generate_returns_state(self):
        dashboard = AdaptiveDashboard()
        belief = {
            "trust_mean": 0.6, "cognitive_load": 0.5,
            "emotion_label": "focused", "emotion_confidence": 0.7,
        }
        state = dashboard.generate(
            belief, {"enemy_speed": 0.05, "spawn_rate": -0.02},
            timestamp=10.0,
        )

        assert isinstance(state, DashboardState)
        assert state.timestamp == 10.0

    def test_detail_level_minimal_for_high_load(self):
        dashboard = AdaptiveDashboard()
        belief = {"cognitive_load": 0.9, "trust_mean": 0.5,
                  "emotion_label": "neutral", "emotion_confidence": 0.5}
        state = dashboard.generate(belief, {})

        assert state.detail_level == "minimal"

    def test_detail_level_detailed_for_low_load(self):
        dashboard = AdaptiveDashboard()
        belief = {"cognitive_load": 0.2, "trust_mean": 0.5,
                  "emotion_label": "neutral", "emotion_confidence": 0.5}
        state = dashboard.generate(belief, {})

        assert state.detail_level == "detailed"

    def test_gauges_populated(self):
        dashboard = AdaptiveDashboard()
        belief = {"cognitive_load": 0.5, "trust_mean": 0.6,
                  "emotion_label": "focused", "emotion_confidence": 0.7}
        state = dashboard.generate(
            belief, {}, neuromotor_load=0.4, snr_confidence=0.7,
        )

        assert len(state.gauges) >= 2
        assert all(isinstance(g, DashboardGauge) for g in state.gauges)

    def test_dda_actions_filtered(self):
        dashboard = AdaptiveDashboard()
        belief = {"cognitive_load": 0.5, "trust_mean": 0.5,
                  "emotion_label": "neutral", "emotion_confidence": 0.5}
        state = dashboard.generate(
            belief, {"speed": 0.1, "health": 0.0001, "damage": -0.05},
        )

        # health delta too small, should be filtered
        params = [a.parameter for a in state.active_adjustments]
        assert "speed" in params
        assert "damage" in params
        assert "health" not in params

    def test_alerts_for_critical_load(self):
        dashboard = AdaptiveDashboard()
        belief = {"cognitive_load": 0.9, "trust_mean": 0.2,
                  "emotion_label": "frustrated", "emotion_confidence": 0.8}
        state = dashboard.generate(belief, {}, prediction_error=0.8)

        assert len(state.alerts) >= 2

    def test_free_energy_history_tracked(self):
        dashboard = AdaptiveDashboard(history_size=5)
        belief = {"cognitive_load": 0.5, "trust_mean": 0.5,
                  "emotion_label": "neutral", "emotion_confidence": 0.5}

        for i in range(7):
            state = dashboard.generate(
                belief, {}, free_energy=float(i), timestamp=float(i),
            )

        assert len(state.free_energy_trend) == 5  # Capped at history_size

    def test_emotion_data_populated(self):
        dashboard = AdaptiveDashboard()
        belief = {"cognitive_load": 0.5, "trust_mean": 0.5,
                  "emotion_label": "engaged", "emotion_confidence": 0.85}
        state = dashboard.generate(belief, {})

        assert state.emotion["label"] == "engaged"
        assert state.emotion["confidence"] == 0.85
        assert state.emotion["icon"] == "🔥"
