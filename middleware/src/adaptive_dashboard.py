"""
Project ECHO — Adaptive Transparency Dashboard
=================================================
Phase 4: Transparency / HCXAI

Generates real-time dashboard state data for the transparency
overlay in the game HUD. This module produces structured data
that a frontend renderer can consume to display:

    - Current emotional state with confidence bar
    - Cognitive load gauge
    - Trust level indicator
    - Active DDA adjustments with explanations
    - Free energy / prediction error trend
    - Feature attribution pie chart data

The dashboard adapts its detail level based on the player's
cognitive load — showing less detail when they're overwhelmed.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class DashboardDetailLevel(Enum):
    """How much detail to show based on cognitive load."""
    MINIMAL = "minimal"    # High load — show only critical alerts
    STANDARD = "standard"  # Normal — show key metrics
    DETAILED = "detailed"  # Low load — show full transparency


@dataclass
class DashboardGauge:
    """A single gauge/metric for the dashboard."""
    label: str
    value: float          # Current value [0, 1]
    trend: str            # "rising" | "falling" | "stable"
    color: str            # CSS color for rendering
    tooltip: str          # Hover explanation


@dataclass
class DDAAction:
    """A single DDA adjustment for display."""
    parameter: str
    direction: str       # "increased" | "decreased" | "unchanged"
    magnitude: float     # Absolute change magnitude
    reason: str          # Human-readable reason


@dataclass
class DashboardState:
    """Complete dashboard data payload for the frontend."""
    detail_level: str
    emotion: Dict[str, object]        # label, confidence, icon
    gauges: List[DashboardGauge]
    active_adjustments: List[DDAAction]
    attribution_chart: List[Dict]     # [{name, value, color}]
    alerts: List[str]                 # Urgent notifications
    free_energy_trend: List[float]    # Last N free energy values
    timestamp: float


class AdaptiveDashboard:
    """
    Generates adaptive transparency dashboard data.

    Adjusts detail level based on cognitive load to avoid
    overwhelming the player with information when they're
    already under high cognitive strain.
    """

    EMOTION_ICONS = {
        "frustrated": "😤",
        "focused": "🎯",
        "fatigued": "😴",
        "neutral": "😐",
        "engaged": "🔥",
    }

    GAUGE_COLORS = {
        "trust": "#4CAF50",       # Green
        "cognitive_load": "#FF9800",  # Orange
        "neuromotor_load": "#2196F3",  # Blue
        "prediction_error": "#F44336",  # Red
        "signal_quality": "#9C27B0",   # Purple
    }

    def __init__(self, history_size: int = 30):
        self._free_energy_history: List[float] = []
        self._history_size = history_size

    def generate(
        self,
        belief_state: Dict,
        dda_adjustments: Dict[str, float],
        explanation: Optional[Dict] = None,
        free_energy: float = 0.0,
        prediction_error: float = 0.0,
        neuromotor_load: float = 0.0,
        snr_confidence: float = 0.5,
        timestamp: float = 0.0,
    ) -> DashboardState:
        """
        Generate dashboard state from current system state.

        Args:
            belief_state: Current belief state dict.
            dda_adjustments: Parameter deltas from DDA.
            explanation: SHAP explanation report (optional).
            free_energy: Current free energy value.
            prediction_error: Current prediction error.
            neuromotor_load: Current neuromotor load.
            snr_confidence: SNR quality confidence.
            timestamp: Current session time.

        Returns:
            DashboardState ready for frontend rendering.
        """
        cognitive_load = belief_state.get("cognitive_load", 0.5)
        trust = belief_state.get("trust_mean", 0.5)
        emotion = belief_state.get("emotion_label", "neutral")
        emotion_conf = belief_state.get("emotion_confidence", 0.5)

        # Determine detail level
        detail = self._compute_detail_level(cognitive_load)

        # Track free energy
        self._free_energy_history.append(free_energy)
        if len(self._free_energy_history) > self._history_size:
            self._free_energy_history.pop(0)

        # Build emotion display
        emotion_data = {
            "label": emotion,
            "confidence": round(emotion_conf, 2),
            "icon": self.EMOTION_ICONS.get(emotion, "😐"),
        }

        # Build gauges
        gauges = self._build_gauges(
            trust, cognitive_load, neuromotor_load,
            prediction_error, snr_confidence
        )

        # Build DDA actions
        actions = self._build_dda_actions(dda_adjustments)

        # Build attribution chart
        chart = []
        if explanation and detail != DashboardDetailLevel.MINIMAL.value:
            chart = self._build_attribution_chart(explanation)

        # Build alerts
        alerts = self._build_alerts(
            cognitive_load, trust, prediction_error, emotion, emotion_conf
        )

        return DashboardState(
            detail_level=detail,
            emotion=emotion_data,
            gauges=gauges if detail != DashboardDetailLevel.MINIMAL.value else gauges[:2],
            active_adjustments=actions if detail != DashboardDetailLevel.MINIMAL.value else [],
            attribution_chart=chart,
            alerts=alerts,
            free_energy_trend=list(self._free_energy_history),
            timestamp=timestamp,
        )

    def _compute_detail_level(self, cognitive_load: float) -> str:
        if cognitive_load > 0.75:
            return DashboardDetailLevel.MINIMAL.value
        elif cognitive_load > 0.4:
            return DashboardDetailLevel.STANDARD.value
        return DashboardDetailLevel.DETAILED.value

    def _build_gauges(
        self, trust: float, cogload: float, neuromotor: float,
        pred_error: float, snr_conf: float,
    ) -> List[DashboardGauge]:
        def trend_str(val, threshold_low=0.4, threshold_high=0.6):
            if val > threshold_high:
                return "rising"
            elif val < threshold_low:
                return "falling"
            return "stable"

        return [
            DashboardGauge(
                label="Trust", value=round(trust, 3),
                trend=trend_str(trust),
                color=self.GAUGE_COLORS["trust"],
                tooltip="Inferred player trust in the DDA system",
            ),
            DashboardGauge(
                label="Cognitive Load", value=round(cogload, 3),
                trend=trend_str(cogload),
                color=self.GAUGE_COLORS["cognitive_load"],
                tooltip="Estimated cognitive load from typing and mouse patterns",
            ),
            DashboardGauge(
                label="Neuromotor Load", value=round(neuromotor, 3),
                trend=trend_str(neuromotor),
                color=self.GAUGE_COLORS["neuromotor_load"],
                tooltip="Physical stress from mouse kinematics analysis",
            ),
            DashboardGauge(
                label="Prediction Error", value=round(min(pred_error, 1.0), 3),
                trend=trend_str(pred_error, 0.2, 0.5),
                color=self.GAUGE_COLORS["prediction_error"],
                tooltip="How much the player's behavior deviates from expectations",
            ),
            DashboardGauge(
                label="Signal Quality", value=round(snr_conf, 3),
                trend=trend_str(snr_conf),
                color=self.GAUGE_COLORS["signal_quality"],
                tooltip="Confidence in biometric signal quality (SNR)",
            ),
        ]

    def _build_dda_actions(self, adjustments: Dict[str, float]) -> List[DDAAction]:
        actions = []
        for param, delta in adjustments.items():
            if abs(delta) < 0.001:
                continue
            actions.append(DDAAction(
                parameter=param,
                direction="increased" if delta > 0 else "decreased",
                magnitude=round(abs(delta), 4),
                reason=f"{'Increased' if delta > 0 else 'Decreased'} by {abs(delta):.1%}",
            ))
        return actions

    def _build_attribution_chart(self, explanation: Dict) -> List[Dict]:
        chart = []
        colors = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
        attrs = explanation.get("attributions", [])
        for i, attr in enumerate(attrs[:5]):
            chart.append({
                "name": attr.get("feature_name", f"feature_{i}") if isinstance(attr, dict)
                        else getattr(attr, "feature_name", f"feature_{i}"),
                "value": abs(attr.get("attribution", 0.0)) if isinstance(attr, dict)
                         else abs(getattr(attr, "attribution", 0.0)),
                "color": colors[i % len(colors)],
            })
        return chart

    def _build_alerts(
        self, cogload: float, trust: float,
        pred_error: float, emotion: str, emotion_conf: float,
    ) -> List[str]:
        alerts = []
        if cogload > 0.85:
            alerts.append("⚠️ Critical cognitive load — simplifying HUD")
        if trust < 0.25:
            alerts.append("⚠️ Very low trust — switching to opt-in mode")
        if pred_error > 0.7:
            alerts.append("⚠️ High prediction error — recalibrating model")
        if emotion == "frustrated" and emotion_conf > 0.75:
            alerts.append("😤 Frustration detected with high confidence")
        return alerts
