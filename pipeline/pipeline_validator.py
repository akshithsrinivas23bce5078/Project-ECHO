"""
Project ECHO — Pipeline Validator
=====================================
Validates data contracts between pipeline stages to ensure
synchronous operation without data shape/type/range errors.

Each validator checks:
    - Required keys are present
    - Data types are correct
    - Value ranges are within expected bounds
    - Shapes/dimensions match expected formats

Usage:
    from pipeline.pipeline_validator import PipelineValidator

    validator = PipelineValidator()
    result = validator.validate_features(features_dict)
    if not result.is_valid:
        print(result.errors)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    is_valid: bool = True
    stage: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str):
        """Add an error and mark as invalid."""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        """Add a warning (does not invalidate)."""
        self.warnings.append(msg)

    def summary(self) -> str:
        """Generate a validation summary string."""
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"Validation [{self.stage}]: {status}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class PipelineValidator:
    """
    Validates inter-stage data contracts in the ECHO pipeline.

    Ensures that each component's output matches the next
    component's expected input format.
    """

    # Expected feature extraction output keys and ranges
    FEATURE_SCHEMA = {
        "clc_raw": {"type": (int, float), "min": 0.0, "max": 1.0, "required": True},
        "iki_mean_ms": {"type": (int, float), "min": 0.0, "max": 5000.0, "required": True},
        "snr_db": {"type": (int, float), "min": -10.0, "max": 60.0, "required": True},
        "stroke_count": {"type": (int, float), "min": 0, "max": 500, "required": False},
        "neuromotor_load": {"type": (int, float), "min": 0.0, "max": 1.0, "required": False},
    }

    # Expected emotion output keys
    EMOTION_SCHEMA = {
        "label": {
            "type": str,
            "valid_values": ["frustrated", "focused", "fatigued", "neutral", "engaged"],
            "required": True,
        },
        "confidence": {"type": (int, float), "min": 0.0, "max": 1.0, "required": True},
    }

    # Expected belief output keys
    BELIEF_SCHEMA = {
        "trust_mean": {"type": (int, float), "min": 0.0, "max": 1.0, "required": True},
        "trust_variance": {"type": (int, float), "min": 0.0, "max": 1.0, "required": True},
        "cognitive_load": {"type": (int, float), "min": 0.0, "max": 1.0, "required": True},
    }

    def validate_features(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate feature extraction stage output."""
        return self._validate_against_schema(
            data, self.FEATURE_SCHEMA, stage="Feature Extraction"
        )

    def validate_emotion(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate DSTER emotion classification output."""
        return self._validate_against_schema(
            data, self.EMOTION_SCHEMA, stage="DSTER Emotion"
        )

    def validate_belief(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate POMDP belief update output."""
        return self._validate_against_schema(
            data, self.BELIEF_SCHEMA, stage="POMDP Belief"
        )

    def validate_full_pipeline(
        self,
        features: Dict,
        emotion: Dict,
        belief: Dict,
    ) -> ValidationResult:
        """
        Validate the complete pipeline data flow.

        Checks all inter-stage contracts and cross-stage consistency.
        """
        result = ValidationResult(stage="Full Pipeline")

        # Validate individual stages
        feat_result = self.validate_features(features)
        emo_result = self.validate_emotion(emotion)
        belief_result = self.validate_belief(belief)

        # Aggregate errors
        for sub_result in [feat_result, emo_result, belief_result]:
            if not sub_result.is_valid:
                result.is_valid = False
                for error in sub_result.errors:
                    result.add_error(f"[{sub_result.stage}] {error}")
            for warning in sub_result.warnings:
                result.add_warning(f"[{sub_result.stage}] {warning}")

        # Cross-stage consistency checks
        clc = features.get("clc_raw", 0.45)
        cogload = belief.get("cognitive_load", 0.5)

        # CLC should roughly correlate with cognitive load
        if abs(clc - 0.35 - cogload * 0.3) > 0.5:
            result.add_warning(
                f"CLC ({clc:.3f}) and cognitive load ({cogload:.3f}) appear uncorrelated"
            )

        return result

    def _validate_against_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Dict],
        stage: str,
    ) -> ValidationResult:
        """Validate data against a schema definition."""
        result = ValidationResult(stage=stage)

        if not isinstance(data, dict):
            result.add_error(f"Expected dict, got {type(data).__name__}")
            return result

        for key, rules in schema.items():
            is_required = rules.get("required", False)

            if key not in data:
                if is_required:
                    result.add_error(f"Missing required key: '{key}'")
                continue

            value = data[key]

            # Type check
            expected_type = rules.get("type")
            if expected_type is not None:
                if not isinstance(value, expected_type):
                    result.add_error(
                        f"Key '{key}': expected {expected_type}, "
                        f"got {type(value).__name__} ({value!r})"
                    )
                    continue

            # Range check (for numeric types)
            if "min" in rules and isinstance(value, (int, float)):
                if value < rules["min"]:
                    result.add_warning(
                        f"Key '{key}': value {value} below minimum {rules['min']}"
                    )

            if "max" in rules and isinstance(value, (int, float)):
                if value > rules["max"]:
                    result.add_warning(
                        f"Key '{key}': value {value} above maximum {rules['max']}"
                    )

            # Valid values check (for categorical)
            if "valid_values" in rules:
                if value not in rules["valid_values"]:
                    result.add_error(
                        f"Key '{key}': value '{value}' not in "
                        f"valid set {rules['valid_values']}"
                    )

        return result
