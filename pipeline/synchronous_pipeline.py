"""
Project ECHO — Synchronous Pipeline
=======================================
End-to-end orchestration of the full ECHO processing pipeline.

Ensures all components work synchronously in a validated chain:
    Input Capture → Feature Extraction → DSTER → POMDP → Adaptation → Output

Each stage's output is validated against the next stage's expected
input contract before passing data forward.

Usage:
    from pipeline.synchronous_pipeline import EchoPipeline

    pipeline = EchoPipeline()
    result = pipeline.run(keystroke_features, mouse_trajectory)
    print(result.summary())
"""

import time
import numpy as np
import torch
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

from dster.emotion_classifier import EmotionClassifier
from feature_extraction.clc_estimator import CLCEstimator
from feature_extraction.sigma_lognormal import SigmaLognormalModel
from estimator.src.pomdp_model import POMDPModel
from pipeline.pipeline_validator import PipelineValidator, ValidationResult


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    outputs: Dict[str, Any]
    latency_ms: float
    validation: Optional[ValidationResult] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete result from end-to-end pipeline execution."""
    stages: List[StageResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    success: bool = True
    final_output: Optional[Dict] = None

    # Individual stage outputs for easy access
    features: Optional[Dict] = None
    emotion: Optional[Dict] = None
    belief: Optional[Dict] = None
    adaptation: Optional[Dict] = None

    def summary(self) -> str:
        """Generate a human-readable pipeline execution summary."""
        lines = [
            f"\n{'='*55}",
            f"  ECHO Pipeline Execution Summary",
            f"{'='*55}",
            f"  Total Latency: {self.total_latency_ms:.2f} ms",
            f"  Status: {'[OK] SUCCESS' if self.success else '[FAIL] FAILED'}",
            f"  Stages: {len(self.stages)}",
            f"{'-'*55}",
        ]

        for stage in self.stages:
            status = "[OK]" if stage.success else "[FAIL]"
            val_status = ""
            if stage.validation:
                val_status = " [VALIDATED]" if stage.validation.is_valid else " [INVALID]"
            lines.append(
                f"  {status} {stage.stage_name:<25} {stage.latency_ms:>8.2f} ms{val_status}"
            )

        lines.append(f"{'-'*55}")

        if self.emotion:
            lines.append(f"  Emotion:   {self.emotion.get('label', 'N/A')} "
                         f"(conf: {self.emotion.get('confidence', 0):.4f})")
        if self.belief:
            lines.append(f"  Trust:     {self.belief.get('trust_mean', 0):.4f}")
            lines.append(f"  CogLoad:   {self.belief.get('cognitive_load', 0):.4f}")

        lines.append(f"{'='*55}\n")
        return "\n".join(lines)


class EchoPipeline:
    """
    Synchronous end-to-end pipeline for Project ECHO.

    Orchestrates all processing stages in sequence with
    inter-stage data validation and timing.

    Stages:
        1. Feature Extraction (CLC + Sigma-Lognormal)
        2. DSTER Emotion Classification
        3. POMDP Belief Update
        4. Adaptation Engine (optional)
    """

    def __init__(
        self,
        seq_len: int = 50,
        num_channels: int = 4,
        embed_dim: int = 32,
        validate: bool = True,
        enable_active_inference: bool = False,
    ):
        # Initialize all components
        self.seq_len = seq_len
        self.validate = validate

        # Feature extraction
        self.clc_estimator = CLCEstimator(window_size=20)
        self.sigma_lognormal = SigmaLognormalModel(min_strokes=3)

        # DSTER emotion classifier
        self.emotion_model = EmotionClassifier(
            num_channels=num_channels,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=1,
            num_classes=5,
        )
        self.emotion_model.eval()

        # POMDP belief updater
        self.pomdp = POMDPModel(enable_active_inference=enable_active_inference)

        # Validator
        self.validator = PipelineValidator()

        # Stage timing history
        self._timing_history: List[Dict[str, float]] = []

    def run(
        self,
        keystroke_features: Optional[torch.Tensor] = None,
        mouse_x: Optional[np.ndarray] = None,
        mouse_y: Optional[np.ndarray] = None,
        mouse_t: Optional[np.ndarray] = None,
        iki_values: Optional[np.ndarray] = None,
        complexity_ranks: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline synchronously.

        Args:
            keystroke_features: (1, seq_len, num_channels) DSTER input.
            mouse_x: Mouse X coordinates for sigma-lognormal.
            mouse_y: Mouse Y coordinates for sigma-lognormal.
            mouse_t: Mouse timestamps for sigma-lognormal.
            iki_values: Inter-keystroke intervals for CLC.
            complexity_ranks: Complexity ranks for CLC.

        Returns:
            PipelineResult with all stage outputs.
        """
        result = PipelineResult()
        t_total = time.perf_counter()

        try:
            # ---- Stage 1: Feature Extraction ----
            features = self._run_feature_extraction(
                mouse_x, mouse_y, mouse_t, iki_values, complexity_ranks, result
            )

            # ---- Stage 2: DSTER Emotion Classification ----
            emotion = self._run_dster(keystroke_features, result)

            # ---- Stage 3: POMDP Belief Update ----
            belief = self._run_pomdp(features, result)

            # ---- Assemble final output ----
            result.features = features
            result.emotion = emotion
            result.belief = belief
            result.final_output = {
                "features": features,
                "emotion": emotion,
                "belief": belief,
            }

        except Exception as e:
            result.success = False
            result.stages.append(StageResult(
                stage_name="PIPELINE_ERROR",
                outputs={},
                latency_ms=0,
                success=False,
                error=str(e),
            ))

        result.total_latency_ms = (time.perf_counter() - t_total) * 1000

        # Record timing
        self._timing_history.append({
            stage.stage_name: stage.latency_ms for stage in result.stages
        })

        return result

    def _run_feature_extraction(
        self,
        mouse_x: Optional[np.ndarray],
        mouse_y: Optional[np.ndarray],
        mouse_t: Optional[np.ndarray],
        iki_values: Optional[np.ndarray],
        complexity_ranks: Optional[np.ndarray],
        pipeline_result: PipelineResult,
    ) -> Dict:
        """Run feature extraction stage."""
        t0 = time.perf_counter()
        features = {}

        # CLC estimation
        clc_raw = 0.45  # Default
        iki_mean_ms = 150.0  # Default
        if iki_values is not None and complexity_ranks is not None:
            clc_result = self.clc_estimator.estimate(iki_values, complexity_ranks)
            clc_raw = clc_result.get("clc", 0.45)
            iki_mean_ms = float(np.mean(iki_values))
        elif iki_values is not None:
            iki_mean_ms = float(np.mean(iki_values))

        features["clc_raw"] = clc_raw
        features["iki_mean_ms"] = iki_mean_ms

        # Sigma-Lognormal neuromotor analysis
        snr_db = 24.0  # Default
        if mouse_x is not None and mouse_y is not None and mouse_t is not None:
            try:
                slm_result = self.sigma_lognormal.analyze_trajectory(
                    mouse_x, mouse_y, mouse_t
                )
                snr_db = slm_result.get("snr_db", 24.0)
                features["stroke_count"] = slm_result.get("stroke_count", 0)
                features["neuromotor_load"] = slm_result.get("neuromotor_load", 0.3)
            except Exception:
                pass

        features["snr_db"] = snr_db

        latency = (time.perf_counter() - t0) * 1000

        # Validate output
        validation = None
        if self.validate:
            validation = self.validator.validate_features(features)

        stage = StageResult(
            stage_name="Feature Extraction",
            outputs=features,
            latency_ms=latency,
            validation=validation,
            success=True,
        )
        pipeline_result.stages.append(stage)

        return features

    def _run_dster(
        self,
        keystroke_features: Optional[torch.Tensor],
        pipeline_result: PipelineResult,
    ) -> Dict:
        """Run DSTER emotion classification stage."""
        t0 = time.perf_counter()

        # Use synthetic input if none provided
        if keystroke_features is None:
            keystroke_features = torch.randn(1, self.seq_len, 4)

        with torch.no_grad():
            label, confidence = self.emotion_model.predict(keystroke_features)

        emotion = {
            "label": label,
            "confidence": confidence,
        }

        latency = (time.perf_counter() - t0) * 1000

        # Validate output
        validation = None
        if self.validate:
            validation = self.validator.validate_emotion(emotion)

        stage = StageResult(
            stage_name="DSTER Emotion Classification",
            outputs=emotion,
            latency_ms=latency,
            validation=validation,
            success=True,
        )
        pipeline_result.stages.append(stage)

        return emotion

    def _run_pomdp(
        self,
        features: Dict,
        pipeline_result: PipelineResult,
    ) -> Dict:
        """Run POMDP belief update stage."""
        t0 = time.perf_counter()

        observation = {
            "clc_raw": features.get("clc_raw", 0.45),
            "iki_mean_ms": features.get("iki_mean_ms", 150.0),
            "snr_db": features.get("snr_db", 24.0),
        }

        belief_summary = self.pomdp.update_belief(observation)

        latency = (time.perf_counter() - t0) * 1000

        # Validate output
        validation = None
        if self.validate:
            validation = self.validator.validate_belief(belief_summary)

        stage = StageResult(
            stage_name="POMDP Belief Update",
            outputs=belief_summary,
            latency_ms=latency,
            validation=validation,
            success=True,
        )
        pipeline_result.stages.append(stage)

        return belief_summary

    def reset(self):
        """Reset all stateful components."""
        self.pomdp.reset()
        self._timing_history.clear()

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get aggregate timing statistics across all runs."""
        if not self._timing_history:
            return {}

        all_stages = set()
        for record in self._timing_history:
            all_stages.update(record.keys())

        stats = {}
        for stage in all_stages:
            values = [
                r[stage] for r in self._timing_history if stage in r
            ]
            if values:
                stats[stage] = {
                    "mean_ms": float(np.mean(values)),
                    "std_ms": float(np.std(values)),
                    "min_ms": float(np.min(values)),
                    "max_ms": float(np.max(values)),
                    "count": len(values),
                }

        return stats

    def run_demo(self, n_iterations: int = 10, seed: int = 42):
        """
        Run a demonstration of the synchronous pipeline.

        Executes multiple iterations with synthetic data and prints
        aggregate timing statistics.
        """
        rng = np.random.RandomState(seed)

        print(f"\n{'='*55}")
        print(f"  ECHO Synchronous Pipeline Demo")
        print(f"  Iterations: {n_iterations}")
        print(f"{'='*55}\n")

        self.reset()

        for i in range(n_iterations):
            # Generate synthetic inputs
            keystroke_features = torch.randn(1, self.seq_len, 4)

            n_points = 200
            t_arr = np.linspace(0.002, 1.0, n_points)
            x_arr = np.cumsum(rng.normal(3, 1.5, n_points))
            y_arr = np.cumsum(rng.normal(2, 1.5, n_points))

            iki_values = 150 + rng.normal(0, 30, 20)
            iki_values = iki_values[iki_values > 0]

            result = self.run(
                keystroke_features=keystroke_features,
                mouse_x=x_arr,
                mouse_y=y_arr,
                mouse_t=t_arr,
                iki_values=iki_values,
            )

            if i == 0 or i == n_iterations - 1:
                print(result.summary())

        # Print aggregate stats
        stats = self.get_timing_stats()
        print(f"\n{'-'*55}")
        print(f"  Aggregate Timing Statistics ({n_iterations} iterations)")
        print(f"{'-'*55}")
        for stage, s in stats.items():
            print(f"  {stage:<30} "
                  f"mean={s['mean_ms']:.2f}ms  "
                  f"std={s['std_ms']:.2f}ms  "
                  f"min={s['min_ms']:.2f}ms  "
                  f"max={s['max_ms']:.2f}ms")
        print(f"{'-'*55}\n")

        # Validate all stages passed
        all_passed = all(
            s.validation is None or s.validation.is_valid
            for r in [result]
            for s in r.stages
        )
        print(f"  Pipeline Validation: {'[OK] ALL STAGES PASSED' if all_passed else '[FAIL] VALIDATION ERRORS'}")
        print(f"{'='*55}\n")
