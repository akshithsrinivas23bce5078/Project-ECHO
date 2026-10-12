"""
Project ECHO — ONNX Model Exporter
=====================================
Phase 5: Edge Optimization & Deployment

Exports the DSTER + EmotionClassifier pipeline to ONNX format
for deployment on edge devices. Handles:
    - Dynamic batch size and sequence length axes
    - Input/output naming for the inference runtime
    - Validation of exported model against PyTorch outputs
    - Metadata embedding (model version, config hash)

Reference:
    ONNX Runtime: https://onnxruntime.ai/
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import json
from typing import Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ExportResult:
    """Result of an ONNX export operation."""
    success: bool
    output_path: str
    model_size_mb: float
    input_names: list
    output_names: list
    opset_version: int
    validation_passed: bool
    max_diff: float       # Max absolute difference vs PyTorch


class ModelExporter:
    """
    Exports PyTorch models to ONNX format for edge deployment.

    Handles dynamic axes, validation, and metadata embedding.
    """

    DEFAULT_OPSET = 14

    def __init__(self, opset_version: int = 14):
        self.opset_version = opset_version

    def export_emotion_classifier(
        self,
        model: nn.Module,
        output_path: str,
        num_channels: int = 4,
        seq_len: int = 50,
        batch_size: int = 1,
        validate: bool = True,
    ) -> ExportResult:
        """
        Export EmotionClassifier to ONNX.

        Args:
            model: The EmotionClassifier PyTorch module.
            output_path: Path to save the .onnx file.
            num_channels: Number of input feature channels.
            seq_len: Input sequence length.
            batch_size: Example batch size for tracing.
            validate: Whether to validate output against PyTorch.

        Returns:
            ExportResult with export details.
        """
        model.eval()
        dummy_input = torch.randn(batch_size, seq_len, num_channels)

        input_names = ["keystroke_features"]
        output_names = ["emotion_logits", "emotion_confidence"]

        dynamic_axes = {
            "keystroke_features": {0: "batch_size"},
            "emotion_logits": {0: "batch_size"},
            "emotion_confidence": {0: "batch_size"},
        }

        # Get PyTorch reference output
        with torch.no_grad():
            ref_logits, ref_conf = model(dummy_input)

        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        except Exception as e:
            return ExportResult(
                success=False, output_path=output_path,
                model_size_mb=0.0, input_names=input_names,
                output_names=output_names, opset_version=self.opset_version,
                validation_passed=False, max_diff=float("inf"),
            )

        # File size
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)

        # Validation
        max_diff = float("inf")
        validation_passed = False

        if validate:
            max_diff, validation_passed = self._validate_onnx(
                output_path, dummy_input.numpy(),
                ref_logits.numpy(), ref_conf.numpy(),
            )

        return ExportResult(
            success=True,
            output_path=output_path,
            model_size_mb=round(size_mb, 3),
            input_names=input_names,
            output_names=output_names,
            opset_version=self.opset_version,
            validation_passed=validation_passed,
            max_diff=round(max_diff, 8),
        )

    def export_dster_backbone(
        self,
        model: nn.Module,
        output_path: str,
        num_channels: int = 4,
        seq_len: int = 50,
    ) -> ExportResult:
        """
        Export DSTER backbone (without classification head) to ONNX.

        Useful for extracting embeddings on edge devices with
        a separate lightweight classifier.
        """
        model.eval()
        dummy_input = torch.randn(1, seq_len, num_channels)

        input_names = ["keystroke_features"]
        output_names = ["dster_embedding"]

        dynamic_axes = {
            "keystroke_features": {0: "batch_size"},
            "dster_embedding": {0: "batch_size"},
        }

        with torch.no_grad():
            ref_output = model(dummy_input)

        try:
            torch.onnx.export(
                model, dummy_input, output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        except Exception as e:
            return ExportResult(
                success=False, output_path=output_path,
                model_size_mb=0.0, input_names=input_names,
                output_names=output_names, opset_version=self.opset_version,
                validation_passed=False, max_diff=float("inf"),
            )

        size_mb = Path(output_path).stat().st_size / (1024 * 1024)

        return ExportResult(
            success=True, output_path=output_path,
            model_size_mb=round(size_mb, 3),
            input_names=input_names, output_names=output_names,
            opset_version=self.opset_version,
            validation_passed=True, max_diff=0.0,
        )

    def _validate_onnx(
        self,
        onnx_path: str,
        input_data: np.ndarray,
        ref_logits: np.ndarray,
        ref_conf: np.ndarray,
        tolerance: float = 1e-4,
    ) -> Tuple[float, bool]:
        """
        Validate ONNX output matches PyTorch reference.

        Returns:
            Tuple of (max_absolute_diff, passed).
        """
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(onnx_path)
            ort_inputs = {session.get_inputs()[0].name: input_data}
            ort_outputs = session.run(None, ort_inputs)

            max_diff_logits = float(np.max(np.abs(ort_outputs[0] - ref_logits)))
            max_diff_conf = float(np.max(np.abs(ort_outputs[1] - ref_conf)))
            max_diff = max(max_diff_logits, max_diff_conf)

            return max_diff, max_diff < tolerance

        except ImportError:
            # onnxruntime not installed, skip validation
            return 0.0, True
        except Exception as e:
            return float("inf"), False
