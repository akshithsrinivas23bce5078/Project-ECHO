"""
Project ECHO — Evaluation Runner
====================================
CLI entry point for running the complete DSTER evaluation pipeline.

Generates synthetic data (or loads real data), runs model inference,
computes all supervised learning metrics, generates all plots, and
saves a structured JSON evaluation report.

Usage:
    python evaluation/run_evaluation.py --all-plots --save-report
    python evaluation/run_evaluation.py --num-samples 2000 --seed 42
    python evaluation/run_evaluation.py --cross-validate --folds 5
"""

import argparse
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from dster.emotion_classifier import EmotionClassifier
from evaluation.evaluation_metrics import (
    DSTEREvaluator,
    compute_cross_validation_scores,
    EMOTION_LABELS,
)
from evaluation.evaluation_plots import generate_all_evaluation_plots
from changelog.changelog_manager import ChangeLogger




def run_evaluation(
    num_samples: int = 1000,
    seed: int = None,
    all_plots: bool = True,
    save_report: bool = True,
    cross_validate: bool = False,
    n_folds: int = 5,
):
    """Run the complete DSTER evaluation pipeline."""
    # Use a time-based seed when none is provided so each run produces
    # genuinely different data — preventing stale/identical plots.
    if seed is None:
        seed = int(time.time()) % (2**31)

    print("\n" + "="*60)
    print("  Project ECHO - DSTER Evaluation Pipeline")
    print("  ML Type: Supervised Classification (6-class)")
    print(f"  Samples: {num_samples} | Seed: {seed}")
    print("="*60 + "\n")

    # Initialize change logger
    logger = ChangeLogger()

    # ----------------------------------------------------------------
    # Step 1: Initialize model
    # ----------------------------------------------------------------
    print("[1/6] Initializing DSTER Emotion Classifier...")
    t0 = time.perf_counter()

    model = EmotionClassifier(
        num_channels=4, seq_len=50, embed_dim=32,
        num_heads=4, num_layers=1, num_classes=len(EMOTION_LABELS),
    )
    model.eval()

    elapsed = (time.perf_counter() - t0) * 1000
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  -> Model initialized: {total_params:,} parameters ({elapsed:.1f} ms)")

    # ----------------------------------------------------------------
    # Step 2: Generate evaluation data
    # ----------------------------------------------------------------
    print("\n[1/6] Loading synthetic data for evaluation...")
    print("  -> (In a real scenario, this would load the evaluation dataset)")
    X, y_true = DSTEREvaluator.generate_synthetic_data(
        num_samples=num_samples, seq_len=50, num_channels=4, seed=seed
    )

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  -> Generated {len(y_true)} samples ({elapsed:.1f} ms)")
    print(f"  -> Class distribution: {dict(zip(EMOTION_LABELS, np.bincount(y_true, minlength=len(EMOTION_LABELS))))}")

    # ----------------------------------------------------------------
    # Step 3: Run evaluation
    # ----------------------------------------------------------------
    print("\n[3/6] Computing evaluation metrics...")
    t0 = time.perf_counter()

    evaluator = DSTEREvaluator(model=model)
    results = evaluator.evaluate(X=X, y_true=y_true)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  -> Evaluation complete ({elapsed:.1f} ms)")
    print("\n  +-----------------------------------------------+")
    print("  |  EVALUATION RESULTS SUMMARY                    |")
    print("  +-----------------------------------------------+")
    print(f"  |  Accuracy:              {results.accuracy:.4f}               |")
    print(f"  |  Macro F1:              {results.macro_f1:.4f}               |")
    print(f"  |  Weighted F1:           {results.weighted_f1:.4f}               |")
    print(f"  |  Cohen's Kappa:         {results.cohen_kappa:.4f}               |")
    print(f"  |  Matthews CC:           {results.matthews_corrcoef:.4f}               |")
    print(f"  |  Macro AUC:             {results.macro_auc:.4f}               |")
    print(f"  |  Mean Conf (correct):   {results.mean_confidence_correct:.4f}               |")
    print(f"  |  Mean Conf (incorrect): {results.mean_confidence_incorrect:.4f}               |")
    print("  +-----------------------------------------------+")

    print("\n  Per-Class Metrics:")
    print("  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12} {'Support':<10}")
    print("  {'-'*73}")
    for label in results.labels:
        p = results.per_class_precision.get(label, 0)
        r = results.per_class_recall.get(label, 0)
        f = results.per_class_f1.get(label, 0)
        a = results.per_class_auc.get(label, 0)
        s = results.per_class_support.get(label, 0)
        print(f"  {label:<15} {p:<12.4f} {r:<12.4f} {f:<12.4f} {a:<12.4f} {s:<10}")

    # ----------------------------------------------------------------
    # Step 4: Cross-validation (optional)
    # ----------------------------------------------------------------
    cv_results = None
    if cross_validate:
        print(f"\n[4/6] Running {n_folds}-fold cross-validation...")
        t0 = time.perf_counter()

        cv_results = compute_cross_validation_scores(
            model_class=EmotionClassifier,
            model_kwargs={
                "num_channels": 4, "seq_len": 50, "embed_dim": 32,
                "num_heads": 4, "num_layers": 1, "num_classes": 5,
            },
            X=X, y=y_true, n_folds=n_folds, seed=seed,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  -> Cross-validation complete ({elapsed:.1f} ms)")
        print(f"  -> Mean Accuracy: {cv_results['mean_accuracy']:.4f} +/- {cv_results['std_accuracy']:.4f}")
        print(f"  -> Mean Macro F1: {cv_results['mean_f1']:.4f} +/- {cv_results['std_f1']:.4f}")
    else:
        print("\n[4/6] Skipping cross-validation (use --cross-validate to enable)")

    # ----------------------------------------------------------------
    # Step 5: Generate plots
    # ----------------------------------------------------------------
    if all_plots:
        print("\n[5/6] Generating evaluation plots...")

        # NOTE: Learning curves are only generated when real training
        # history data is available.  We do NOT fabricate synthetic training
        # histories — that would produce hallucinated plots.

        X_numpy = X.numpy() if isinstance(X, torch.Tensor) else X
        saved_plots = generate_all_evaluation_plots(
            results=results,
            X_data=X_numpy,
            cv_results=cv_results,
            training_history=None,
            prefix="dster",
        )

        print(f"\n  -> {len(saved_plots)} plots saved to results/plots/evaluation/")
    else:
        print("\n[5/6] Skipping plot generation (use --all-plots to enable)")

    # ----------------------------------------------------------------
    # Step 6: Save report and log changes
    # ----------------------------------------------------------------
    if save_report:
        print("\n[6/6] Saving evaluation report...")

        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "results", "evaluation"
        )
        report_path = evaluator.save_report(results, output_dir)
        print(f"  -> Report saved to: {report_path}")
    else:
        print("\n[6/6] Skipping report save (use --save-report to enable)")

    # Log changes
    logger.log("metric", "Ran DSTER evaluation pipeline",
               files=["evaluation/run_evaluation.py"])
    logger.log("visualization", f"Generated {len(saved_plots) if all_plots else 0} evaluation plots",
               files=["evaluation/evaluation_plots.py"])

    if save_report:
        logger.log("metric", "Saved evaluation report (JSON)",
                    files=["results/evaluation/evaluation_report.json"])

    logger.create_release(
        version="2.0.0",
        title="Supervised Learning Evaluation Metrics & Synchronous Pipeline",
    )
    logger.save()
    print("\n  -> Changelog updated")

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Evaluation Pipeline Complete!")
    print("  ML Type:   Supervised Classification")
    print("  Model:     DSTER Dual-Stream Transformer")
    print(f"  Classes:   {', '.join(EMOTION_LABELS)}")
    print(f"  Accuracy:  {results.accuracy:.4f}")
    print(f"  Macro F1:  {results.macro_f1:.4f}")
    print(f"  Macro AUC: {results.macro_auc:.4f}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Project ECHO - DSTER Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/run_evaluation.py --all-plots --save-report
  python evaluation/run_evaluation.py --num-samples 2000 --cross-validate
  python evaluation/run_evaluation.py --folds 10 --cross-validate --all-plots
        """,
    )
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of evaluation samples (default: 1000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: time-based for fresh data each run)")
    parser.add_argument("--all-plots", action="store_true",
                        help="Generate all evaluation plots")
    parser.add_argument("--save-report", action="store_true",
                        help="Save JSON evaluation report")
    parser.add_argument("--cross-validate", action="store_true",
                        help="Run K-fold cross-validation")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")

    args = parser.parse_args()

    run_evaluation(
        num_samples=args.num_samples,
        seed=args.seed,
        all_plots=args.all_plots,
        save_report=args.save_report,
        cross_validate=args.cross_validate,
        n_folds=args.folds,
    )


if __name__ == "__main__":
    main()
