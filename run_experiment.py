"""
Project ECHO - Experiment Runner
===================================
CLI entry point for running a single game genre experiment.

Usage:
    python run_experiment.py --game fps --duration 300 --scenario natural --visualize
    python run_experiment.py --game puzzle --duration 600 --scenario struggle --visualize
    python run_experiment.py --game fps --duration 300 --visualize --evaluate
"""

import argparse
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from game_framework.game_registry import GameRegistry
from game_framework.session_simulator import SessionSimulator
from dster.emotion_classifier import EmotionClassifier
from feature_extraction.sigma_lognormal import SigmaLognormalModel
from feature_extraction.clc_estimator import CLCEstimator
from estimator.src.pomdp_model import POMDPModel


def run_experiment(genre: str, duration: int, scenario: str, visualize: bool, seed: int, evaluate: bool = False):
    """Run a complete experiment for one game genre."""
    print(f"\n{'='*60}")
    print(f"  Project ECHO - Experiment: {genre.upper()}")
    print(f"  Duration: {duration}s | Scenario: {scenario} | Seed: {seed}")
    print(f"{'='*60}\n")

    # ----------------------------------------------------------------
    # Step 1: Simulate game session
    # ----------------------------------------------------------------
    print("[1/5] Simulating game session...")
    t0 = time.perf_counter()

    simulator = SessionSimulator(seed=seed)
    df = simulator.simulate(genre, duration_seconds=duration, scenario=scenario)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  -> Generated {len(df)} timesteps in {elapsed:.1f} ms")
    print(f"  -> Columns: {list(df.columns[:8])}... ({len(df.columns)} total)")

    # ----------------------------------------------------------------
    # Step 2: Run DSTER model on synthetic data
    # ----------------------------------------------------------------
    print("\n[2/5] Running DSTER emotion classifier...")
    t0 = time.perf_counter()

    model = EmotionClassifier(
        num_channels=4, seq_len=50, embed_dim=32,
        num_heads=4, num_layers=1, num_classes=5,
    )
    model.eval()
    dummy = torch.randn(1, 50, 4)
    with torch.no_grad():
        label, conf = model.predict(dummy)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  -> DSTER inference: label={label}, confidence={conf:.4f}")
    print(f"  -> Latency: {elapsed:.2f} ms")

    # ----------------------------------------------------------------
    # Step 3: Run Sigma-Lognormal on synthetic trajectory
    # ----------------------------------------------------------------
    print("\n[3/5] Running Sigma-Lognormal neuromotor analysis...")
    t0 = time.perf_counter()

    slm = SigmaLognormalModel(min_strokes=3)
    rng = np.random.RandomState(seed)
    n_points = 500
    t_arr = np.linspace(0.002, 2.0, n_points)
    x_arr = np.cumsum(rng.normal(5, 2, n_points))
    y_arr = np.cumsum(rng.normal(3, 2, n_points))

    result = slm.analyze_trajectory(x_arr, y_arr, t_arr)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  -> Strokes: {result['stroke_count']}, SNR: {result['snr_db']:.1f} dB")
    print(f"  -> Latency: {elapsed:.2f} ms")

    # ----------------------------------------------------------------
    # Step 4: Run POMDP belief update sequence
    # ----------------------------------------------------------------
    print("\n[4/5] Running POMDP belief updates...")
    t0 = time.perf_counter()

    pomdp = POMDPModel()
    n_updates = min(50, len(df))
    for i in range(n_updates):
        obs = {
            "clc_raw": float(df.iloc[i]["clc_raw"]),
            "iki_mean_ms": 150.0 + float(df.iloc[i]["cognitive_load"]) * 200,
            "snr_db": 24.0 + rng.normal(0, 2),
        }
        summary = pomdp.update_belief(obs)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  -> Final belief: trust={summary['trust_mean']:.4f}, "
          f"cogload={summary['cognitive_load']:.4f}")
    print(f"  -> {n_updates} updates in {elapsed:.2f} ms "
          f"({elapsed/n_updates:.2f} ms/update)")

    # ----------------------------------------------------------------
    # Step 5: Generate plots
    # ----------------------------------------------------------------
    if visualize:
        print("\n[5/6] Generating plots...")
        _generate_plots(df, genre, pomdp, result, slm, t_arr, x_arr, y_arr)
    else:
        print("\n[5/6] Skipping visualization (use --visualize to enable)")

    # ----------------------------------------------------------------
    # Step 6: Evaluation metrics (optional)
    # ----------------------------------------------------------------
    if evaluate:
        print("\n[6/6] Running DSTER evaluation metrics...")
        _run_evaluation(model, seed)
    else:
        print("\n[6/6] Skipping evaluation (use --evaluate to enable)")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Experiment complete: {genre.upper()}")
    print(f"  Session statistics:")
    print(f"    Trust:    mean={df['trust_mean'].mean():.3f}, "
          f"std={df['trust_mean'].std():.3f}")
    print(f"    CogLoad:  mean={df['cognitive_load'].mean():.3f}, "
          f"std={df['cognitive_load'].std():.3f}")
    print(f"    Emotions: {df['emotion_label'].value_counts().to_dict()}")
    print(f"{'='*60}\n")

    # Save session data
    output_dir = os.path.join(os.path.dirname(__file__), "results", "data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"session_{genre}_{scenario}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Session data saved to: {csv_path}")

    return df


def _generate_plots(df, genre, pomdp, slm_result, slm_model, t_arr, x_arr, y_arr):
    """Generate all plots for the experiment."""
    from visualization.belief_plots import (
        plot_trust_vs_cognitive_load, plot_belief_evolution,
        plot_transition_matrix, plot_belief_distribution,
        plot_emotion_distribution,
    )
    from visualization.game_parameter_plots import (
        plot_parameter_evolution, plot_genre_radar,
    )
    from visualization.neuromotor_plots import (
        plot_velocity_profile, plot_iki_distribution,
    )
    from visualization.plot_config import clear_output_dir

    # Clean out stale plots from previous experiment runs
    clear_output_dir("belief")
    clear_output_dir("game_params")
    clear_output_dir("neuromotor")

    prefix = f"{genre}"

    # Belief plots
    plot_trust_vs_cognitive_load(df, save_name=f"{prefix}_trust_vs_cogload")
    print("  -> Trust vs. Cognitive Load scatter")

    plot_belief_evolution(df, save_name=f"{prefix}_belief_evolution")
    print("  -> Belief state evolution")

    plot_transition_matrix(pomdp.transition, save_name=f"{prefix}_transition_matrix")
    print("  -> POMDP transition matrix")

    plot_belief_distribution(pomdp.belief, save_name=f"{prefix}_belief_distribution")
    print("  -> Belief distribution heatmap")

    plot_emotion_distribution(df, save_name=f"{prefix}_emotion_distribution")
    print("  -> Emotion distribution")

    # Game parameter plots
    plot_parameter_evolution(df, genre, save_name=f"{prefix}_param_evolution")
    print("  -> Parameter evolution")

    plot_genre_radar(df, genre, save_name=f"{prefix}_radar")
    print("  -> Genre radar chart")

    # Neuromotor plots
    times, velocities = slm_model.compute_velocity_profile(x_arr, y_arr, t_arr)
    if len(velocities) > 0:
        strokes = slm_model.extract_strokes(times, velocities)
        reconstructed = slm_model.reconstruct_velocity(strokes, times)
        snr = slm_model.compute_snr(velocities, reconstructed)
        plot_velocity_profile(times, velocities, reconstructed, snr,
                              save_name=f"{prefix}_velocity_profile")
        print("  -> Velocity profile")

    iki_values = 150 + np.random.normal(0, 30, 200)
    iki_values = iki_values[iki_values > 0]
    plot_iki_distribution(iki_values, save_name=f"{prefix}_iki_distribution")
    print("  -> IKI distribution")

    print(f"\n  All plots saved to: results/plots/")


def _run_evaluation(model, seed):
    """Run DSTER evaluation metrics and generate evaluation plots."""
    from evaluation.evaluation_metrics import DSTEREvaluator
    from evaluation.evaluation_plots import generate_all_evaluation_plots

    evaluator = DSTEREvaluator(model=model)
    X_eval, y_eval = DSTEREvaluator.generate_synthetic_data(
        num_samples=500, seq_len=50, num_channels=4, seed=seed,
    )
    results = evaluator.evaluate(X=X_eval, y_true=y_eval)

    print(f"  -> Accuracy: {results.accuracy:.4f}")
    print(f"  -> Macro F1: {results.macro_f1:.4f}")
    print(f"  -> Macro AUC: {results.macro_auc:.4f}")

    X_numpy = X_eval.numpy()
    saved = generate_all_evaluation_plots(results=results, X_data=X_numpy, prefix="experiment")
    print(f"\n  -> {len(saved)} evaluation plots saved to results/plots/evaluation/")

    # Save report
    output_dir = os.path.join(os.path.dirname(__file__), "results", "evaluation")
    report_path = evaluator.save_report(results, output_dir, filename="experiment_eval.json")
    print(f"  -> Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Project ECHO - Single Game Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --game fps --duration 300 --visualize
  python run_experiment.py --game puzzle --scenario struggle --visualize
  python run_experiment.py --game fps --visualize --evaluate
        """,
    )
    parser.add_argument("--game", type=str, default="fps",
                        choices=["fps", "puzzle", "strategy", "racing", "survival_horror"],
                        help="Game genre to simulate (default: fps)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Session duration in seconds (default: 300)")
    parser.add_argument("--scenario", type=str, default="natural",
                        choices=["natural", "struggle", "mastery", "volatile"],
                        help="Simulation scenario (default: natural)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate plots after simulation")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run DSTER evaluation metrics and generate plots")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()
    run_experiment(args.game, args.duration, args.scenario, args.visualize,
                   args.seed, args.evaluate)


if __name__ == "__main__":
    main()
