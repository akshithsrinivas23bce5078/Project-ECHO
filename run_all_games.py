"""
Project ECHO - All Games Experiment Runner
=============================================
Runs experiments across all 5 game genres and produces
comparative cross-genre analysis plots.

Usage:
    python run_all_games.py --duration 300 --visualize
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from game_framework.session_simulator import SessionSimulator
from game_framework.game_registry import GameRegistry


def run_all_games(duration: int, scenario: str, visualize: bool, seed: int):
    """Run experiments for all game genres and compare."""
    print(f"\n{'='*60}")
    print(f"  Project ECHO - Cross-Genre Experiment")
    print(f"  Duration: {duration}s | Scenario: {scenario}")
    print(f"{'='*60}\n")

    registry = GameRegistry()
    simulator = SessionSimulator(registry=registry, seed=seed)
    genres = registry.list_genres()

    all_sessions = {}

    for genre in genres:
        print(f"\n{'-'*40}")
        print(f"  Simulating: {genre.upper()}")
        print(f"{'-'*40}")

        t0 = time.perf_counter()
        df = simulator.simulate(genre, duration_seconds=duration, scenario=scenario)
        elapsed = (time.perf_counter() - t0) * 1000

        all_sessions[genre] = df

        print(f"  -> {len(df)} timesteps in {elapsed:.1f} ms")
        print(f"  -> Trust:    mean={df['trust_mean'].mean():.3f}")
        print(f"  -> CogLoad:  mean={df['cognitive_load'].mean():.3f}")
        print(f"  -> Emotions: {df['emotion_label'].value_counts().to_dict()}")

        # Save individual session data
        output_dir = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"session_{genre}_{scenario}.csv")
        df.to_csv(csv_path, index=False)

    # ----------------------------------------------------------------
    # Cross-genre analysis
    # ----------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print(f"  Cross-Genre Comparison")
    print(f"{'='*60}\n")

    # Summary table
    print(f"{'Genre':<20} {'Mean Trust':<12} {'Mean CogLoad':<14} {'Dominant Emotion':<18}")
    print(f"{'-'*64}")
    for genre, df in all_sessions.items():
        dominant = df['emotion_label'].value_counts().index[0]
        print(f"{genre:<20} {df['trust_mean'].mean():<12.4f} "
              f"{df['cognitive_load'].mean():<14.4f} {dominant:<18}")

    # ----------------------------------------------------------------
    # Cross-genre plots
    # ----------------------------------------------------------------
    if visualize:
        print("\nGenerating cross-genre comparison plots...")
        _generate_cross_genre_plots(all_sessions)
    else:
        print("\nSkipping visualization (use --visualize to enable)")

    print(f"\n{'='*60}")
    print(f"  All experiments complete!")
    print(f"  Data saved to: results/data/")
    if visualize:
        print(f"  Plots saved to: results/plots/")
    print(f"{'='*60}\n")

    return all_sessions


def _generate_cross_genre_plots(all_sessions):
    """Generate comparative plots across all genres."""
    from visualization.game_parameter_plots import plot_cross_genre_comparison
    from visualization.belief_plots import (
        plot_trust_vs_cognitive_load, plot_emotion_distribution,
    )
    from visualization.plot_config import clear_output_dir
    import pandas as pd

    # Clean out stale plots from previous cross-genre runs
    clear_output_dir("belief")
    clear_output_dir("game_params")

    # Cross-genre parameter comparison
    plot_cross_genre_comparison(all_sessions, save_name="cross_genre_deltas")
    print("  -> Cross-genre parameter deltas")

    # Combined trust-vs-cogload scatter (all genres)
    combined = pd.concat([
        df.assign(genre=genre)
        for genre, df in all_sessions.items()
    ], ignore_index=True)

    plot_trust_vs_cognitive_load(
        combined,
        title="Trust vs. Cognitive Load - All Genres",
        save_name="all_genres_trust_vs_cogload",
    )
    print("  -> Combined trust vs. cognitive load scatter")

    # Combined emotion distribution
    plot_emotion_distribution(
        combined,
        title="Emotion Distribution - All Genres Combined",
        save_name="all_genres_emotion_distribution",
    )
    print("  -> Combined emotion distribution")

    # Per-genre radar charts
    from visualization.game_parameter_plots import plot_genre_radar
    for genre, df in all_sessions.items():
        plot_genre_radar(df, genre, save_name=f"{genre}_radar_cross")
    print("  -> Per-genre radar charts")


def main():
    parser = argparse.ArgumentParser(
        description="Project ECHO - Cross-Genre Experiment Runner",
    )
    parser.add_argument("--duration", type=int, default=300,
                        help="Session duration per genre in seconds (default: 300)")
    parser.add_argument("--scenario", type=str, default="natural",
                        choices=["natural", "struggle", "mastery", "volatile"],
                        help="Simulation scenario (default: natural)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate plots after simulation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()
    run_all_games(args.duration, args.scenario, args.visualize, args.seed)


if __name__ == "__main__":
    main()
