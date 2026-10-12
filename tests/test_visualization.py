"""
Project ECHO — Visualization Tests
=====================================
Validates that all plot functions execute without errors
and produce output files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from visualization.plot_config import get_output_dir


class TestBeliefPlots:
    """Validates belief state plot generation."""

    def test_trust_vs_cognitive_load_runs(self, short_fps_session):
        from visualization.belief_plots import plot_trust_vs_cognitive_load
        fig = plot_trust_vs_cognitive_load(short_fps_session)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_belief_evolution_runs(self, short_fps_session):
        from visualization.belief_plots import plot_belief_evolution
        fig = plot_belief_evolution(short_fps_session)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_transition_matrix_runs(self):
        from visualization.belief_plots import plot_transition_matrix
        from estimator.src.pomdp_model import POMDPModel
        pomdp = POMDPModel()
        fig = plot_transition_matrix(pomdp.transition)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_belief_distribution_runs(self):
        from visualization.belief_plots import plot_belief_distribution
        from estimator.src.pomdp_model import POMDPModel
        pomdp = POMDPModel()
        fig = plot_belief_distribution(pomdp.belief)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_emotion_distribution_runs(self, short_fps_session):
        from visualization.belief_plots import plot_emotion_distribution
        fig = plot_emotion_distribution(short_fps_session)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestGameParameterPlots:
    """Validates game parameter plot generation."""

    def test_parameter_evolution_runs(self, short_fps_session):
        from visualization.game_parameter_plots import plot_parameter_evolution
        figs = plot_parameter_evolution(short_fps_session, "fps")
        assert isinstance(figs, list) and len(figs) > 0
        import matplotlib.pyplot as plt
        for fig in figs:
            plt.close(fig)

    def test_genre_radar_runs(self, short_fps_session):
        from visualization.game_parameter_plots import plot_genre_radar
        fig = plot_genre_radar(short_fps_session, "fps")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestNeuromotorPlots:
    """Validates neuromotor plot generation."""

    def test_velocity_profile_runs(self):
        from visualization.neuromotor_plots import plot_velocity_profile
        t = np.linspace(0, 1, 100)
        orig = np.sin(t * 10)
        recon = np.sin(t * 10) + np.random.normal(0, 0.1, 100)
        fig = plot_velocity_profile(t, orig, recon, snr_db=25.0)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_snr_bar_chart_runs(self):
        from visualization.neuromotor_plots import plot_snr_bar_chart
        fig = plot_snr_bar_chart([22.0, 25.5, 28.0, 19.0, 30.0])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_iki_distribution_runs(self):
        from visualization.neuromotor_plots import plot_iki_distribution
        iki = np.random.normal(150, 30, 200)
        iki = iki[iki > 0]
        fig = plot_iki_distribution(iki)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDSTERPlots:
    """Validates DSTER model plot generation."""

    def test_confusion_matrix_runs(self):
        from visualization.dster_plots import plot_confusion_matrix
        y_true = np.random.randint(0, 5, 100)
        y_pred = np.random.randint(0, 5, 100)
        fig = plot_confusion_matrix(y_true, y_pred)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_training_curves_runs(self):
        from visualization.dster_plots import plot_training_curves
        losses = [1.0 - 0.02 * i for i in range(50)]
        fig = plot_training_curves(losses)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_tsne_embedding_runs(self):
        from visualization.dster_plots import plot_tsne_embedding
        embeddings = np.random.randn(100, 32)
        labels = np.random.randint(0, 5, 100)
        fig = plot_tsne_embedding(embeddings, labels, perplexity=10)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_roc_curves_runs(self):
        from visualization.dster_plots import plot_roc_curves
        y_true = np.random.randint(0, 5, 200)
        y_scores = np.random.rand(200, 5)
        y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)
        fig = plot_roc_curves(y_true, y_scores)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestSaveFunctionality:
    """Validates that plots can be saved to disk."""

    def test_plot_saves_to_file(self, short_fps_session, tmp_path):
        import visualization.plot_config as pc
        original_dir = pc.DEFAULT_OUTPUT_DIR
        pc.DEFAULT_OUTPUT_DIR = str(tmp_path)

        from visualization.belief_plots import plot_trust_vs_cognitive_load
        plot_trust_vs_cognitive_load(short_fps_session, save_name="test_save")

        expected = tmp_path / "belief" / "test_save.png"
        assert expected.exists(), f"Plot file not found: {expected}"

        pc.DEFAULT_OUTPUT_DIR = original_dir
