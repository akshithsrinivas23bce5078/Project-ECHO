"""
Project ECHO — Integration Tests
====================================
End-to-end pipeline tests that exercise the full chain:
    input → feature extraction → DSTER → POMDP → game adaptation → plots

These tests verify that no component crashes when integrated
and that the outputs are structurally correct.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

from feature_extraction.digraph_latency import DigraphExtractor
from feature_extraction.clc_estimator import CLCEstimator
from feature_extraction.sigma_lognormal import SigmaLognormalModel
from dster.emotion_classifier import EmotionClassifier
from estimator.src.pomdp_model import POMDPModel
from estimator.src.belief_updater import BeliefUpdater
from game_framework.game_registry import GameRegistry
from game_framework.parameter_modulator import ParameterModulator
from game_framework.session_simulator import SessionSimulator


class TestEndToEndPipeline:
    """Full pipeline integration tests."""

    def test_feature_to_dster_to_pomdp(self):
        """
        Feature extraction → DSTER inference → POMDP belief update.
        Verifies the core signal pipeline doesn't crash.
        """
        # Step 1: Feature extraction
        extractor = DigraphExtractor()
        timestamps = np.cumsum(np.random.uniform(0.05, 0.3, 100))
        iki = extractor.extract_from_timestamps(timestamps)
        assert len(iki) > 0

        # Step 2: CLC estimation
        clc_est = CLCEstimator(window_size=20)
        clc_raw, clc_nominal = clc_est.compute_clc(iki)

        # Step 3: Sigma-Lognormal
        slm = SigmaLognormalModel(min_strokes=3)
        t = np.linspace(0, 2.0, 200)
        x = np.cumsum(np.random.normal(5, 2, 200))
        y = np.cumsum(np.random.normal(3, 2, 200))
        result = slm.analyze_trajectory(x, y, t)

        # Step 4: DSTER inference
        model = EmotionClassifier(
            num_channels=4, seq_len=50, embed_dim=32,
            num_heads=4, num_layers=1, num_classes=5,
        )
        model.eval()
        dummy = torch.randn(1, 50, 4)
        with torch.no_grad():
            label, conf = model.predict(dummy)

        assert label in ["frustrated", "focused", "fatigued", "neutral", "engaged"]
        assert 0.0 <= conf <= 1.0

        # Step 5: POMDP belief update
        updater = BeliefUpdater()
        belief = updater.update(
            clc_raw=clc_raw,
            clc_is_nominal=clc_nominal,
            iki_mean_ms=float(np.mean(iki)),
            snr_db=result["snr_db"],
            emotion_label=label,
            emotion_confidence=conf,
        )

        assert "trust_mean" in belief
        assert "cognitive_load" in belief
        assert 0.0 <= belief["trust_mean"] <= 1.0
        assert 0.0 <= belief["cognitive_load"] <= 1.0

    def test_belief_to_game_adaptation(self):
        """
        POMDP belief → game parameter modulation.
        Verifies the adaptation chain works for all genres.
        """
        registry = GameRegistry()
        modulator = ParameterModulator(smoothing_factor=1.0)

        for genre in registry.list_genres():
            profile = registry.get_profile(genre)
            adj = modulator.compute_adjustments(
                profile=profile,
                trust_mean=0.6,
                cognitive_load=0.7,
                emotion_label="frustrated",
                emotion_confidence=0.8,
            )

            assert adj.genre == genre
            assert isinstance(adj.after, dict)
            assert len(adj.after) == len(profile.parameters)

            # All values must be within bounds
            for name, val in adj.after.items():
                spec = profile.parameters[name]
                assert spec.min_val <= val <= spec.max_val, (
                    f"{genre}/{name}: {val} out of [{spec.min_val}, {spec.max_val}]"
                )

    def test_full_session_simulation(self):
        """
        Full session simulation → DataFrame → basic analysis.
        """
        simulator = SessionSimulator(seed=42)
        df = simulator.simulate("left_4_dead", duration_seconds=60)

        # Basic structure checks
        assert len(df) == 60
        assert "trust_mean" in df.columns
        assert "cognitive_load" in df.columns
        assert "emotion_label" in df.columns

        # No NaN values
        assert not df.isnull().any().any(), (
            f"NaN values found in columns: {df.columns[df.isnull().any()].tolist()}"
        )

    def test_simulation_to_plots(self):
        """
        Session simulation → plot generation (no crash).
        """
        simulator = SessionSimulator(seed=42)
        df = simulator.simulate("left_4_dead", duration_seconds=30)

        from visualization.belief_plots import plot_trust_vs_cognitive_load
        fig = plot_trust_vs_cognitive_load(df)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

        from visualization.game_parameter_plots import plot_parameter_evolution
        figs = plot_parameter_evolution(df, "left_4_dead")
        assert isinstance(figs, list) and len(figs) > 0
        for fig in figs:
            plt.close(fig)

    def test_vision_pipeline_integration(self, synthetic_frame_pair):
        """
        OpenCV vision pipeline integration test.
        """
        frame1, frame2 = synthetic_frame_pair

        from vision.frame_analyzer import FrameAnalyzer
        from vision.attention_mapper import AttentionMapper
        from vision.motion_analyzer import MotionAnalyzer

        # Frame analysis
        analyzer = FrameAnalyzer()
        f1 = analyzer.analyze_frame(frame1)
        f2 = analyzer.analyze_frame(frame2)
        assert f2["mean_flow_magnitude"] >= 0

        # Attention mapping
        mapper = AttentionMapper()
        xs = np.random.normal(320, 80, 50)
        ys = np.random.normal(240, 60, 50)
        heatmap = mapper.generate_heatmap(xs, ys)
        assert heatmap.shape == (480, 640)

        # Motion analysis
        motion = MotionAnalyzer()
        metrics = motion.compute_motion_energy(frame1, frame2)
        assert "action_level" in metrics

    def test_multi_genre_cross_comparison(self):
        """
        Simulate all genres and verify cross-genre comparison works.
        """
        simulator = SessionSimulator(seed=42)
        all_sessions = simulator.simulate_all_genres(duration_seconds=30)

        assert len(all_sessions) == 4
        for genre, df in all_sessions.items():
            assert len(df) == 30
            assert not df.isnull().any().any()

        # Cross-genre plot
        from visualization.game_parameter_plots import plot_cross_genre_comparison
        fig = plot_cross_genre_comparison(all_sessions)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
