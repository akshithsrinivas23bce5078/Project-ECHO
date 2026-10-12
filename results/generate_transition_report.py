"""
Project ECHO — Transition Report Generator
=============================================
Simulates data for the 6x6 State Transition Model and Cognitive DDA Control,
then runs evaluation metrics and generates the 10 transition visualizations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.transition_metrics import TransitionMetricsEvaluator
from visualization.transition_plots import (
    plot_transition_heatmap, plot_transition_graph, plot_state_dda_heatmap,
    plot_clc_vs_transition, plot_snr_vs_transition, plot_dda_change_histogram,
    plot_state_duration_distribution, plot_state_dda_flow
)
from visualization.belief_plots import plot_belief_over_time
from visualization.game_parameter_plots import plot_dda_parameter_evolution_with_states

EMOTION_LABELS = ["angry", "focused", "thinking", "neutral", "joy", "given_up"]


def generate_simulated_data(num_steps=300):
    """Generates simulated data adhering to the 6x6 transition logic."""
    np.random.seed(42)
    
    # Simulate realistic transition matrix T_{ij}
    # F, J, A, T, G, N
    T = np.array([
        [0.60, 0.05, 0.10, 0.20, 0.01, 0.04], # F
        [0.30, 0.50, 0.05, 0.05, 0.02, 0.08], # J
        [0.10, 0.05, 0.50, 0.10, 0.05, 0.20], # A
        [0.40, 0.05, 0.10, 0.40, 0.01, 0.04], # T
        [0.05, 0.02, 0.10, 0.03, 0.60, 0.20], # G
        [0.20, 0.10, 0.10, 0.30, 0.05, 0.25], # N
    ])
    
    # Simulate states sequence
    states = [1] # Start at Focused (index 1)
    for _ in range(num_steps - 1):
        prev = states[-1]
        next_s = np.random.choice(6, p=T[prev])
        states.append(next_s)
        
    states = np.array(states)
    
    # Simulate Belief sequence (noisy versions of true state)
    belief = np.zeros((num_steps, 6))
    for i, s in enumerate(states):
        b = np.random.dirichlet(np.ones(6) * 0.5)
        b[s] += 2.0
        belief[i] = b / np.sum(b)
        
    # Simulate CLC and SNR
    clc = np.random.beta(2, 5, num_steps)
    snr = np.random.normal(5, 2, num_steps)
    
    # Simulate DDA (e.g., L4D Enemy Spawn Rate, 0.1 - 1.0)
    # F: +, J: +, A: -, T: -, G: --, N: 0
    # EMOTION_LABELS = ["angry", "focused", "thinking", "neutral", "joy", "given_up"]
    # F=1, J=4, A=0, T=2, N=3, G=5
    # L4D Enemy Spawn: A=-0.6, F=0.5, T=-0.3, N=0.0, J=0.2, G=-0.9
    param_sens = np.array([-0.6, 0.5, -0.3, 0.0, 0.2, -0.9])
    
    dda_param = np.zeros(num_steps)
    dda_param[0] = 0.5
    dda_deltas = np.zeros(num_steps)
    for i in range(1, num_steps):
        expected_delta = np.dot(belief[i], param_sens) * 0.1 # scaling
        dda_deltas[i] = expected_delta
        dda_param[i] = np.clip(dda_param[i-1] + expected_delta, 0.1, 1.0)
        
    df = pd.DataFrame({
        "time": np.arange(num_steps) * 15, # 15 sec intervals
        "state_idx": states,
        "emotion_label": [EMOTION_LABELS[i] for i in states],
        "clc_raw": clc,
        "snr": snr,
        "enemy_spawn_rate": dda_param,
        "enemy_spawn_delta": dda_deltas
    })
    
    for i, label in enumerate(EMOTION_LABELS):
        df[f"belief_{label}"] = belief[:, i]
        
    return df, T, belief, dda_deltas


def run_evaluation():
    print("Generating simulated data...")
    df, true_T, belief, dda_deltas = generate_simulated_data(num_steps=300)
    
    # Metrics
    evaluator = TransitionMetricsEvaluator()
    states = df["state_idx"].values
    
    print("\n--- Transition Metrics ---")
    sp = evaluator.calculate_state_persistence(states)
    for state, dur in sp.items():
        print(f"Persistence ({state}): {dur:.2f} steps")
        
    stability = evaluator.calculate_transition_stability(states)
    print(f"Transition Stability: {stability:.2f}")
    
    am = evaluator.calculate_adaptation_magnitude(df["enemy_spawn_rate"].values)
    print(f"Adaptation Magnitude (AM): {am:.4f}")
    
    pc = evaluator.calculate_policy_consistency(belief, dda_deltas)
    print(f"Policy Consistency Score: {pc:.4f}")
    
    bounded = evaluator.check_parameter_boundedness(df["enemy_spawn_rate"].values, 0.1, 1.0)
    print(f"Parameter Boundedness Respected: {bounded}")
    
    # Plots
    print("\nGenerating Visualizations...")
    os.makedirs("results/plots/transition", exist_ok=True)
    
    # Vis 1 & 2
    plot_transition_heatmap(true_T, EMOTION_LABELS, save_name="vis1_transition_heatmap")
    plot_transition_graph(true_T, EMOTION_LABELS, save_name="vis2_transition_graph")
    
    # Vis 3
    plot_belief_over_time(df, save_name="vis3_belief_over_time")
    
    # Vis 4
    plot_dda_parameter_evolution_with_states(df, "enemy_spawn_rate", save_name="vis4_dda_evolution")
    
    # Vis 5
    sens_matrix = pd.DataFrame({
        "Enemy Spawn Rate": [-0.6, 0.5, -0.3, 0.0, 0.2, -0.9],
        "Horde Frequency":  [-0.5, 0.4, -0.2, 0.0, 0.0, -0.8],
        "Action Score":     [-0.5, 0.6, -0.4, 0.0, 0.3, -0.9]
    }, index=["Angry", "Focused", "Thinking", "Neutral", "Joy", "Given Up"])
    plot_state_dda_heatmap(sens_matrix, save_name="vis5_state_dda_heatmap")
    
    # Vis 6 & 7 (Simulated response curves)
    clc_vals = np.linspace(0, 1, 100)
    clc_probs = {
        "focused": 1 - clc_vals,
        "thinking": clc_vals * 0.8,
        "given_up": clc_vals**2
    }
    plot_clc_vs_transition(clc_vals, clc_probs, save_name="vis6_clc_vs_transition")
    
    snr_vals = np.linspace(0, 10, 100)
    snr_probs = {
        "focused": snr_vals / 10,
        "neutral": np.exp(-snr_vals/2)
    }
    plot_snr_vs_transition(snr_vals, snr_probs, save_name="vis7_snr_vs_transition")
    
    # Vis 8
    plot_dda_change_histogram(dda_deltas, save_name="vis8_dda_change_hist")
    
    # Vis 9
    dur_dict = {lbl: [] for lbl in EMOTION_LABELS}
    curr = states[0]
    count = 1
    for s in states[1:]:
        if s == curr: count += 1
        else:
            dur_dict[EMOTION_LABELS[curr]].append(count)
            curr = s; count = 1
    dur_dict[EMOTION_LABELS[curr]].append(count)
    plot_state_duration_distribution(dur_dict, save_name="vis9_duration_dist")
    
    # Vis 10
    plot_state_dda_flow(save_name="vis10_flowchart")
    
    print("All transition visualizations generated in results/plots/transition/")


if __name__ == "__main__":
    run_evaluation()
