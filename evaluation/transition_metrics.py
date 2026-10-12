"""
Project ECHO — Transition & Cognitive DDA Control Metrics
=============================================================
Implementation of the evaluation metrics proposed for the 6x6
State Transition Matrix and Belief-Weighted DDA Control.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

EMOTION_LABELS = ["angry", "focused", "thinking", "neutral", "joy", "given_up"]


class TransitionMetricsEvaluator:
    """Computes transition and DDA specific metrics."""
    
    def __init__(self, labels: Optional[List[str]] = None):
        self.labels = labels or EMOTION_LABELS
        self.num_classes = len(self.labels)

    def calculate_transition_accuracy(self, y_true_transitions: np.ndarray, y_pred_transitions: np.ndarray) -> float:
        """
        Transition Accuracy (TA) = Correct State Transitions / Total State Transitions.
        Expects 1D arrays of integer label indices representing the transitioned-to state.
        """
        if len(y_true_transitions) == 0: return 0.0
        return np.mean(y_true_transitions == y_pred_transitions)

    def calculate_transition_log_likelihood(self, y_true_transitions: np.ndarray, predicted_probs: np.ndarray) -> float:
        """
        Transition Log-Likelihood: Sum of log P(S_{t+1} | S_t, O_t)
        y_true_transitions: 1D array of next states.
        predicted_probs: 2D array of (N, num_classes) where each row is the predicted next-state distribution.
        """
        epsilon = 1e-9
        ll = 0.0
        for i, true_state in enumerate(y_true_transitions):
            prob = predicted_probs[i, true_state]
            ll += np.log(max(prob, epsilon))
        return ll

    def calculate_transition_entropy(self, transition_matrix: np.ndarray) -> float:
        """
        Transition Entropy: - sum_j T_ij log T_ij (mean across all states i)
        transition_matrix: 2D array (6x6)
        """
        epsilon = 1e-9
        entropies = []
        for row in transition_matrix:
            h = -np.sum(row * np.log(np.maximum(row, epsilon)))
            entropies.append(h)
        return float(np.mean(entropies))
        
    def calculate_state_persistence(self, state_sequence: np.ndarray) -> Dict[str, float]:
        """
        Mean duration per state (measured in number of contiguous steps).
        """
        if len(state_sequence) == 0:
            return {lbl: 0.0 for lbl in self.labels}
            
        durations = {lbl: [] for lbl in self.labels}
        
        current_state = state_sequence[0]
        current_duration = 1
        
        for state in state_sequence[1:]:
            if state == current_state:
                current_duration += 1
            else:
                durations[self.labels[current_state]].append(current_duration)
                current_state = state
                current_duration = 1
        durations[self.labels[current_state]].append(current_duration)
        
        return {lbl: float(np.mean(d)) if d else 0.0 for lbl, d in durations.items()}

    def calculate_transition_stability(self, state_sequence: np.ndarray) -> float:
        """
        Transition Stability: Measures frequency of A -> B -> A (oscillation).
        Returns a normalized score (0 to 1), where 1 means no oscillations.
        """
        if len(state_sequence) < 3: return 1.0
        
        oscillations = 0
        for i in range(len(state_sequence) - 2):
            if state_sequence[i] != state_sequence[i+1] and state_sequence[i] == state_sequence[i+2]:
                oscillations += 1
                
        max_possible_oscillations = len(state_sequence) - 2
        oscillation_rate = oscillations / max_possible_oscillations
        return 1.0 - oscillation_rate

    def calculate_adaptation_magnitude(self, parameter_sequence: np.ndarray) -> float:
        """
        Adaptation Magnitude (AM) = || p_{t+1} - p_t ||
        """
        if len(parameter_sequence) < 2: return 0.0
        diffs = np.abs(np.diff(parameter_sequence))
        return float(np.mean(diffs))

    def calculate_dda_responsiveness(self, state_sequence: np.ndarray, parameter_sequence: np.ndarray) -> float:
        """
        Simulated Responsiveness: Correlation between state change events and significant parameter changes.
        In a real scenario, this would measure actual time delay.
        """
        state_changes = np.concatenate([[0], np.diff(state_sequence) != 0]).astype(float)
        param_changes = np.concatenate([[0], np.abs(np.diff(parameter_sequence)) > 1e-3]).astype(float)
        
        if np.sum(state_changes) == 0 or np.sum(param_changes) == 0:
            return 0.0
            
        return float(np.corrcoef(state_changes, param_changes)[0, 1])
        
    def calculate_policy_consistency(self, belief_states: np.ndarray, dda_deltas: np.ndarray) -> float:
        """
        Variance of delta_p for similar belief states (b(s)).
        Lower variance means higher consistency. Returns inverse variance (higher is better).
        """
        # Very simple clustering/binning based on dominant state
        dominant_states = np.argmax(belief_states, axis=1)
        variances = []
        for i in range(self.num_classes):
            mask = dominant_states == i
            if np.sum(mask) > 1:
                variances.append(np.var(dda_deltas[mask]))
                
        if not variances: return 0.0
        mean_var = np.mean(variances)
        return float(1.0 / (1.0 + mean_var))

    def check_parameter_boundedness(self, parameter_sequence: np.ndarray, p_min: float, p_max: float) -> bool:
        """
        Checks if p_min <= p(t) <= p_max for all t.
        """
        if len(parameter_sequence) == 0: return True
        return bool(np.all((parameter_sequence >= p_min - 1e-6) & (parameter_sequence <= p_max + 1e-6)))
