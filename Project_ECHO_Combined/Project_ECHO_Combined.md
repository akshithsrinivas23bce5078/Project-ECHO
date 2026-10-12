# Project ECHO Combined Documentation

## 1. Project Overview & Architecture
Project ECHO (Emotionally Cognizant Heuristic Orchestrator) is a multimodal affective computing framework that leverages computer vision, keystroke dynamics, and neuromotor mouse tracking to infer player emotional states. It uses these inferences to drive a Dynamic Difficulty Adjustment (DDA) engine across four different game genres simultaneously.

### Tech Stack
- **Vision Processing:** OpenCV, MediaPipe Hands/Face
- **Input Monitoring:** `pynput` (Keystrokes, Mouse Trajectories)
- **Machine Learning Core:** PyTorch, Custom DSTER (Deep Spatiotemporal Emotion Recognition) Transformer
- **Neuromotor Analysis:** Sigma-Lognormal Model (SLM) for mouse kinematics
- **State Estimation:** POMDP (Partially Observable Markov Decision Process)
- **Evaluation & Visualization:** Matplotlib, Seaborn, Scikit-learn, Pandas

### Pipeline Architecture
1. **Feature Extraction:** Occurs every 2 seconds. The `WebcamAnalyzer` identifies facial features and hand tracking. The `KeystrokeListener` and `MouseListener` compute Cognitive Load (CLC) and Signal-to-Noise Ratio (SNR).
2. **Classification (DSTER):** Extracts a tensor from recent keystroke events, passing it to the 6-class DSTER model for a preliminary classification.
3. **State Updates & POMDP:** Uses the raw classifier output, alongside visual cues (smiles, chin contact), to calculate the final POMDP belief state.
4. **DDA Modulation:** Every 15 seconds, the `ParameterModulator` updates parameters for all active game environments based on the user's emotional state.

---

## 2. Emotional States & Trigger Mechanisms
The state machine drives the entire project, consisting of 6 explicit states:

1. **`focused`**: The baseline engaged state. Triggered when the user is actively pressing game-bound keys (WASD, Spacebar, Mouse clicks).
2. **`joy`**: Triggered when the user continuously smiles at the webcam for **10 seconds** without interruption.
3. **`angry`**: Triggered when the user aggressively presses non-game keys (bashing the keyboard), introducing obstacles or penalty prompts.
4. **`thinking`**: Triggered when the user rests their chin on their hand (detected via MediaPipe) for **3 seconds**. If maintained for 15 seconds, a contextual hint is displayed on the screen.
5. **`given_up`**: Triggered when the user provides absolutely zero input (mouse or keyboard) for **30 seconds**. Displays a red warning banner.
6. **`neutral`**: The fallback state when input is sparse but not absent, and no specific intense emotion is detected.

---

## 3. Game Visualization & Monitoring
The project simulates monitoring and modulating four diverse game genres simultaneously:

1. **Left 4 Dead (AI Director):** Adjusts enemy spawn rates, health item scarcity, and horde frequency.
2. **Alien: Isolation (Menace Gauge):** Adjusts tension budget, alien search radius, and noise sensitivity.
3. **Resident Evil 4 (Action Scoring):** Modifies enemy aggression, damage multipliers, and action scoring limits.
4. **Silent Hill (Atmosphere):** Controls world decay speed, puzzle obfuscation, and static interference.

### Monitoring Mechanisms
- **Visual Parameters:** MediaPipe evaluates facial landmarks to detect smiling and calculates the distance between wrist/finger landmarks and the chin for the "Thinking" state.
- **Keystrokes:** Inter-Key Interval (IKI) and Key Press Speed (KPS) evaluate cognitive load. A high ratio of non-game keys to game keys signals frustration/anger.
- **Mouse Kinematics:** Trajectories are parsed into strokes and evaluated against the Sigma-Lognormal model to detect neuromotor jitter (indicative of stress).

---

## 4. Evaluation Methodology & Test Suite
When the user exits the session (pressing 'q'), the pipeline evaluates the session. It cross-references the live demo's 6-class state machine truth (`y_true`) against the model's confidence arrays (`y_scores`) to grade the system.

### Performance Heatmaps & Matrices
The test suite automatically generates ROC curves, Confusion Matrices, and Precision/Recall calculations.

#### Confusion Matrix & Error Analysis
![Confusion Matrix](../results/plots/evaluation/unified_v2_confusion_matrix.png)
![Error Analysis](../results/plots/evaluation/unified_v2_error_analysis.png)

#### Prediction Confidence & Distribution
![Confidence Histogram](../results/plots/evaluation/unified_v2_confidence_histogram.png)
![Emotion Pie Chart](../results/plots/evaluation/unified_v2_emotion_pie_chart.png)

#### F1 Scores and Precision-Recall
![F1 Scores](../results/plots/evaluation/unified_v2_f1_scores.png)
![Precision Recall F1](../results/plots/evaluation/unified_v2_precision_recall_f1.png)
![PR Curves](../results/plots/evaluation/unified_v2_pr_curves.png)

#### ROC Curves & Calibration
![ROC Curves](../results/plots/evaluation/unified_v2_roc_curves.png)
![Calibration](../results/plots/evaluation/unified_v2_calibration.png)

---

## 5. Session State & Parameter Evolution
The POMDP tracks the belief state dynamically over time. As the state changes, the DDA parameters across all 4 games modulate instantly.

#### Session State Distribution (Belief)
![State Distribution](../results/plots/belief/unified_v2_state_distribution.png)

#### DDA Parameter Evolution (Examples)
*Left 4 Dead (Enemy Spawn Rate)*
![L4D Spawn Rate](../results/plots/game_params/unified_v2_left_4_dead_param_evolution_enemy_spawn_rate.png)

*Left 4 Dead (Health Item Scarcity)*
![L4D Health Scarcity](../results/plots/game_params/unified_v2_left_4_dead_param_evolution_health_item_scarcity.png)

*Left 4 Dead (Horde Frequency)*
![L4D Horde Frequency](../results/plots/game_params/unified_v2_left_4_dead_param_evolution_horde_frequency.png)

*Left 4 Dead (Special Infected Cap)*
![L4D Special Cap](../results/plots/game_params/unified_v2_left_4_dead_param_evolution_special_infected_cap.png)

*Left 4 Dead (Wanderer Density)*
![L4D Wanderer Density](../results/plots/game_params/unified_v2_left_4_dead_param_evolution_wanderer_density.png)

*Alien: Isolation (Menace Gauge)*
![Alien Menace Gauge](../results/plots/game_params/unified_v2_alien_isolation_param_evolution_alien_menace_gauge.png)

*Alien: Isolation (Alien Search Radius)*
![Alien Search Radius](../results/plots/game_params/unified_v2_alien_isolation_param_evolution_alien_search_radius.png)

*Alien: Isolation (Noise Sensitivity)*
![Alien Noise Sensitivity](../results/plots/game_params/unified_v2_alien_isolation_param_evolution_noise_sensitivity.png)

*Alien: Isolation (Tension Budget)*
![Alien Tension Budget](../results/plots/game_params/unified_v2_alien_isolation_param_evolution_tension_budget.png)

*Alien: Isolation (Vent Drop Probability)*
![Alien Vent Drop](../results/plots/game_params/unified_v2_alien_isolation_param_evolution_vent_drop_probability.png)

*Resident Evil 4 (Action Score)*
![RE4 Action Score](../results/plots/game_params/unified_v2_resident_evil_4_param_evolution_action_score.png)

*Resident Evil 4 (Damage Taken Multiplier)*
![RE4 Damage Taken](../results/plots/game_params/unified_v2_resident_evil_4_param_evolution_damage_taken_mult.png)

*Resident Evil 4 (Enemy Accuracy)*
![RE4 Enemy Accuracy](../results/plots/game_params/unified_v2_resident_evil_4_param_evolution_enemy_accuracy.png)

*Resident Evil 4 (Enemy Aggression)*
![RE4 Enemy Aggression](../results/plots/game_params/unified_v2_resident_evil_4_param_evolution_enemy_aggression.png)

*Resident Evil 4 (Item Drop Quality)*
![RE4 Item Drop Quality](../results/plots/game_params/unified_v2_resident_evil_4_param_evolution_item_drop_quality.png)

*Silent Hill (World Decay Speed)*
![Silent Hill World Decay](../results/plots/game_params/unified_v2_silent_hill_param_evolution_world_decay_speed.png)

*Silent Hill (Environment Lighting)*
![Silent Hill Lighting](../results/plots/game_params/unified_v2_silent_hill_param_evolution_environment_lighting.png)

*Silent Hill (Monster Type Bias)*
![Silent Hill Monster Bias](../results/plots/game_params/unified_v2_silent_hill_param_evolution_monster_type_bias.png)

*Silent Hill (Puzzle Obfuscation)*
![Silent Hill Puzzle Obfuscation](../results/plots/game_params/unified_v2_silent_hill_param_evolution_puzzle_obfuscation.png)

*Silent Hill (Static Interference)*
![Silent Hill Static Interference](../results/plots/game_params/unified_v2_silent_hill_param_evolution_static_interference.png)

#### Neuromotor Kinematics Analysis
*Sigma-Lognormal Mouse Velocity Profile*
![Velocity Profile](../results/plots/neuromotor/fps_velocity_profile.png)
Demonstrates the extraction of individual neuromotor strokes from high-frequency mouse movements, used to estimate the user's Signal-to-Noise Ratio (SNR) and underlying stress.

*Inter-Key Interval (IKI) Distribution*
![IKI Distribution](../results/plots/neuromotor/fps_iki_distribution.png)
Visualizes the raw keystroke rhythm over time, which forms the basis for calculating Cognitive Load Complexity (CLC).

---

## 6. State Transition and Cognitive DDA Control Model

In order to analyze how the player moves between emotional states over time and how the Dynamic Difficulty Adjustment (DDA) parameters holistically react, we present the State Transition analysis. 

#### Transition Matrices and State Flow
*State and DDA Control Flowchart*
![Flowchart](../results/plots/transition/vis10_flowchart.png)

*Macro State Transition Heatmap*
![Transition Heatmap](../results/plots/transition/vis1_transition_heatmap.png)

*State Transition Directed Graph*
![Transition Graph](../results/plots/transition/vis2_transition_graph.png)

*State Duration Distribution*
![Duration Dist](../results/plots/transition/vis9_duration_dist.png)

#### DDA Parameter Impact and Theoretical Curves
*Belief Probabilities Over Time*
![Belief Over Time](../results/plots/transition/vis3_belief_over_time.png)

*Global DDA Parameter Evolution*
![DDA Evolution](../results/plots/transition/vis4_dda_evolution.png)

*State × DDA Parameter Heatmap*
![DDA Heatmap](../results/plots/transition/vis5_state_dda_heatmap.png)

*DDA Parameter Change Distribution*
![Change Hist](../results/plots/transition/vis8_dda_change_hist.png)

*Cognitive Load Complexity vs Transition*
![CLC vs Transition](../results/plots/transition/vis6_clc_vs_transition.png)

*Signal-to-Noise Ratio vs Transition*
![SNR vs Transition](../results/plots/transition/vis7_snr_vs_transition.png)

---

## 7. Raw Data Outputs
All evaluations are backed by real-time CSV and JSON generations:
- **Master Session Log:** `../results/data/unified_session_v2_master.csv`
- **Evaluation Report:** `../results/data/unified_v2_eval.json`
