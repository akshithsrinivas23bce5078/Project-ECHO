"""
Project ECHO — Live Interactive Testing Module
=================================================
Runs an interactive session using the laptop webcam to analyze
facial expressions and movement, dynamically adjusting game and 
environmental parameters in real-time.

Press 'q' to quit and generate session plots.
"""

import cv2
import time
import numpy as np
import pandas as pd
import argparse
import os

from vision.webcam_analyzer import WebcamAnalyzer
from game_framework.game_registry import GameRegistry
from game_framework.parameter_modulator import ParameterModulator

def draw_hud(frame, metrics, current_params, genre):
    """Draws a Heads-Up Display on the OpenCV frame."""
    h, w, _ = frame.shape
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 1. State info
    cv2.putText(frame, f"ECHO LIVE - {genre.upper()}", (20, 30), font, 0.8, (255, 255, 255), 2)
    
    emotion = metrics.get('emotion', 'neutral').upper()
    color = (0, 255, 0)
    if emotion == 'FRUSTRATED': color = (0, 0, 255) # Red for panic
    elif emotion == 'FATIGUED': color = (0, 165, 255) # Orange
    
    cv2.putText(frame, f"Emotion: {emotion}", (20, 70), font, 0.7, color, 2)
    cv2.putText(frame, f"Blinks: {metrics.get('blink_rate', 0)}/min", (20, 100), font, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Movement: {metrics.get('movement_energy', 0.0):.1f}", (20, 130), font, 0.6, (200, 200, 200), 1)
    
    # 2. Key game parameters
    cv2.putText(frame, "Environmental & DDA Params:", (w - 350, 30), font, 0.6, (255, 255, 0), 1)
    
    y = 60
    for key in ['music_intensity', 'lighting_brightness', 'enemy_spawn_frequency', 'enemy_accuracy', 'jumpscare_intensity']:
        if key in current_params:
            val = current_params[key]
            # Show bars
            cv2.putText(frame, f"{key[:15]}: {val:.2f}", (w - 350, y), font, 0.5, (255, 255, 255), 1)
            bar_w = int(val * 100)
            cv2.rectangle(frame, (w - 150, y - 10), (w - 150 + bar_w, y), (0, 255, 255), -1)
            y += 25
            
    return frame

def run_live_test(genre="survival_horror"):
    print("\n============================================================")
    print("  Project ECHO - Live Interactive Test")
    print("============================================================")
    print("  Initializing Webcam and MediaPipe...")
    
    analyzer = WebcamAnalyzer()
    analyzer.start()
    
    registry = GameRegistry()
    profile = registry.get_profile(genre)
    modulator = ParameterModulator(smoothing_factor=0.2)
    
    current_params = profile.get_defaults()
    
    # History for plotting
    history = []
    
    print("\n  [READY] Live webcam test started. Press 'q' in the video window to stop.")
    
    # We simulate a base POMDP trust mean since we only have webcam
    trust_mean = 0.5 
    
    start_time = time.time()
    
    while True:
        frame, metrics = analyzer.analyze_frame()
        if frame is None:
            time.sleep(0.01)
            continue
            
        elapsed = time.time() - start_time
        
        # Heuristically map movement and emotion to cognitive load for the modulator
        # High movement = higher cognitive load
        cog_load = min(1.0, 0.3 + (metrics.get("movement_energy", 0) / 20.0))
        if metrics.get("emotion") == "frustrated":
            cog_load = min(1.0, cog_load + 0.3)
            
        # Run parameter modulator
        adj = modulator.compute_adjustments(
            profile=profile,
            trust_mean=trust_mean,
            cognitive_load=cog_load,
            emotion_label=metrics.get("emotion", "neutral"),
            emotion_confidence=1.0, # WebCam gives direct inference
            current_values=current_params,
            timestamp=elapsed
        )
        
        current_params = adj.after
        
        # Log for history
        row = {
            "time": elapsed,
            "trust_mean": trust_mean,
            "cognitive_load": cog_load,
            "emotion_label": metrics.get("emotion", "neutral"),
            "emotion_confidence": 1.0,
            "dda_action": adj.reason,
            "movement_energy": metrics.get("movement_energy", 0.0),
            "blink_rate": metrics.get("blink_rate", 0),
        }
        row.update(current_params)
        history.append(row)
        
        # Render
        frame = draw_hud(frame, metrics, current_params, genre)
        cv2.imshow("ECHO Live Testing", frame)
        
        # Quit check
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    analyzer.stop()
    cv2.destroyAllWindows()
    
    print("\n[FINISHED] Processing session data...")
    df = pd.DataFrame(history)
    
    if len(df) > 0:
        output_dir = os.path.join(os.path.dirname(__file__), "results", "data")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"live_session_{genre}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  -> Live session data saved to {csv_path}")
        
        # Generate plots
        from visualization.game_parameter_plots import plot_parameter_evolution
        figs = plot_parameter_evolution(df, genre, save_name=f"live_{genre}_param_evolution")
        import matplotlib.pyplot as plt
        for fig in figs:
            plt.close(fig)
        print(f"  -> Generated parameter evolution plot in results/plots/")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Live Interactive ECHO session")
    parser.add_argument("--genre", type=str, default="survival_horror", help="Game genre to test")
    args = parser.parse_args()
    
    run_live_test(args.genre)
