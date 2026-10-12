"""
Project ECHO — Embedded Interactive Demo (v2)
===============================================
A fully embedded Python application that combines:
1. Visual Panic Detection (Webcam + chin-contact)
2. Keystroke Dynamics (explicit game-key mapping)
3. Dynamic Difficulty Adjustment (DDA) — 3 states + 2 modifiers
4. Game-Specific Hint Engine (scaffolding)
5. Procedural Calm Music Generation (winsound)

Emotion Model:
    Core:  Normal | Joy | Frustrated
    Mods:  Thinking (idle 10s + chin contact) | Given-Up (idle 16s)

Press 'q' to quit.
"""

import cv2
import time
import numpy as np
import threading
import os

try:
    import winsound
except ImportError:
    winsound = None

from vision.webcam_analyzer import WebcamAnalyzer
from input_capture.keystroke_listener import KeystrokeListener
from game_framework.game_registry import GameRegistry
from game_framework.parameter_modulator import ParameterModulator
from game_framework.hint_engine import HintEngine

# ---------------------------------------------------------
# CALM MUSIC GENERATOR
# ---------------------------------------------------------
class CalmMusicPlayer:
    def __init__(self):
        self.playing = False
        self.thread = None
        self.stop_event = threading.Event()

    def _play_loop(self):
        # Music disabled per user request
        while not self.stop_event.is_set():
            time.sleep(1.0)

    def start(self):
        if not self.playing:
            self.playing = True
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._play_loop, daemon=True)
            self.thread.start()

    def stop(self):
        if self.playing:
            self.playing = False
            self.stop_event.set()


# ---------------------------------------------------------
# HUD DRAWING
# ---------------------------------------------------------
def draw_hud(
    frame, vision_metrics, kb_metrics, current_params,
    music_playing, emotion, state_desc,
    fps=0.0, countdown=16.0, profiles=None,
    hint_text=None, hint_tier=None,
    thinking_seconds=0.0, idle_seconds=0.0,
    non_game_ratio=0.0,
):
    """Draw the full HUD overlay on the webcam canvas."""
    h, w, _ = frame.shape

    # Wider canvas for game params on the right
    canvas_w = max(w + 420, 1100)
    canvas_h = max(h, 650)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_s = cv2.FONT_HERSHEY_PLAIN

    # ---- LEFT SIDE: Translucent overlay for text ----
    ov_h = 300
    ov_w = min(350, w)
    roi = canvas[0:ov_h, 0:ov_w]
    black_rect = np.zeros_like(roi)
    canvas[0:ov_h, 0:ov_w] = cv2.addWeighted(roi, 0.55, black_rect, 0.45, 0)

    # Title
    cv2.putText(canvas, "PROJECT ECHO - DDA ENGINE v2", (15, 28), font, 0.6, (255, 255, 255), 2)

    # Emotion color mapping (3 core + 2 modifiers)
    color_map = {
        'joy':        (255, 105, 180),   # Pink
        'frustrated': (0, 80, 255),      # Orange-red
        'thinking':   (255, 200, 0),     # Cyan-gold
        'given_up':   (100, 100, 255),   # Muted red
        'normal':     (0, 255, 0),       # Green
    }
    color = color_map.get(emotion, (0, 255, 0))

    # State display
    cv2.putText(canvas, f"STATE: {emotion.upper()}", (15, 60), font, 0.75, color, 2)
    cv2.putText(canvas, state_desc, (15, 82), font, 0.38, (200, 200, 200), 1)

    # Metrics
    y_m = 110
    cv2.putText(canvas, f"Movement Energy: {vision_metrics.get('movement_energy', 0):.1f}", (15, y_m), font, 0.42, (200, 200, 200), 1)
    y_m += 22
    cv2.putText(canvas, f"Keys/sec: {kb_metrics.get('kps', 0):.1f}", (15, y_m), font, 0.42, (200, 200, 200), 1)
    y_m += 22
    cv2.putText(canvas, f"Cog Load: {kb_metrics.get('cog_load', 0):.2f}", (15, y_m), font, 0.42, (200, 200, 200), 1)
    y_m += 22
    cv2.putText(canvas, f"Non-Game Keys: {non_game_ratio:.0%}", (15, y_m), font, 0.42,
                (0, 0, 255) if non_game_ratio > 0.4 else (200, 200, 200), 1)
    y_m += 22
    cv2.putText(canvas, f"Idle: {idle_seconds:.1f}s", (15, y_m), font, 0.42,
                (0, 0, 255) if idle_seconds > 10.0 else (200, 200, 200), 1)

    # Performance & Timer
    y_m += 28
    latency = vision_metrics.get('processing_latency_ms', 0)
    cv2.putText(canvas, f"FPS: {fps:.0f} | Vision: {latency:.0f}ms", (15, y_m), font, 0.42, (0, 255, 255), 1)
    y_m += 24
    timer_color = (0, 0, 255) if countdown <= 3.0 else (0, 255, 255)
    cv2.putText(canvas, f"NEXT DDA IN: {max(0, countdown):.1f}s", (15, y_m), font, 0.55, timer_color, 2)

    # Music indicator
    y_m += 24
    music_color = (0, 255, 0) if music_playing else (100, 100, 100)
    music_text = "CALM MUSIC: ON" if music_playing else "CALM MUSIC: OFF"
    cv2.putText(canvas, music_text, (15, y_m), font, 0.42, music_color, 1)

    # Key mapping legend (bottom-left of webcam)
    legend_y = h - 15
    cv2.putText(canvas, "Game Keys: WASD Space Shift Ctrl | Others = Frustration signal",
                (10, legend_y), font_s, 0.9, (120, 120, 120), 1)

    # ---- HINT DISPLAY (below webcam, golden box) ----
    if hint_text:
        hint_box_y = h + 5
        tier_colors = {
            "nudge":    (0, 180, 255),   # Orange
            "guide":    (0, 255, 200),   # Teal
            "solution": (100, 255, 100), # Green
        }
        hc = tier_colors.get(hint_tier, (0, 200, 255))

        # Background bar
        cv2.rectangle(canvas, (0, hint_box_y), (w, hint_box_y + 50), (30, 30, 40), -1)
        cv2.rectangle(canvas, (0, hint_box_y), (w, hint_box_y + 50), hc, 1)

        tier_label = f"[{hint_tier.upper()}]" if hint_tier else "[HINT]"
        cv2.putText(canvas, f"{tier_label} {hint_text}", (10, hint_box_y + 20),
                    font, 0.45, hc, 1)

        # Thinking progress bar (0-30s mapped to bar width)
        prog = min(thinking_seconds / 30.0, 1.0)
        bar_w = int(prog * (w - 20))
        cv2.rectangle(canvas, (10, hint_box_y + 35), (10 + bar_w, hint_box_y + 42), hc, -1)
        cv2.rectangle(canvas, (10, hint_box_y + 35), (w - 10, hint_box_y + 42), (80, 80, 80), 1)
        cv2.putText(canvas, f"Thinking: {thinking_seconds:.0f}s", (w - 130, hint_box_y + 45),
                    font_s, 0.9, (180, 180, 180), 1)

    elif emotion == "thinking" and thinking_seconds > 0:
        # Show "THINKING..." indicator before first hint triggers
        hint_box_y = h + 5
        cv2.rectangle(canvas, (0, hint_box_y), (w, hint_box_y + 35), (30, 30, 40), -1)
        cv2.putText(canvas, f"THINKING... (hint in {max(0, 10 - thinking_seconds):.0f}s)",
                    (10, hint_box_y + 22), font, 0.5, (255, 200, 0), 1)

    elif emotion == "given_up":
        # Given-up prompt
        hint_box_y = h + 5
        cv2.rectangle(canvas, (0, hint_box_y), (w, hint_box_y + 40), (20, 20, 60), -1)
        cv2.rectangle(canvas, (0, hint_box_y), (w, hint_box_y + 40), (0, 0, 200), 2)
        cv2.putText(canvas, "PLAYER IDLE - Need a break? Difficulty lowered.",
                    (10, hint_box_y + 25), font, 0.45, (100, 100, 255), 1)

    # ---- RIGHT SIDE: Game Parameters Panel ----
    panel_x = w + 10
    bar_max_w = 120

    game_colors = {
        "left_4_dead":     (100, 180, 255),
        "alien_isolation":  (100, 255, 100),
        "resident_evil_4": (100, 100, 255),
        "silent_hill":     (200, 150, 255),
    }

    y = 25
    cv2.putText(canvas, "GAME DDA PARAMETERS", (panel_x, y), font, 0.6, (255, 255, 0), 2)
    y += 10

    if profiles:
        for p in profiles:
            y += 20
            game_col = game_colors.get(p.genre, (255, 255, 255))
            cv2.putText(canvas, p.display_name.upper(), (panel_x, y), font, 0.45, game_col, 1)
            y += 5
            cv2.line(canvas, (panel_x, y), (panel_x + 380, y), game_col, 1)
            y += 15

            for param_name, spec in p.parameters.items():
                val = current_params.get(param_name, spec.default)
                norm_val = (val - spec.min_val) / (spec.max_val - spec.min_val) if spec.max_val > spec.min_val else 0
                norm_val = max(0.0, min(1.0, norm_val))
                norm_default = (spec.default - spec.min_val) / (spec.max_val - spec.min_val) if spec.max_val > spec.min_val else 0

                if norm_val > norm_default + 0.05:
                    bar_color = (0, 200, 0)
                elif norm_val < norm_default - 0.05:
                    bar_color = (0, 0, 220)
                else:
                    bar_color = (200, 200, 200)

                display_name = param_name.replace('_', ' ').title()
                if len(display_name) > 18:
                    display_name = display_name[:18]
                cv2.putText(canvas, display_name, (panel_x, y), font, 0.35, (200, 200, 200), 1)

                bar_x = panel_x + 160
                cv2.rectangle(canvas, (bar_x, y - 8), (bar_x + bar_max_w, y + 2), (40, 40, 40), -1)
                fill_w = int(norm_val * bar_max_w)
                if fill_w > 0:
                    cv2.rectangle(canvas, (bar_x, y - 8), (bar_x + fill_w, y + 2), bar_color, -1)
                default_x = bar_x + int(norm_default * bar_max_w)
                cv2.line(canvas, (default_x, y - 10), (default_x, y + 4), (255, 255, 255), 1)
                cv2.putText(canvas, f"{val:.2f}", (bar_x + bar_max_w + 5, y), font, 0.35, (255, 255, 255), 1)
                y += 18

    # Footer
    cv2.putText(canvas, "Type on keyboard to simulate gameplay | Press 'q' to exit",
                (20, canvas.shape[0] - 10), font, 0.45, (120, 120, 120), 1)

    return canvas


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def run_embedded_demo():
    import pandas as pd

    print("\n============================================================")
    print("  Project ECHO - Embedded Interactive Demo v2")
    print("  3-State Emotion Model + Thinking/Given-Up Detection")
    print("============================================================")

    # 1. Init Vision
    analyzer = WebcamAnalyzer()
    analyzer.start()

    # 2. Init Keyboard capture
    kb_listener = KeystrokeListener(buffer_size=500)
    kb_listener.start()

    # 3. Init Music
    music_player = CalmMusicPlayer()

    # 4. Init DDA for ALL profiles
    registry = GameRegistry()
    profiles = registry.list_profiles()
    modulator = ParameterModulator(smoothing_factor=1.0)

    # 5. Init Hint Engine
    hint_engine = HintEngine()

    # Aggregate all parameter defaults
    current_params = {}
    for p in profiles:
        current_params.update(p.get_defaults())

    # Telemetry history
    history = []

    print(f"\n  [READY] Tracking ALL games. Window opens shortly.")
    print("  Game Keys: WASD, Space, Shift, Ctrl, Enter, Esc, Tab, Arrows")
    print("  Non-game keys -> Frustration signal")
    print("  Press 'q' in the OpenCV window to exit.\n")

    start_time = time.time()

    # FPS tracking
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    # DDA timing — 16-second cycles
    dda_update_interval = 16.0
    next_dda_time = time.time() + dda_update_interval

    # Emotion priority for period aggregation
    emotion_priority = {
        'given_up': 10, 'frustrated': 7, 'thinking': 5,
        'joy': 2, 'normal': 0,
    }
    period_max_emotion = 'normal'

    # Thinking state tracking
    thinking_start_time = None  # When thinking state first triggered
    thinking_seconds = 0.0

    # Current hint state
    current_hint_text = None
    current_hint_tier = None

    try:
        while True:
            # -- Read Vision --
            frame, vis_metrics = analyzer.analyze_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            elapsed = time.time() - start_time

            # FPS tracking
            frame_count += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                frame_count = 0

            # -- Read Keyboard --
            events = kb_listener.get_recent_events(50)
            now = time.perf_counter()
            recent_events = [e for e in events if e.event_type == 'press' and (now - e.timestamp) < 2.0]
            kps = len(recent_events) / 2.0

            # Game vs non-game key analysis
            non_game_ratio = kb_listener.get_non_game_ratio(30)
            idle_seconds = kb_listener.get_idle_seconds()

            # IKI analysis
            ikis = kb_listener.get_iki_series(20)
            iki_cv = 0
            if len(ikis) > 3:
                iki_mean = np.mean(ikis)
                iki_std = np.std(ikis)
                iki_cv = (iki_std / iki_mean) if iki_mean > 0 else 0

            mash_ratio = len([i for i in ikis if i < 80]) / len(ikis) if len(ikis) > 0 else 0

            # -- Cognitive Load Fusion --
            move_energy = vis_metrics.get("movement_energy", 0)
            cog_load = np.clip(
                0.3 + (move_energy / 20.0) + (iki_cv * 0.4) + (mash_ratio * 0.5),
                0.1, 0.95
            )

            kb_metrics = {
                'kps': kps,
                'cog_load': cog_load,
                'mash_ratio': mash_ratio,
            }

            # ===========================================================
            # 3-STATE + 2-MODIFIER EMOTION FUSION
            # ===========================================================
            face_detected = vis_metrics.get("face_detected", False)
            smile_detected = vis_metrics.get("smile", False)
            chin_contact = vis_metrics.get("chin_contact", False)

            emotion = 'normal'
            state_desc = 'NORMAL: Relaxed baseline'

            # --- Modifier: Given-Up (highest priority) ---
            # No input for 16s + face detected without smile
            if idle_seconds >= 16.0 and face_detected and not smile_detected:
                emotion = 'given_up'
                state_desc = f'GIVEN UP: No input for {idle_seconds:.0f}s - player disengaged'
                
                # Continue generating hints even in given_up state per user request
                if thinking_start_time is None:
                    thinking_start_time = time.time()
                thinking_seconds = time.time() - thinking_start_time
                
                if profiles:
                    hint_result = hint_engine.get_hint(profiles[0].genre, thinking_seconds)
                    if hint_result:
                        current_hint_text, current_hint_tier = hint_result

            # --- Modifier: Thinking ---
            # Face detected + chin contact (immediate) OR (face detected + idle 10s+ without smile)
            elif (face_detected and chin_contact) or \
                 (face_detected and not smile_detected and idle_seconds >= 10.0 and kps < 0.5):
                emotion = 'thinking'
                if thinking_start_time is None:
                    thinking_start_time = time.time()
                thinking_seconds = time.time() - thinking_start_time
                
                if chin_contact and idle_seconds < 10.0:
                    state_desc = 'THINKING: Chin contact detected'
                else:
                    state_desc = f'THINKING: Idle {idle_seconds:.0f}s, pondering...'

                # Get hint from engine (use first profile genre as primary)
                if profiles:
                    hint_result = hint_engine.get_hint(profiles[0].genre, thinking_seconds)
                    if hint_result:
                        current_hint_text, current_hint_tier = hint_result

            # --- Core: Joy ---
            elif smile_detected:
                emotion = 'joy'
                state_desc = 'JOY: Smiling detected!'
                cog_load = max(0.1, cog_load - 0.2)
                thinking_start_time = None
                thinking_seconds = 0.0
                current_hint_text = None

            # --- Core: Frustrated ---
            # Non-game keys > 40% AND typing speed > 3 kps
            elif non_game_ratio > 0.4 and kps > 3.0:
                emotion = 'frustrated'
                state_desc = f'FRUSTRATED: {non_game_ratio:.0%} non-game keys at {kps:.1f} kps'
                thinking_start_time = None
                thinking_seconds = 0.0
                current_hint_text = None

            # High movement energy + fast mashing
            elif move_energy > 3.0 and mash_ratio > 0.2 and kps > 4.0:
                emotion = 'frustrated'
                state_desc = 'FRUSTRATED: Agitated movement + key mashing'
                thinking_start_time = None
                thinking_seconds = 0.0
                current_hint_text = None

            # Very high movement, low input (physical frustration)
            elif move_energy > 4.0 and kps < 1.5:
                emotion = 'frustrated'
                state_desc = 'FRUSTRATED: Agitated body, low input'
                thinking_start_time = None
                thinking_seconds = 0.0
                current_hint_text = None

            # --- Core: Normal ---
            else:
                emotion = 'normal'
                state_desc = 'NORMAL: Balanced baseline'
                thinking_start_time = None
                thinking_seconds = 0.0
                # Clear hints when player resumes normal activity
                if kps > 1.0:
                    current_hint_text = None

            # Aggregate most extreme emotion in this 16s period
            if emotion_priority.get(emotion, 0) > emotion_priority.get(period_max_emotion, 0):
                period_max_emotion = emotion

            # -- Trigger Calm Music --
            if emotion in ['frustrated', 'given_up']:
                music_player.start()
            else:
                music_player.stop()

            # -- DDA Modulation (every 16s) --
            time_left = next_dda_time - time.time()
            if time_left <= 0:
                print(f"\n  [DDA TRIGGER] Emotion={period_max_emotion} | CogLoad={cog_load:.2f} | Idle={idle_seconds:.1f}s | NonGame={non_game_ratio:.0%}")
                for p in profiles:
                    before_vals = {k: current_params[k] for k in p.get_parameter_names()}
                    adj = modulator.compute_adjustments(
                        profile=p,
                        trust_mean=0.5,
                        cognitive_load=cog_load,
                        emotion_label=period_max_emotion,
                        emotion_confidence=0.8,
                        current_values=before_vals,
                        timestamp=elapsed,
                    )
                    current_params.update(adj.after)
                    changed = {
                        k: (before_vals[k], adj.after[k])
                        for k in adj.after
                        if abs(adj.after[k] - before_vals[k]) > 0.001
                    }
                    if changed:
                        safe_reason = adj.reason.encode('ascii', 'replace').decode('ascii')
                        print(f"    [{p.display_name}] {safe_reason}")
                        for k, (bv, av) in changed.items():
                            direction = "UP" if av > bv else "DN"
                            print(f"      {direction} {k}: {bv:.3f} -> {av:.3f}")

                # Reset timer and period state
                next_dda_time = time.time() + dda_update_interval
                period_max_emotion = 'normal'

            # -- Log Telemetry --
            row = {
                "time": elapsed,
                "trust_mean": 0.5,
                "cognitive_load": cog_load,
                "emotion_label": emotion,
                "period_emotion": period_max_emotion,
                "emotion_confidence": 0.8,
                "movement_energy": move_energy,
                "kps": kps,
                "mash_ratio": mash_ratio,
                "non_game_ratio": non_game_ratio,
                "idle_seconds": idle_seconds,
                "thinking_seconds": thinking_seconds,
                "chin_contact": vis_metrics.get("chin_contact", False),
                "vision_latency_ms": vis_metrics.get('processing_latency_ms', 0),
            }
            row.update(current_params)
            history.append(row)

            # -- Render --
            display_emotion = period_max_emotion
            display_desc = state_desc if period_max_emotion == emotion else f"{period_max_emotion.upper()}: (Locked for this cycle)"

            canvas = draw_hud(
                frame, vis_metrics, kb_metrics, current_params,
                music_player.playing, display_emotion, display_desc,
                current_fps, time_left, profiles,
                hint_text=current_hint_text,
                hint_tier=current_hint_tier,
                thinking_seconds=thinking_seconds,
                idle_seconds=idle_seconds,
                non_game_ratio=non_game_ratio,
            )
            cv2.imshow("Project ECHO - Embedded Demo v2", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        analyzer.stop()
        kb_listener.stop()
        music_player.stop()
        cv2.destroyAllWindows()

        print("\n[FINISHED] Embedded demo terminated.")

        df = pd.DataFrame(history)
        if len(df) > 0:
            output_dir = os.path.join(os.path.dirname(__file__), "results", "data")
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "embedded_session_all_games.csv")
            df.to_csv(csv_path, index=False)
            print(f"  -> Session telemetry saved to {csv_path}")


if __name__ == "__main__":
    run_embedded_demo()
