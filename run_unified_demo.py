"""
Project ECHO - Unified Live Demo (v2)
====================================
Full synchronous integration with updated emotion detection:

    Emotion States:
        FOCUSED  - User pressing game keys (WASD, mouse, spacebar)
        JOY      - User smiling continuously for 10 seconds
        ANGRY    - User presses non-game keys OR shows fist to camera
        THINKING - User rests chin on hand for >15s (hint at 20s, stays visible)
        GIVEN_UP - User inactive for >30s (red warning banner)

    State evaluation cycle: every 15 seconds

    Inputs:
        1. Webcam -> face, smile, chin-contact, fist, movement
        2. Keyboard -> IKI, KPS, game/non-game keys
        3. Mouse -> trajectory, Sigma-Lognormal neuromotor

    Pipeline (every 2s):
        Feature Extraction -> DSTER -> POMDP Belief

    DDA (every 16s):
        Parameter modulation for all 4 games simultaneously

    On quit ('q'):
        Evaluation metrics + 13 plots + JSON report

Usage:
    python run_unified_demo.py
    python run_unified_demo.py --pipeline-interval 3 --dda-interval 20
"""

import cv2
import time
import numpy as np
import pandas as pd
import torch
import argparse
import os

from vision.webcam_analyzer import WebcamAnalyzer
from input_capture.keystroke_listener import KeystrokeListener
from input_capture.mouse_listener import MouseListener
from dster.emotion_classifier import EmotionClassifier
from feature_extraction.clc_estimator import CLCEstimator
from feature_extraction.sigma_lognormal import SigmaLognormalModel
from estimator.src.pomdp_model import POMDPModel
from game_framework.game_registry import GameRegistry
from game_framework.parameter_modulator import ParameterModulator
from game_framework.hint_engine import HintEngine
from pipeline.pipeline_validator import PipelineValidator
from changelog.changelog_manager import ChangeLogger

# Emotion labels matching DSTER output
EMOTION_LABELS = ["frustrated", "focused", "fatigued", "neutral", "engaged"]

# Map unified demo states to DSTER label indices for evaluation
STATE_TO_DSTER_IDX = {
    "focused": 1,
    "joy": 4,       # joy -> engaged
    "angry": 0,     # angry -> frustrated
    "thinking": 1,  # thinking -> focused
    "given_up": 2,  # given_up -> fatigued
    "neutral": 3,
}


# ---------------------------------------------------------
# DSTER Feature Builder (live keystroke -> tensor)
# ---------------------------------------------------------
class LiveFeatureBuilder:
    """Builds DSTER input tensors from live keystroke events."""

    def __init__(self, seq_len: int = 50):
        self.seq_len = seq_len
        self._press_times = {}

    def build_tensor(self, events) -> torch.Tensor:
        features = []
        prev_release_time = None
        prev_press_time = None

        for event in events:
            if event.event_type == "press":
                self._press_times[event.key_id] = event.timestamp
            elif event.event_type == "release":
                press_time = self._press_times.pop(event.key_id, None)
                if press_time is None:
                    continue

                press_duration = min(event.timestamp - press_time, 2.0)
                iki = 0.0
                if prev_press_time is not None:
                    iki = min(press_time - prev_press_time, 5.0)
                key_velocity = min(1.0 / max(press_duration, 0.01), 100.0) / 100.0
                flight_time = 0.0
                if prev_release_time is not None:
                    flight_time = min(max(press_time - prev_release_time, 0), 5.0)

                features.append([press_duration, iki, key_velocity, flight_time])
                prev_release_time = event.timestamp
                prev_press_time = press_time

        tensor = np.zeros((self.seq_len, 4), dtype=np.float32)
        if features:
            arr = np.array(features[-self.seq_len:], dtype=np.float32)
            arr[:, 0] = np.clip(arr[:, 0] / 0.5, 0, 1)
            arr[:, 1] = np.clip(arr[:, 1] / 1.0, 0, 1)
            arr[:, 3] = np.clip(arr[:, 3] / 1.0, 0, 1)
            n = len(arr)
            tensor[-n:] = arr

        return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)


# ---------------------------------------------------------
# EMOTION STATE MACHINE
# ---------------------------------------------------------
class EmotionStateMachine:
    """
    5-state emotion state machine with 15-second evaluation cycles.

    States:
        focused  - game keys active
        joy      - 10s continuous smile
        angry    - non-game keys or fist detected
        thinking - 15s chin contact (hint at 20s, stays)
        given_up - 30s total inactivity (red warning)
    """

    # Evaluation interval (seconds)
    EVAL_INTERVAL = 15.0

    # State thresholds
    JOY_SMILE_DURATION = 10.0       # seconds of continuous smile
    THINKING_CHIN_DURATION = 15.0   # seconds of chin contact
    HINT_DISPLAY_DURATION = 20.0    # seconds before hint appears
    GIVEN_UP_IDLE_DURATION = 30.0   # seconds of total inactivity

    def __init__(self):
        self.current_state = "neutral"
        self.last_eval_time = time.time()

        # Tracking durations
        self.smile_start_time = None
        self.smile_duration = 0.0

        self.chin_start_time = None
        self.chin_duration = 0.0

        self.idle_start_time = time.time()  # Starts as idle
        self.idle_duration = 0.0

        # Hint state
        self.hint_text = None
        self.hint_tier = None
        self.hint_locked = False  # Once shown, stays visible

        # State descriptions for HUD
        self.state_description = "Initializing..."

    def evaluate(
        self,
        game_keys_active: bool,
        smile_detected: bool,
        non_game_keys_pressed: bool,
        fist_detected: bool,
        chin_contact: bool,
        any_input: bool,
        hint_engine=None,
        game_genre: str = "left_4_dead",
    ) -> str:
        """
        Evaluate and potentially transition to a new emotion state.

        This method should be called every frame. The state only
        transitions at 15-second evaluation boundaries.

        Returns:
            The current emotion state string.
        """
        now = time.time()

        # --- Track continuous durations ---

        # Smile tracking
        if smile_detected:
            if self.smile_start_time is None:
                self.smile_start_time = now
            self.smile_duration = now - self.smile_start_time
        else:
            self.smile_start_time = None
            self.smile_duration = 0.0

        # Chin contact tracking
        if chin_contact:
            if self.chin_start_time is None:
                self.chin_start_time = now
            self.chin_duration = now - self.chin_start_time
        else:
            self.chin_start_time = None
            self.chin_duration = 0.0

        # Idle tracking (reset on any input)
        if any_input:
            self.idle_start_time = now
            self.idle_duration = 0.0
        else:
            self.idle_duration = now - self.idle_start_time

        # --- Hint logic (runs continuously, not gated by eval interval) ---
        if self.chin_duration >= self.HINT_DISPLAY_DURATION and hint_engine:
            hint_result = hint_engine.get_hint(game_genre, self.chin_duration)
            if hint_result:
                self.hint_text, self.hint_tier = hint_result
                self.hint_locked = True

        # --- State evaluation (only at 15-second boundaries) ---
        time_since_eval = now - self.last_eval_time

        if time_since_eval >= self.EVAL_INTERVAL:
            self.last_eval_time = now
            self._transition_state(
                game_keys_active, smile_detected,
                non_game_keys_pressed, fist_detected,
                chin_contact, any_input,
            )

        return self.current_state

    def _transition_state(
        self,
        game_keys_active, smile_detected,
        non_game_keys_pressed, fist_detected,
        chin_contact, any_input,
    ):
        """Determine the new state based on priority rules."""

        old_state = self.current_state

        # Priority 1: Given_Up (30s idle)
        if self.idle_duration >= self.GIVEN_UP_IDLE_DURATION:
            self.current_state = "given_up"
            self.state_description = (
                f"GIVEN UP: No input for {self.idle_duration:.0f}s - player disengaged"
            )

        # Priority 2: Angry (non-game keys or fist)
        elif non_game_keys_pressed or fist_detected:
            self.current_state = "angry"
            if fist_detected:
                self.state_description = "ANGRY: Fist detected on camera!"
            else:
                self.state_description = "ANGRY: Non-game keys pressed rapidly"

        # Priority 3: Thinking (chin contact > 15s)
        elif self.chin_duration >= self.THINKING_CHIN_DURATION:
            self.current_state = "thinking"
            self.state_description = (
                f"THINKING: Chin contact for {self.chin_duration:.0f}s"
            )

        # Priority 4: Joy (continuous smile > 10s)
        elif self.smile_duration >= self.JOY_SMILE_DURATION:
            self.current_state = "joy"
            self.state_description = (
                f"JOY: Smiling for {self.smile_duration:.0f}s!"
            )

        # Priority 5: Focused (game keys active)
        elif game_keys_active:
            self.current_state = "focused"
            self.state_description = "FOCUSED: Game keys active (WASD/Space/Mouse)"

        # Default: keep current or go neutral
        else:
            self.current_state = "neutral"
            self.state_description = "NEUTRAL: Waiting for input..."

        # Clear hints if state changed away from thinking
        if old_state == "thinking" and self.current_state != "thinking":
            if not self.hint_locked:
                self.hint_text = None
                self.hint_tier = None

    def get_hint(self):
        """Return current hint (text, tier) or (None, None)."""
        return self.hint_text, self.hint_tier

    def clear_hint(self):
        """Clear hint state (called when state changes away from thinking)."""
        self.hint_text = None
        self.hint_tier = None
        self.hint_locked = False


# ---------------------------------------------------------
# HUD DRAWING
# ---------------------------------------------------------
def draw_unified_hud(
    frame, vision_metrics, kb_metrics, mouse_metrics,
    pipeline_status, dster_emotion, dster_confidence,
    belief_state, current_params, profiles, dda_countdown,
    fps, state_machine, reference_frame=None,
):
    """Draw the full unified HUD overlay."""
    h, w, _ = frame.shape

    canvas_w = max(w + 450, 1200)
    canvas_h = max(h + 100, 750)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = frame

    font = cv2.FONT_HERSHEY_SIMPLEX

    # ---- LEFT OVERLAY ----
    ov_h = min(420, h)
    ov_w = min(400, w)
    roi = canvas[0:ov_h, 0:ov_w]
    black = np.zeros_like(roi)
    canvas[0:ov_h, 0:ov_w] = cv2.addWeighted(roi, 0.45, black, 0.55, 0)

    # Title
    cv2.putText(canvas, "PROJECT ECHO - UNIFIED DEMO v2", (15, 28), font, 0.6, (255, 255, 255), 2)

    # Current emotion state (primary - from state machine)
    state_colors = {
        "focused": (255, 200, 0),
        "joy": (255, 105, 180),
        "angry": (0, 0, 255),
        "thinking": (0, 200, 255),
        "given_up": (100, 100, 255),
        "neutral": (0, 255, 0),
    }
    state = state_machine.current_state
    emo_color = state_colors.get(state, (0, 255, 0))
    cv2.putText(canvas, f"STATE: {state.upper()}", (15, 60), font, 0.7, emo_color, 2)
    cv2.putText(canvas, state_machine.state_description, (15, 82), font, 0.35, (200, 200, 200), 1)

    # DSTER model prediction (secondary)
    y = 105
    cv2.putText(canvas, f"DSTER: {dster_emotion} ({dster_confidence:.1%})",
                (15, y), font, 0.42, (180, 180, 180), 1)

    # Pipeline status
    y += 25
    for stage_name, stage_ok, stage_ms in pipeline_status:
        status_text = "[OK]" if stage_ok else "[FAIL]"
        status_color = (0, 255, 0) if stage_ok else (0, 0, 255)
        cv2.putText(canvas, f"{status_text} {stage_name}: {stage_ms:.1f}ms",
                    (15, y), font, 0.38, status_color, 1)
        y += 18

    # Belief state
    y += 5
    cv2.putText(canvas, f"Trust: {belief_state.get('trust_mean', 0.5):.3f}",
                (15, y), font, 0.42, (200, 200, 200), 1)
    y += 20
    cv2.putText(canvas, f"CogLoad: {belief_state.get('cognitive_load', 0.5):.3f}",
                (15, y), font, 0.42, (200, 200, 200), 1)

    # Duration trackers
    y += 25
    cv2.putText(canvas, f"Smile: {state_machine.smile_duration:.1f}s / 10s",
                (15, y), font, 0.42,
                (255, 105, 180) if state_machine.smile_duration > 5 else (200, 200, 200), 1)
    y += 20
    cv2.putText(canvas, f"Chin: {state_machine.chin_duration:.1f}s / 15s",
                (15, y), font, 0.42,
                (0, 200, 255) if state_machine.chin_duration > 10 else (200, 200, 200), 1)
    y += 20
    cv2.putText(canvas, f"Idle: {state_machine.idle_duration:.1f}s / 30s",
                (15, y), font, 0.42,
                (0, 0, 255) if state_machine.idle_duration > 20 else (200, 200, 200), 1)

    # Input metrics
    y += 25
    cv2.putText(canvas, f"Keys/sec: {kb_metrics.get('kps', 0):.1f} | "
                f"NonGame: {kb_metrics.get('non_game_ratio', 0):.0%}",
                (15, y), font, 0.38, (200, 200, 200), 1)
    y += 18
    cv2.putText(canvas, f"Mouse pts: {mouse_metrics.get('point_count', 0)} | "
                f"SNR: {mouse_metrics.get('snr_db', 0):.1f}dB",
                (15, y), font, 0.38, (200, 200, 200), 1)
    y += 18
    cv2.putText(canvas, f"Movement: {vision_metrics.get('movement_energy', 0):.1f} | "
                f"Fist: {'YES' if vision_metrics.get('fist_detected', False) else 'no'}",
                (15, y), font, 0.38,
                (0, 0, 255) if vision_metrics.get('fist_detected', False) else (200, 200, 200), 1)

    # FPS and timers
    y += 25
    cv2.putText(canvas, f"FPS: {fps:.0f}", (15, y), font, 0.42, (0, 255, 255), 1)
    y += 20
    next_eval = max(0, state_machine.EVAL_INTERVAL - (time.time() - state_machine.last_eval_time))
    cv2.putText(canvas, f"Next eval: {next_eval:.0f}s | DDA: {max(0, dda_countdown):.0f}s",
                (15, y), font, 0.42, (0, 255, 255), 1)

    # ---- HINT / GIVEN_UP DISPLAY (below webcam) ----
    hint_text, hint_tier = state_machine.get_hint()

    if state == "given_up":
        # RED WARNING BANNER
        banner_y = h + 5
        cv2.rectangle(canvas, (0, banner_y), (w, banner_y + 60), (0, 0, 180), -1)
        cv2.rectangle(canvas, (0, banner_y), (w, banner_y + 60), (0, 0, 255), 3)
        cv2.putText(canvas, "THE USER HAS GIVEN UP", (30, banner_y + 25),
                    font, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "START AGAIN", (30, banner_y + 50),
                    font, 0.6, (200, 200, 255), 2)

    elif hint_text:
        # HINT DISPLAY (golden box, persistent)
        hint_y = h + 5
        tier_colors = {"nudge": (0, 180, 255), "guide": (0, 255, 200), "solution": (100, 255, 100)}
        hc = tier_colors.get(hint_tier, (0, 200, 255))
        cv2.rectangle(canvas, (0, hint_y), (w, hint_y + 55), (30, 30, 40), -1)
        cv2.rectangle(canvas, (0, hint_y), (w, hint_y + 55), hc, 2)
        tier_label = f"[{hint_tier.upper()}]" if hint_tier else "[HINT]"
        cv2.putText(canvas, f"{tier_label} {hint_text}", (10, hint_y + 22),
                    font, 0.45, hc, 1)
        # Thinking progress bar
        prog = min(state_machine.chin_duration / 30.0, 1.0)
        bar_w = int(prog * (w - 20))
        cv2.rectangle(canvas, (10, hint_y + 38), (10 + bar_w, hint_y + 46), hc, -1)
        cv2.rectangle(canvas, (10, hint_y + 38), (w - 10, hint_y + 46), (80, 80, 80), 1)

    elif state == "thinking":
        # THINKING indicator before hint threshold
        think_y = h + 5
        cv2.rectangle(canvas, (0, think_y), (w, think_y + 35), (30, 30, 40), -1)
        secs_to_hint = max(0, state_machine.HINT_DISPLAY_DURATION - state_machine.chin_duration)
        cv2.putText(canvas, f"THINKING... (hint in {secs_to_hint:.0f}s)",
                    (10, think_y + 22), font, 0.5, (255, 200, 0), 1)

    # ---- Reference photo thumbnail (top-right of webcam) ----
    if reference_frame is not None:
        ref_h, ref_w = 80, 80
        ref_thumb = cv2.resize(reference_frame, (ref_w, ref_h))
        rx = w - ref_w - 10
        ry = 10
        canvas[ry:ry+ref_h, rx:rx+ref_w] = ref_thumb
        cv2.rectangle(canvas, (rx, ry), (rx+ref_w, ry+ref_h), (255, 255, 255), 1)
        cv2.putText(canvas, "REF", (rx + 25, ry + ref_h + 15),
                    font, 0.35, (255, 255, 255), 1)

    # ---- RIGHT PANEL: Game DDA Parameters ----
    panel_x = w + 15
    bar_max_w = 120

    game_colors = {
        "left_4_dead": (100, 180, 255),
        "alien_isolation": (100, 255, 100),
        "resident_evil_4": (100, 100, 255),
        "silent_hill": (200, 150, 255),
    }

    y = 25
    cv2.putText(canvas, "GAME DDA PARAMETERS", (panel_x, y), font, 0.55, (255, 255, 0), 2)
    y += 10

    for p in profiles:
        y += 18
        game_col = game_colors.get(p.genre, (255, 255, 255))
        cv2.putText(canvas, p.display_name.upper(), (panel_x, y), font, 0.4, game_col, 1)
        y += 5
        cv2.line(canvas, (panel_x, y), (panel_x + 380, y), game_col, 1)
        y += 12

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
            cv2.putText(canvas, display_name, (panel_x, y), font, 0.33, (200, 200, 200), 1)

            bar_x = panel_x + 155
            cv2.rectangle(canvas, (bar_x, y - 7), (bar_x + bar_max_w, y + 2), (40, 40, 40), -1)
            fill_w = int(norm_val * bar_max_w)
            if fill_w > 0:
                cv2.rectangle(canvas, (bar_x, y - 7), (bar_x + fill_w, y + 2), bar_color, -1)
            default_x = bar_x + int(norm_default * bar_max_w)
            cv2.line(canvas, (default_x, y - 9), (default_x, y + 4), (255, 255, 255), 1)
            cv2.putText(canvas, f"{val:.2f}", (bar_x + bar_max_w + 5, y), font, 0.33, (255, 255, 255), 1)
            y += 15

    # Footer
    cv2.putText(canvas, "Webcam + Keyboard + Mouse | All 4 Games | Press 'q' to exit & evaluate",
                (20, canvas.shape[0] - 10), font, 0.4, (120, 120, 120), 1)

    return canvas


# ---------------------------------------------------------
# REFERENCE PHOTO CAPTURE FLOW
# ---------------------------------------------------------
def capture_reference_flow(analyzer):
    """
    Interactive reference photo capture.

    Shows a countdown, captures the user's neutral face, and
    displays it for 3 seconds before starting the main demo.
    """
    print("\n  [REFERENCE PHOTO CAPTURE]")
    print("  Look directly at the camera with a neutral expression...")

    # Show countdown in the webcam window
    for countdown in range(3, 0, -1):
        for _ in range(15):  # ~0.5s per count at 30fps
            frame, _ = analyzer.analyze_frame()
            if frame is not None:
                display = frame.copy()
                h, w = display.shape[:2]
                # Dark overlay
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                display = cv2.addWeighted(overlay, 0.4, display, 0.6, 0)
                # Countdown text
                cv2.putText(display, f"Look at camera...", (w//2 - 150, h//2 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display, str(countdown), (w//2 - 30, h//2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
                cv2.imshow("Project ECHO - Unified Demo v2", display)
                cv2.waitKey(33)

    # Capture reference
    ref_frame = analyzer.update_reference_from_current()

    if ref_frame is not None:
        print("  -> Reference photo captured!")
        # Show the captured photo for 2 seconds
        display = ref_frame.copy()
        h, w = display.shape[:2]
        cv2.putText(display, "BASELINE CAPTURED!", (w//2 - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(display, (0, 0), (w, h), (0, 255, 0), 4)
        cv2.imshow("Project ECHO - Unified Demo v2", display)
        cv2.waitKey(2000)
    else:
        print("  -> WARNING: Could not detect face for reference photo")
        print("  -> Continuing without baseline comparison")

    return ref_frame


# ---------------------------------------------------------
# MAIN DEMO
# ---------------------------------------------------------
def run_unified_demo(pipeline_interval: float = 2.0, dda_interval: float = 16.0):
    """Run the complete unified ECHO demo with updated emotion states."""

    print("\n============================================================")
    print("  Project ECHO - Unified Live Demo v2")
    print("  Updated Emotion States + Reference Photo + Fist Detection")
    print("============================================================")

    # ---- 1. Initialize input sources ----
    print("\n[1/6] Initializing input sources...")
    analyzer = WebcamAnalyzer()
    analyzer.start()
    time.sleep(1.0)  # Let camera warm up
    print("  -> Webcam started (with MediaPipe Hands)")

    kb_listener = KeystrokeListener(buffer_size=500)
    kb_listener.start()
    print("  -> Keyboard listener started")

    mouse_listener = MouseListener(buffer_size=2000)
    mouse_listener.start()
    print("  -> Mouse listener started")

    # ---- 2. Capture reference photo ----
    print("\n[2/6] Capturing reference photo...")
    reference_frame = capture_reference_flow(analyzer)

    # ---- 3. Initialize ML models ----
    print("\n[3/6] Initializing ML models...")
    dster_model = EmotionClassifier(
        num_channels=4, seq_len=50, embed_dim=32,
        num_heads=4, num_layers=1, num_classes=5,
    )
    dster_model.eval()
    print(f"  -> DSTER model: {sum(p.numel() for p in dster_model.parameters()):,} params")

    pomdp = POMDPModel()
    print("  -> POMDP belief model (15 states)")

    clc_estimator = CLCEstimator(window_size=20)
    slm_model = SigmaLognormalModel(min_strokes=3)
    feature_builder = LiveFeatureBuilder(seq_len=50)
    validator = PipelineValidator()
    print("  -> Feature extractors ready")

    # ---- 4. Initialize game framework ----
    print("\n[4/6] Initializing game framework...")
    registry = GameRegistry()
    profiles = registry.list_profiles()
    modulator = ParameterModulator(smoothing_factor=1.0)
    hint_engine = HintEngine()

    current_params = {}
    for p in profiles:
        current_params.update(p.get_defaults())
        print(f"  -> {p.display_name} ({len(p.parameters)} params)")

    # ---- 5. Initialize state machine ----
    print("\n[5/6] Initializing emotion state machine...")
    state_machine = EmotionStateMachine()
    print("  -> 5 states: Focused, Joy, Angry, Thinking, Given_Up")
    print("  -> Evaluation cycle: every 15 seconds")

    # ---- 6. State variables ----
    history = []
    dster_predictions = []
    features = {}

    dster_emotion = "neutral"
    dster_confidence = 0.2
    belief_state = {"trust_mean": 0.5, "trust_variance": 0.05, "cognitive_load": 0.4}
    pipeline_status = []
    mouse_metrics = {"point_count": 0, "snr_db": 0.0}

    start_time = time.time()
    next_pipeline_time = time.time() + 2.0
    next_dda_time = time.time() + dda_interval

    frame_count = 0
    fps_start = time.time()
    current_fps = 0.0

    print("\n[6/6] Starting main loop...")
    print("  Game Keys: WASD, Space, Shift, Ctrl")
    print("  Show fist to camera -> Angry")
    print("  Smile for 10s -> Joy")
    print("  Chin on hand for 15s -> Thinking")
    print("  Press 'q' to exit & evaluate\n")

    try:
        while True:
            # ---- Read webcam ----
            frame, vis_metrics = analyzer.analyze_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            elapsed = time.time() - start_time

            # FPS
            frame_count += 1
            if time.time() - fps_start > 1.0:
                current_fps = frame_count / (time.time() - fps_start)
                fps_start = time.time()
                frame_count = 0

            # ---- Read keyboard ----
            events = kb_listener.get_recent_events(100)
            now_perf = time.perf_counter()
            recent = [e for e in events if e.event_type == 'press' and (now_perf - e.timestamp) < 2.0]
            kps = len(recent) / 2.0

            non_game_ratio = kb_listener.get_non_game_ratio(30)
            idle_seconds = kb_listener.get_idle_seconds()

            ikis = kb_listener.get_iki_series(20)
            iki_cv = 0.0
            if len(ikis) > 3:
                iki_mean_val = np.mean(ikis)
                iki_std_val = np.std(ikis)
                iki_cv = (iki_std_val / iki_mean_val) if iki_mean_val > 0 else 0

            mash_ratio = len([i for i in ikis if i < 80]) / len(ikis) if len(ikis) > 0 else 0

            # ---- Read mouse ----
            xs, ys, ts = mouse_listener.get_trajectory_arrays(200)
            mouse_metrics["point_count"] = len(xs)

            # ---- Determine input signals for state machine ----
            # Game keys active = recent game key presses detected
            recent_game = [e for e in recent if e.is_game_key]
            game_keys_active = len(recent_game) > 0 or len(xs) > 10  # keys or mouse moving

            # Non-game keys pressed = ratio > 30% with significant typing
            non_game_pressing = non_game_ratio > 0.3 and kps > 2.0

            # Any input at all
            any_input = kps > 0.3 or len(xs) > 5

            # Vision signals
            smile_detected = vis_metrics.get("smile", False)
            chin_contact = vis_metrics.get("chin_contact", False)
            fist_detected = vis_metrics.get("fist_detected", False)

            # ---- Evaluate emotion state machine ----
            primary_genre = profiles[0].genre if profiles else "left_4_dead"
            state_machine.evaluate(
                game_keys_active=game_keys_active,
                smile_detected=smile_detected,
                non_game_keys_pressed=non_game_pressing,
                fist_detected=fist_detected,
                chin_contact=chin_contact,
                any_input=any_input,
                hint_engine=hint_engine,
                game_genre=primary_genre,
            )

            # ---- Cognitive load fusion ----
            move_energy = vis_metrics.get("movement_energy", 0)
            cog_load = float(np.clip(
                0.3 + (move_energy / 20.0) + (iki_cv * 0.4) + (mash_ratio * 0.5),
                0.1, 0.95
            ))

            kb_metrics = {
                "kps": kps,
                "cog_load": cog_load,
                "mash_ratio": mash_ratio,
                "non_game_ratio": non_game_ratio,
            }

            # ============================================================
            # SYNCHRONOUS PIPELINE TICK
            # ============================================================
            if time.time() >= next_pipeline_time:
                next_pipeline_time = time.time() + pipeline_interval
                pipeline_status = []

                # -- Stage 1: Feature Extraction --
                t0 = time.perf_counter()
                features = {}

                iki_array = np.array(ikis) if len(ikis) > 0 else np.array([150.0])
                clc_val, _ = clc_estimator.compute_clc(iki_array)
                features["clc_raw"] = max(0.0, min(1.0, clc_val)) if clc_val != 0 else 0.45
                features["iki_mean_ms"] = float(np.mean(iki_array))

                snr_db = 0.0
                if len(xs) >= 20:
                    try:
                        slm_result = slm_model.analyze_trajectory(
                            np.array(xs, dtype=np.float64),
                            np.array(ys, dtype=np.float64),
                            np.array(ts, dtype=np.float64),
                        )
                        snr_db = slm_result.get("snr_db", 0.0)
                    except Exception:
                        snr_db = 0.0
                features["snr_db"] = max(-10.0, min(60.0, snr_db))
                mouse_metrics["snr_db"] = snr_db

                feat_ms = (time.perf_counter() - t0) * 1000
                feat_valid = validator.validate_features(features)
                pipeline_status.append(("Feature Extract", feat_valid.is_valid, feat_ms))

                # -- Stage 2: DSTER Classification --
                t0 = time.perf_counter()
                all_events = kb_listener.get_recent_events(200)
                dster_input = feature_builder.build_tensor(all_events)

                with torch.no_grad():
                    logits, probs = dster_model(dster_input)
                    pred_idx = torch.argmax(probs, dim=-1).item()
                    pred_conf = probs[0, pred_idx].item()
                    dster_emotion = EMOTION_LABELS[pred_idx]
                    dster_confidence = pred_conf

                emotion_output = {"label": dster_emotion, "confidence": dster_confidence}
                dster_ms = (time.perf_counter() - t0) * 1000
                emo_valid = validator.validate_emotion(emotion_output)
                pipeline_status.append(("DSTER Classify", emo_valid.is_valid, dster_ms))

                # Store for post-session eval
                state_label_idx = STATE_TO_DSTER_IDX.get(state_machine.current_state, 3)
                dster_predictions.append({
                    "y_pred": pred_idx,
                    "y_scores": probs[0].cpu().numpy().copy(),
                    "y_true": state_label_idx,
                    "timestamp": elapsed,
                })

                # -- Stage 3: POMDP Belief --
                t0 = time.perf_counter()
                belief_state = pomdp.update_belief({
                    "clc_raw": features["clc_raw"],
                    "iki_mean_ms": features["iki_mean_ms"],
                    "snr_db": features["snr_db"],
                })
                pomdp_ms = (time.perf_counter() - t0) * 1000
                belief_valid = validator.validate_belief(belief_state)
                pipeline_status.append(("POMDP Belief", belief_valid.is_valid, pomdp_ms))

            # ============================================================
            # DDA MODULATION
            # ============================================================
            dda_countdown = next_dda_time - time.time()
            if dda_countdown <= 0:
                # Map state machine state to DDA emotion
                dda_emotion_map = {
                    "focused": "focused", "joy": "engaged", "angry": "frustrated",
                    "thinking": "focused", "given_up": "frustrated", "neutral": "neutral",
                }
                dda_emotion = dda_emotion_map.get(state_machine.current_state, "neutral")

                print(f"\n  [DDA] State={state_machine.current_state} | "
                      f"Trust={belief_state.get('trust_mean', 0.5):.3f} | "
                      f"CogLoad={belief_state.get('cognitive_load', 0.5):.3f}")

                for p in profiles:
                    before_vals = {k: current_params[k] for k in p.get_parameter_names()}
                    adj = modulator.compute_adjustments(
                        profile=p,
                        trust_mean=belief_state.get("trust_mean", 0.5),
                        cognitive_load=belief_state.get("cognitive_load", 0.5),
                        emotion_label=dda_emotion,
                        emotion_confidence=dster_confidence,
                        current_values=before_vals,
                        timestamp=elapsed,
                    )
                    current_params.update(adj.after)
                    changed = {k: (before_vals[k], adj.after[k])
                               for k in adj.after if abs(adj.after[k] - before_vals[k]) > 0.001}
                    if changed:
                        safe_reason = adj.reason.encode('ascii', 'replace').decode('ascii')
                        print(f"    [{p.display_name}] {safe_reason}")
                        for k, (bv, av) in changed.items():
                            print(f"      {'UP' if av > bv else 'DN'} {k}: {bv:.3f} -> {av:.3f}")

                next_dda_time = time.time() + dda_interval

            # ---- Log telemetry ----
            row = {
                "time": elapsed,
                "state": state_machine.current_state,
                "state_desc": state_machine.state_description,
                "dster_emotion": dster_emotion,
                "dster_confidence": dster_confidence,
                "trust_mean": belief_state.get("trust_mean", 0.5),
                "cognitive_load": belief_state.get("cognitive_load", 0.5),
                "smile_duration": state_machine.smile_duration,
                "chin_duration": state_machine.chin_duration,
                "idle_duration": state_machine.idle_duration,
                "fist_detected": vis_metrics.get("fist_detected", False),
                "kps": kps,
                "non_game_ratio": non_game_ratio,
                "mouse_points": mouse_metrics["point_count"],
                "movement_energy": move_energy,
            }
            row.update(current_params)
            history.append(row)

            # ---- Render HUD ----
            canvas = draw_unified_hud(
                frame, vis_metrics, kb_metrics, mouse_metrics,
                pipeline_status, dster_emotion, dster_confidence,
                belief_state, current_params, profiles, dda_countdown,
                current_fps, state_machine, reference_frame,
            )
            cv2.imshow("Project ECHO - Unified Demo v2", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        analyzer.stop()
        kb_listener.stop()
        mouse_listener.stop()
        cv2.destroyAllWindows()

        print("\n============================================================")
        print("  Session Ended - Generating Reports")
        print("============================================================")

        # Save session data
        df = pd.DataFrame(history)
        if len(df) > 0:
            output_dir = os.path.join(os.path.dirname(__file__), "results", "data")
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, "unified_session_v2.csv")
            df.to_csv(csv_path, index=False)
            print(f"  -> Session data saved: {csv_path}")

        # Post-session evaluation
        if len(dster_predictions) >= 5:
            print("\n  Running post-session evaluation...")
            _run_post_session_evaluation(dster_predictions, dster_model)
        else:
            print(f"\n  Not enough data ({len(dster_predictions)} ticks) - run longer next time")

        # Update changelog
        logger = ChangeLogger()
        logger.log("pipeline", "Ran unified live demo v2 session",
                   files=["run_unified_demo.py"])
        logger.save()

        print("\n============================================================")
        print("  Unified Demo v2 Complete!")
        print("============================================================\n")


def _run_post_session_evaluation(predictions, model):
    """Generate evaluation metrics from session data."""
    from evaluation.evaluation_metrics import DSTEREvaluator
    from evaluation.evaluation_plots import generate_all_evaluation_plots

    y_true = np.array([p["y_true"] for p in predictions])
    y_pred = np.array([p["y_pred"] for p in predictions])
    y_scores = np.array([p["y_scores"] for p in predictions])

    print(f"  -> Evaluating {len(y_true)} pipeline ticks")

    evaluator = DSTEREvaluator(model=model)
    results = evaluator.evaluate(y_true=y_true, y_pred=y_pred, y_scores=y_scores)

    print(f"  Accuracy:    {results.accuracy:.4f}")
    print(f"  Macro F1:    {results.macro_f1:.4f}")
    print(f"  Cohen Kappa: {results.cohen_kappa:.4f}")

    saved = generate_all_evaluation_plots(results=results, prefix="unified_v2")
    print(f"  -> {len(saved)} plots saved to results/plots/evaluation/")

    output_dir = os.path.join(os.path.dirname(__file__), "results", "evaluation")
    report_path = evaluator.save_report(results, output_dir, filename="unified_v2_eval.json")
    print(f"  -> Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Project ECHO - Unified Live Demo v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Emotion States:
  FOCUSED  - Game keys active (WASD, Space, Mouse)
  JOY      - Continuous smile for 10 seconds
  ANGRY    - Non-game keys OR fist shown to camera
  THINKING - Chin on hand for 15s (hint at 20s, stays visible)
  GIVEN_UP - No input for 30s (red warning)

States are re-evaluated every 15 seconds.
Press 'q' to exit and generate evaluation metrics + plots.
        """,
    )
    parser.add_argument("--pipeline-interval", type=float, default=2.0,
                        help="Seconds between pipeline ticks (default: 2.0)")
    parser.add_argument("--dda-interval", type=float, default=16.0,
                        help="Seconds between DDA modulations (default: 16.0)")

    args = parser.parse_args()
    run_unified_demo(
        pipeline_interval=args.pipeline_interval,
        dda_interval=args.dda_interval,
    )


if __name__ == "__main__":
    main()
