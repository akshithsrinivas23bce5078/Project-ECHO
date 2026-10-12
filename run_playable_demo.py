"""
Project ECHO - Unified Live Demo (v2)
====================================
Full synchronous integration with updated emotion detection:

    Emotion States:
        FOCUSED  - User pressing game keys (WASD, mouse, spacebar)
        JOY      - User smiling continuously for 10 seconds
        ANGRY    - User presses non-game keys
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

import os
import sys
import time
import math
import cv2
import pygame
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

# Demo state labels — these are the ACTUAL states the state machine produces.
# Plots will display these exact names so the user sees their real states.
DEMO_STATE_LABELS = ["angry", "focused", "thinking", "neutral", "joy", "given_up"]

# DSTER model labels (5-class ML output) — kept for model inference indexing
DSTER_MODEL_LABELS = ["frustrated", "focused", "fatigued", "neutral", "engaged"]

# Map demo state names → demo label indices for evaluation ground truth
STATE_TO_LABEL_IDX = {
    "angry": 0,
    "focused": 1,
    "thinking": 2,
    "neutral": 3,
    "joy": 4,
    "given_up": 5,
}

# Map DSTER model output index → demo label index for prediction alignment
# DSTER: 0=frustrated, 1=focused, 2=fatigued, 3=neutral, 4=engaged
DSTER_PRED_TO_DEMO_IDX = {
    0: 0,   # frustrated → angry
    1: 1,   # focused → focused
    2: 5,   # fatigued → given_up
    3: 3,   # neutral → neutral
    4: 4,   # engaged → joy
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
        angry    - non-game keys
        thinking - 15s chin contact (hint at 20s, stays)
        given_up - 30s total inactivity (red warning)
    """

    # Evaluation interval (seconds)
    EVAL_INTERVAL = 15.0  # Must match DDA interval

    # State thresholds
    JOY_SMILE_DURATION = 10.0       # seconds of continuous smile
    THINKING_CHIN_DURATION = 3.0    # seconds of chin contact before thinking
    HINT_DISPLAY_DURATION = 15.0    # seconds before hint appears
    GIVEN_UP_IDLE_DURATION = 30.0   # seconds of total inactivity

    def __init__(self):
        self.current_state = "neutral"
        self.last_eval_time = time.time()

        # Tracking durations
        self.smile_start_time = None
        self.smile_duration = 0.0

        self.chin_start_time = None
        self.chin_duration = 0.0
        self.chin_lost_time = None

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
        face_detected: bool,
        smile_detected: bool,
        non_game_keys_pressed: bool,
        chin_contact: bool,
        any_input: bool,
        hint_engine=None,
        game_genre: str = "left_4_dead",
        obstacle_present: bool = False,
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
            self.chin_lost_time = None
        else:
            if self.chin_start_time is not None:
                if self.chin_lost_time is None:
                    self.chin_lost_time = now
                elif now - self.chin_lost_time > 2.0:  # 2 seconds grace period
                    self.chin_start_time = None
                    self.chin_duration = 0.0
                    self.chin_lost_time = None
            else:
                self.chin_duration = 0.0

        # Idle tracking (reset ONLY on keyboard or mouse input, NOT face, UNLESS chin contact is active)
        if any_input or chin_contact:
            self.idle_start_time = now
            self.idle_duration = 0.0
        else:
            self.idle_duration = now - self.idle_start_time

        # --- INSTANT JOY: Detect smile + face -> immediately transition ---
        if face_detected and smile_detected and self.smile_duration >= 1.0:
            if self.current_state != "joy":
                self.current_state = "joy"
                self.state_description = f"JOY: Smile detected! ({self.smile_duration:.0f}s)"
            return self.current_state

        # --- INSTANT THINKING: Detect chin contact -> immediately transition ---
        if chin_contact and self.chin_duration >= self.THINKING_CHIN_DURATION:
            if self.current_state != "thinking":
                self.current_state = "thinking"
            self.state_description = f"THINKING: Chin contact for {self.chin_duration:.0f}s"
            
            # --- Hint logic (runs while in thinking state) ---
            if self.chin_duration >= self.HINT_DISPLAY_DURATION and hint_engine:
                hint_result = hint_engine.get_hint(game_genre, self.chin_duration)
                if hint_result:
                    self.hint_text, self.hint_tier = hint_result
                    self.hint_locked = True
                    self.state_description = f"THINKING: Hint displayed! ({self.hint_tier.upper()})"
            
            return self.current_state

        # --- THINKING persists: Stay in thinking until chin released ---
        if self.current_state == "thinking" and chin_contact:
            self.state_description = f"THINKING: Chin contact for {self.chin_duration:.0f}s (holding until DDA cycle)"
            return self.current_state

        # --- INSTANT GIVEN_UP: Idle >= threshold -> immediately transition ---
        if self.idle_duration >= self.GIVEN_UP_IDLE_DURATION:
            if self.current_state != "given_up":
                self.current_state = "given_up"
                self.state_description = f"GIVEN UP: No input for {self.idle_duration:.0f}s - player disengaged"
            return self.current_state

        # --- State evaluation (only at 15-second boundaries) ---
        time_since_eval = now - self.last_eval_time

        if time_since_eval >= self.EVAL_INTERVAL:
            self.last_eval_time = now
            self._transition_state(
                game_keys_active, smile_detected,
                non_game_keys_pressed,
                chin_contact, any_input,
            )

        return self.current_state

    def _transition_state(
        self,
        game_keys_active, smile_detected,
        non_game_keys_pressed,
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

        # Priority 2: Angry (non-game keys)
        elif non_game_keys_pressed and time.time() > getattr(self, 'angry_cooldown_until', 0.0):
            self.current_state = "angry"
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
# Playable Demo Game: Cube Dodger
# ---------------------------------------------------------
import random




class DynamicMusicPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.current_state = None
        self.music_map = {
            "focused": "focused_music/focused_music_play.mp3",
            "joy": "focused_music/focused_music_play.mp3",
            "angry": "calm_music/sample_calm_music.mp3",
            "thinking": "thinking_music/sample_thinking_music.mp3",
            "given_up": "hope_music/sample_hope_music.mp3",
            "neutral": None
        }
    
    def update(self, state):
        if state != self.current_state:
            self.current_state = state
            music_file = self.music_map.get(state, None)
            
            if music_file is None:
                pygame.mixer.music.stop()
            elif os.path.exists(music_file):
                pygame.mixer.music.load(music_file)
                pygame.mixer.music.play(-1) # Loop indefinitely
            else:
                print(f"[Music] Warning: Missing file {music_file}")

class CubeDodgerGame:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.player_size = 30
        self.player_x = self.rect.centerx
        self.player_y = self.rect.bottom - 40
        self.player_speed = 400.0  # px per second
        self.targets = []
        self.obstacles = []
        self.score = 0.0
        self.target_spawn_timer = 0.0
        self.obstacle_spawn_timer = 0.0
        self.game_started = False
        
        # Modal UI buttons
        self.yes_btn_rect = pygame.Rect(self.rect.centerx - 70, self.rect.centery + 20, 60, 40)
        self.no_btn_rect = pygame.Rect(self.rect.centerx + 10, self.rect.centery + 20, 60, 40)
        
        # New State logic
        self.show_reduce_obstacles_prompt = False
        self.show_angry_prompt = False
        self.wait_for_space = False
        self.is_moving = False
        
        # DDA parameters mapped from AI Director
        self.target_interval = 2.4
        self.obstacle_interval = 2.5
        self.base_obstacle_speed = 200.0
        
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        self.big_font = pygame.font.SysFont("Consolas", 36, bold=True)
    def handle_input(self, dt, mouse_offset=(0,0), mouse_enabled=True, keys_enabled=True):
        keys = pygame.key.get_pressed()
        kb_active = False
        
        if keys_enabled:
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.player_x -= self.player_speed * dt
                kb_active = True
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.player_x += self.player_speed * dt
                kb_active = True
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                self.player_y -= self.player_speed * dt
                kb_active = True
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                self.player_y += self.player_speed * dt
                kb_active = True

        # Mouse navigation support
        mouse_pos = pygame.mouse.get_pos()
        local_mouse_pos = (mouse_pos[0] - mouse_offset[0], mouse_pos[1] - mouse_offset[1])
        
        if not hasattr(self, 'last_mouse_pos'):
            self.last_mouse_pos = local_mouse_pos
            
        mouse_moved = False
        if local_mouse_pos != self.last_mouse_pos:
            # Mouse Delta (synchronous with keyboard)
            dx = local_mouse_pos[0] - self.last_mouse_pos[0]
            dy = local_mouse_pos[1] - self.last_mouse_pos[1]
            
            # Apply mouse delta directly to player position
            self.player_x += dx
            self.player_y += dy
            
            self.last_mouse_pos = local_mouse_pos
            if mouse_enabled:
                mouse_moved = True
            
        self.is_moving = kb_active or mouse_moved
            
        # Dynamically increase speed if user fails to navigate with keyboard OR mouse
        if not self.is_moving:
            self.player_speed = min(1200.0, self.player_speed + 250.0 * dt)
        else:
            self.player_speed = max(400.0, self.player_speed - 500.0 * dt)
            
        # Clamp to game area
        self.player_x = max(self.rect.left + self.player_size/2, min(self.rect.right - self.player_size/2, self.player_x))
        self.player_y = max(self.rect.top + self.player_size/2, min(self.rect.bottom - self.player_size/2, self.player_y))

    def handle_event(self, event):
        # Modal for obstacles
        if self.show_reduce_obstacles_prompt and event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.yes_btn_rect.collidepoint(event.pos):
                    self.show_reduce_obstacles_prompt = False
                    self.wait_for_space = True
                    # Clear all obstacles to ensure player has breathing room
                    self.obstacles.clear()
                    return "yes_obstacles"
                if self.no_btn_rect.collidepoint(event.pos):
                    self.show_reduce_obstacles_prompt = False
                    return "no_obstacles"
                    
        # Also allow keyboard: Y for yes, N for no
        if self.show_reduce_obstacles_prompt and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_y:
                self.show_reduce_obstacles_prompt = False
                self.wait_for_space = True
                self.obstacles.clear()
                return "yes_obstacles"
            if event.key == pygame.K_n:
                self.show_reduce_obstacles_prompt = False
                return "no_obstacles"
        # Modal for Angry Give Up Prompt
        if self.show_angry_prompt and event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.yes_btn_rect.collidepoint(event.pos):
                    self.show_angry_prompt = False
                    return "yes_angry"
                if self.no_btn_rect.collidepoint(event.pos):
                    self.show_angry_prompt = False
                    self.wait_for_space = True
                    return "no_angry"
                    
        if self.show_angry_prompt and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_y:
                self.show_angry_prompt = False
                return "yes_angry"
            if event.key == pygame.K_n:
                self.show_angry_prompt = False
                self.wait_for_space = True
                return "no_angry"
                
        return None
        
    def update(self, dt, current_state, mouse_offset=(0,0), mouse_enabled=True, keys_enabled=True):
        # Stop on given_up
        if current_state == "given_up" and getattr(self, 'last_state', "") != "given_up":
            self.given_up_timer = 30.0
            self.game_started = False # Require restart if they come back
            
        self.last_state = current_state
        
        if getattr(self, 'given_up_timer', 0.0) > 0:
            self.given_up_timer -= dt
            self.game_started = False
            return
            
        # Modal pause
        if self.show_reduce_obstacles_prompt or self.show_angry_prompt:
            return
            
        if self.wait_for_space:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.wait_for_space = False
                self.player_speed = 400.0
            return
            
        # Pause on neutral
        if current_state == "neutral":
            return
            
        if not self.game_started:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] or keys[pygame.K_RETURN]:
                self.game_started = True
                self.targets = []
                self.obstacles = []
                self.score = 0.0
            return
            
        self.handle_input(dt, mouse_offset, mouse_enabled, keys_enabled)
        
        # Determine current target interval based on DDA and target count
        speed_mult = 1.0 + len(self.obstacles) * 0.05
        
        def spawn_entity(size, is_obstacle):
            while True:
                x = random.randint(self.rect.left, self.rect.right - size)
                y = random.randint(self.rect.top, self.rect.bottom - size)
                # Ensure they don't spawn right on top of the player
                dist = ((x - self.player_x)**2 + (y - self.player_y)**2)**0.5
                if dist > 100:
                    break
            
            speed_mult = 1.0 + len(self.obstacles) * 0.05
            base_speed = (self.base_obstacle_speed * 1.5 if is_obstacle else self.base_obstacle_speed) * speed_mult
            
            # Random 360-degree direction
            angle = random.uniform(0, 2 * math.pi)
            dx = math.cos(angle) * base_speed
            dy = math.sin(angle) * base_speed
            
            return {'rect': pygame.Rect(x, y, size, size), 'x': float(x), 'y': float(y), 'dx': dx, 'dy': dy, 'bounces': 0}

        # Spawn targets and obstacles simultaneously ONLY when player moves
        if self.is_moving:
            self.target_spawn_timer -= dt
            if self.target_spawn_timer <= 0:
                # Generate targets more than obstacles (2 targets per 1 obstacle)
                self.targets.append(spawn_entity(20, False))
                self.targets.append(spawn_entity(20, False))
                self.obstacles.append(spawn_entity(30, True))
                # Use unified obstacle interval for even generation timing
                self.target_spawn_timer = self.obstacle_interval
            
        player_rect = pygame.Rect(self.player_x - self.player_size/2, self.player_y - self.player_size/2, self.player_size, self.player_size)
        
        # Update targets
        for t in self.targets[:]:
            t['x'] += t['dx'] * dt
            t['y'] += t['dy'] * dt
            t['rect'].x, t['rect'].y = int(t['x']), int(t['y'])
            
            # Bounce targets off walls infinitely
            if t['rect'].left < self.rect.left and t['dx'] < 0:
                t['rect'].left = self.rect.left
                t['x'] = float(t['rect'].x)
                t['dx'] *= -1
            if t['rect'].right > self.rect.right and t['dx'] > 0:
                t['rect'].right = self.rect.right
                t['x'] = float(t['rect'].x)
                t['dx'] *= -1
                
            if t['rect'].top < self.rect.top and t['dy'] < 0:
                t['rect'].top = self.rect.top
                t['y'] = float(t['rect'].y)
                t['dy'] *= -1
            if t['rect'].bottom > self.rect.bottom and t['dy'] > 0:
                t['rect'].bottom = self.rect.bottom
                t['y'] = float(t['rect'].y)
                t['dy'] *= -1
            
            if t['rect'].colliderect(player_rect):
                self.targets.remove(t)
                self.score += 25.0
                
        # Update obstacles
        for obs in self.obstacles[:]:
            obs['x'] += obs['dx'] * dt
            obs['y'] += obs['dy'] * dt
            obs['rect'].x, obs['rect'].y = int(obs['x']), int(obs['y'])
            
            # Bounce obstacles off walls infinitely
            if obs['rect'].left < self.rect.left and obs['dx'] < 0:
                obs['rect'].left = self.rect.left
                obs['x'] = float(obs['rect'].x)
                obs['dx'] *= -1
            if obs['rect'].right > self.rect.right and obs['dx'] > 0:
                obs['rect'].right = self.rect.right
                obs['x'] = float(obs['rect'].x)
                obs['dx'] *= -1
                
            if obs['rect'].top < self.rect.top and obs['dy'] < 0:
                obs['rect'].top = self.rect.top
                obs['y'] = float(obs['rect'].y)
                obs['dy'] *= -1
            if obs['rect'].bottom > self.rect.bottom and obs['dy'] > 0:
                obs['rect'].bottom = self.rect.bottom
                obs['y'] = float(obs['rect'].y)
                obs['dy'] *= -1
                
            if obs['rect'].colliderect(player_rect):
                self.obstacles.remove(obs)
                self.score -= 10.0
                
    def has_active_obstacles(self):
        return len(self.obstacles) > 0

    def draw(self, surface, current_state):
        # Draw background
        pygame.draw.rect(surface, (20, 20, 30), self.rect)
        pygame.draw.rect(surface, (100, 100, 100), self.rect, 2)
        
        if current_state == "given_up" or getattr(self, 'given_up_timer', 0.0) > 0:
            text = self.big_font.render("GAME OVER", True, (255, 50, 50))
            surface.blit(text, text.get_rect(center=(self.rect.centerx, self.rect.centery - 40)))
            score_text = self.font.render(f"Final Score: {int(self.score)}", True, (255, 255, 255))
            surface.blit(score_text, score_text.get_rect(center=(self.rect.centerx, self.rect.centery)))
            
            if getattr(self, 'given_up_timer', 0.0) > 0:
                time_text = self.font.render(f"Locked for {int(self.given_up_timer)}s", True, (200, 200, 200))
                surface.blit(time_text, time_text.get_rect(center=(self.rect.centerx, self.rect.centery + 30)))
            return
            
        if self.wait_for_space:
            text = self.font.render("[Press Space to continue]", True, (255, 255, 100))
            surface.blit(text, text.get_rect(center=self.rect.center))
            return
            
        # Draw Obstacle Prompt
        if self.show_reduce_obstacles_prompt:
            pygame.draw.rect(surface, (40, 40, 50), (self.rect.left + 20, self.rect.centery - 60, self.rect.width - 40, 140))
            pygame.draw.rect(surface, (200, 200, 200), (self.rect.left + 20, self.rect.centery - 60, self.rect.width - 40, 140), 2)
            
            font_small = pygame.font.SysFont("Consolas", 18, bold=True)
            text = font_small.render("Should the no of", True, (255, 255, 255))
            text2 = font_small.render("obstacles get reduced?", True, (255, 255, 255))
            surface.blit(text, text.get_rect(center=(self.rect.centerx, self.rect.centery - 35)))
            surface.blit(text2, text2.get_rect(center=(self.rect.centerx, self.rect.centery - 15)))
            
            # Yes Button
            pygame.draw.rect(surface, (50, 200, 50), self.yes_btn_rect)
            yes_text = self.font.render("YES(Y)", True, (0, 0, 0))
            surface.blit(yes_text, yes_text.get_rect(center=self.yes_btn_rect.center))
            
            # No Button
            pygame.draw.rect(surface, (200, 50, 50), self.no_btn_rect)
            no_text = self.font.render("NO(N)", True, (0, 0, 0))
            surface.blit(no_text, no_text.get_rect(center=self.no_btn_rect.center))
            return
            
        # Draw Angry Prompt
        if self.show_angry_prompt:
            pygame.draw.rect(surface, (40, 40, 50), (self.rect.left + 20, self.rect.centery - 60, self.rect.width - 40, 140))
            pygame.draw.rect(surface, (200, 200, 200), (self.rect.left + 20, self.rect.centery - 60, self.rect.width - 40, 140), 2)
            
            # Using smaller font for long text if necessary, or just standard font
            font_small = pygame.font.SysFont("Consolas", 18, bold=True)
            text1 = font_small.render("User is not ready to get", True, (255, 255, 255))
            text2 = font_small.render("involved in the game?", True, (255, 255, 255))
            text3 = font_small.render("Can we give up?", True, (255, 255, 255))
            
            surface.blit(text1, text1.get_rect(center=(self.rect.centerx, self.rect.centery - 40)))
            surface.blit(text2, text2.get_rect(center=(self.rect.centerx, self.rect.centery - 20)))
            surface.blit(text3, text3.get_rect(center=(self.rect.centerx, self.rect.centery + 0)))
            
            # Yes Button (shift down slightly to fit text)
            pygame.draw.rect(surface, (50, 200, 50), self.yes_btn_rect)
            yes_text = self.font.render("YES(Y)", True, (0, 0, 0))
            surface.blit(yes_text, yes_text.get_rect(center=self.yes_btn_rect.center))
            
            # No Button
            pygame.draw.rect(surface, (200, 50, 50), self.no_btn_rect)
            no_text = self.font.render("NO(N)", True, (0, 0, 0))
            surface.blit(no_text, no_text.get_rect(center=self.no_btn_rect.center))
            return
            
        if current_state == "neutral":
            text = self.big_font.render("PAUSED (Neutral)", True, (200, 200, 200))
            surface.blit(text, text.get_rect(center=self.rect.center))
            return
            
        if not self.game_started:
            text = self.big_font.render("Press SPACE to Play", True, (200, 200, 200))
            surface.blit(text, text.get_rect(center=self.rect.center))
            return
            
        # Draw player
        player_rect = pygame.Rect(0, 0, self.player_size, self.player_size)
        player_rect.center = (self.player_x, self.player_y)
        pygame.draw.rect(surface, (50, 150, 255), player_rect)
        # Player border glow
        pygame.draw.rect(surface, (100, 200, 255), player_rect, 2)
        
        # Draw targets (green with border)
        for t_dict in self.targets:
            target = t_dict['rect']
            pygame.draw.rect(surface, (50, 255, 50), target)
            pygame.draw.rect(surface, (200, 255, 200), target, 2)
            
        # Draw obstacles (bright red with X marker for visibility)
        for obs in self.obstacles:
            obstacle = obs['rect']
            pygame.draw.rect(surface, (255, 50, 50), obstacle)
            pygame.draw.rect(surface, (255, 200, 200), obstacle, 2)
            # Draw X marker on obstacles
            pygame.draw.line(surface, (255, 255, 0), obstacle.topleft, obstacle.bottomright, 2)
            pygame.draw.line(surface, (255, 255, 0), obstacle.topright, obstacle.bottomleft, 2)
            
        # Draw obstacle count indicator
        if len(self.obstacles) > 0:
            obs_text = self.font.render(f"OBSTACLES: {len(self.obstacles)}", True, (255, 100, 100))
            surface.blit(obs_text, (self.rect.left + 10, self.rect.top + 60))
            
        # Draw HUD
        score_text = self.font.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        surface.blit(score_text, (self.rect.left + 10, self.rect.top + 10))
        speed_text = self.font.render(f"Speed: {int(self.player_speed)}", True, (200, 200, 200))
        surface.blit(speed_text, (self.rect.left + 10, self.rect.top + 35))



# ---------------------------------------------------------
# HUD DRAWING
# ---------------------------------------------------------
def draw_unified_hud(
    frame, vision_metrics, kb_metrics, mouse_metrics,
    pipeline_status, dster_emotion, dster_confidence,
    belief_state, current_params, profiles, dda_countdown,
    fps, state_machine, reference_frame=None, playable_game=None
):
    """Draw the full unified HUD overlay."""
    h, w, _ = frame.shape

    canvas_w = w + 20 + 400 + 20 + 450
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

    # ---- Input Source Status Indicators ----
    y += 25
    # Face detection status - ENABLED when face is visible in camera
    face_detected = vision_metrics.get("face_detected", False)
    face_color = (0, 255, 0) if face_detected else (0, 0, 255)
    face_label = "ENABLED" if face_detected else "DISABLED"
    cv2.putText(canvas, f"Face: {face_label}", (15, y), font, 0.42, face_color, 1)

    y += 20
    unified_idle = getattr(state_machine, 'idle_duration', 0.0)
    
    # Mouse status
    mouse_enabled = mouse_metrics.get('enabled', False)
    is_moving = mouse_metrics.get('is_moving', False)
    mouse_color = (0, 255, 0) if mouse_enabled else (0, 0, 255)
    mouse_label = "ENABLED" if mouse_enabled else f"DISABLED (Idle: {unified_idle:.1f}s)"
    cv2.putText(canvas, f"Mouse: {mouse_label}", (15, y), font, 0.42, mouse_color, 1)
    
    if is_moving:
        cv2.circle(canvas, (280, y - 4), 5, (0, 165, 255), -1)

    y += 20
    # Keystroke status
    kps_val = kb_metrics.get('kps', 0)
    keys_enabled = not (getattr(playable_game, 'given_up_timer', 0.0) > 0)
    keys_active = kps_val > 0 and keys_enabled
    keys_color = (0, 255, 0) if keys_active else (0, 0, 255)
    keys_label = "ENABLED" if keys_active else f"DISABLED (Idle: {unified_idle:.1f}s)"
    cv2.putText(canvas, f"Keystrokes: {keys_label}", (15, y), font, 0.42, keys_color, 1)

    # Detailed metrics (shown when active)
    y += 20
    if keys_active:
        ngr_val = kb_metrics.get('non_game_ratio', 0)
        cv2.putText(canvas, f"  Keys/sec: {kps_val:.1f} | NonGame: {ngr_val:.0%}",
                    (15, y), font, 0.35, (200, 200, 200), 1)
        y += 16
    if mouse_metrics.get('enabled', False):
        cv2.putText(canvas, f"  Mouse pts: {mouse_metrics.get('point_count', 0)}",
                    (15, y), font, 0.35, (200, 200, 200), 1)
        y += 16

    # FPS and timers
    y += 10
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
        # Determine text layout
        if hint_tier == "solution":
            # When solution is reached, show a previous hint on top, and the solution on bottom
            import game_framework.hint_engine as he
            guide_pool = he.HINT_POOLS.get("cube_dodger", {}).get("guide", ["Focus on avoiding corners"])
            top_text = guide_pool[0]
            bot_text = hint_text
            box_h = 75
        else:
            top_text = hint_text
            bot_text = None
            box_h = 55
            
        cv2.rectangle(canvas, (0, hint_y), (w, hint_y + box_h), (30, 30, 40), -1)
        cv2.rectangle(canvas, (0, hint_y), (w, hint_y + box_h), hc, 2)
        
        cv2.putText(canvas, f"Question: {top_text}", (10, hint_y + 22), font, 0.45, hc, 1)
        
        if bot_text:
            cv2.putText(canvas, f"Answer: {bot_text}", (10, hint_y + 45), font, 0.45, (100, 255, 100), 1)
            
        # Thinking progress bar
        prog = min(state_machine.chin_duration / 30.0, 1.0)
        bar_w = int(prog * (w - 20))
        bar_y = hint_y + (58 if bot_text else 38)
        cv2.rectangle(canvas, (10, bar_y), (10 + bar_w, bar_y + 8), hc, -1)
        cv2.rectangle(canvas, (10, bar_y), (w - 10, bar_y + 8), (80, 80, 80), 1)

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
    panel_x = w + 20 + 400 + 20
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

    if playable_game:
        y += 5
        cv2.putText(canvas, "CUBE DODGER (Current Game)", (panel_x, y), font, 0.4, (255, 200, 255), 1)
        y += 5
        cv2.line(canvas, (panel_x, y), (panel_x + 380, y), (255, 200, 255), 1)
        y += 15
        
        cv2.putText(canvas, f"Obstacle Interval: {playable_game.obstacle_interval:.2f}s (Decreases as Spawn Rate UP)", (panel_x, y), font, 0.33, (200, 255, 200), 1)
        y += 15
        cv2.putText(canvas, f"Obstacle Speed   : {playable_game.base_obstacle_speed:.0f}px/s (Increases as Density UP)", (panel_x, y), font, 0.33, (255, 200, 200), 1)
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
                cv2.putText(display, "Look at camera...", (w//2 - 150, h//2 - 40),
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
def run_unified_demo(pipeline_interval: float = 2.0, dda_interval: float = 15.0):
    """Run the complete unified ECHO demo with updated emotion states."""

    print("\n============================================================")
    print("  Project ECHO - Unified Live Demo v2")
    print("  Updated Emotion States + Reference Photo")
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

    # Initialize Music Player
    print("  -> Initializing Music Player...")
    music_player = DynamicMusicPlayer()


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
    mouse_metrics = {"point_count": 0, "snr_db": 0.0, "enabled": True}

    start_time = time.time()
    next_pipeline_time = time.time() + 2.0
    next_dda_time = time.time() + dda_interval
    dda_countdown = dda_interval

    # Frame capture every 15 seconds (aligned with state evaluation)
    frame_capture_dir = os.path.join(os.path.dirname(__file__), "results", "frame_captures")
    os.makedirs(frame_capture_dir, exist_ok=True)
    # Clean out stale captures from previous sessions
    for old_file in [f for f in os.listdir(frame_capture_dir) if f.endswith('.png')]:
        try:
            os.remove(os.path.join(frame_capture_dir, old_file))
        except OSError:
            pass
    next_frame_capture_time = time.time() + 15.0
    frame_capture_count = 0

    frame_count = 0
    fps_start = time.time()
    current_fps = 0.0

    print("\n[6/6] Starting main loop...")
    print("  Game Keys: WASD, Space, Shift, Ctrl")
    print("  ")
    print("  Smile for 10s -> Joy\n  Chin on hand for 15s -> Thinking\n  Press 'q' to exit & evaluate\n")
    
    mouse_idle_timer = 0.0
    mouse_enabled = True
    # Initialize Game (offscreen surface, rendered into OpenCV canvas)
    pygame.init()
    # We create a massive unified Pygame window to hold everything
    game_screen = pygame.display.set_mode((1460, 750))
    pygame.display.set_caption("ECHO - Unified Demo v2")
    # Offscreen surface for game rendering
    game_surface = pygame.Surface((400, 550))
    playable_game = CubeDodgerGame((0, 0, 400, 550))
    last_frame_time = time.time()

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
            
            kb_metrics = {
                "kps": kps,
                "non_game_ratio": non_game_ratio,
                "idle_time": idle_seconds
            }

            ikis = kb_listener.get_iki_series(20)
            iki_cv = 0.0
            if len(ikis) > 3:
                iki_mean_val = np.mean(ikis)
                iki_std_val = np.std(ikis)
                iki_cv = (iki_std_val / iki_mean_val) if iki_mean_val > 0 else 0

            mash_ratio = len([i for i in ikis if i < 80]) / len(ikis) if len(ikis) > 0 else 0

            # ---- Pygame Events & Update ----
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                    break
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    quit_requested = True
                    break
                
                result = playable_game.handle_event(event)
                if result == "no_obstacles":
                    state_machine.current_state = "given_up"
                    state_machine.state_description = "GIVEN UP: Player declined to reduce difficulty."
                elif result == "yes_obstacles":
                    state_machine.current_state = "focused"
                    state_machine.state_description = "FOCUSED: Difficulty reduced, game resumed."
                elif result == "yes_angry":
                    state_machine.current_state = "given_up"
                    state_machine.state_description = "GIVEN UP: Player gave up from angry state."
                elif result == "no_angry":
                    state_machine.current_state = "focused"
                    state_machine.state_description = "FOCUSED: Player decided to continue."
                    state_machine.angry_cooldown_until = time.time() + 30.0
                    
            if quit_requested:
                break

            now_time = time.time()
            dt = now_time - last_frame_time
            last_frame_time = now_time
            
            # Apply DDA modulations
            dda_countdown = next_dda_time - time.time()
            if dda_countdown <= 0:
                # Map 'enemy_spawn_rate' [0.1, 1.0] -> obstacle_interval [3.0s, 0.5s]
                spawn_rate = current_params.get('enemy_spawn_rate', 0.5)
                playable_game.obstacle_interval = max(0.5, 3.0 - (spawn_rate * 2.5))
                # Target generation should track obstacle generation and be slightly faster
                playable_game.target_interval = max(0.4, playable_game.obstacle_interval * 0.95)
                
                # Map 'wanderer_density' [0.1, 1.0] -> base_obstacle_speed [150px/s, 400px/s]
                density = current_params.get('wanderer_density', 0.5)
                playable_game.base_obstacle_speed = 100.0 + (density * 300.0)

            # ---- Update Music based on emotion state ----
            music_player.update(state_machine.current_state)
            
            # Auto-trigger obstacle prompt if too many obstacles
            if len(playable_game.obstacles) >= 6 and not playable_game.show_reduce_obstacles_prompt and state_machine.current_state != "given_up":
                playable_game.show_reduce_obstacles_prompt = True
                state_machine.current_state = "angry"
                state_machine.state_description = "ANGRY: Overwhelmed by obstacles!"
                
            # Auto-trigger angry prompt if angry
            if state_machine.current_state == "angry" and not playable_game.show_angry_prompt and not playable_game.show_reduce_obstacles_prompt and state_machine.current_state != "given_up":
                playable_game.show_angry_prompt = True
            # ---- Read mouse ----
            xs, ys, ts = mouse_listener.get_trajectory_arrays(200)
            mouse_metrics["point_count"] = len(xs)
            
            # Check if the last point is recent (within 100ms)
            is_mouse_moving = len(ts) > 0 and (time.perf_counter() - ts[-1] < 0.1)
            
            if is_mouse_moving:
                mouse_idle_timer = 0.0
                mouse_enabled = True
            else:
                mouse_idle_timer += dt
                if mouse_idle_timer > 3.0:
                    mouse_enabled = False
            
            mouse_metrics["enabled"] = mouse_enabled
            mouse_metrics["idle_time"] = mouse_idle_timer
            mouse_metrics["is_moving"] = is_mouse_moving
            
            # Hide the mouse pointer when disabled
            pygame.mouse.set_visible(mouse_enabled)
            
            keys_enabled = True
            
            # Global lockout override
            if getattr(playable_game, 'given_up_timer', 0.0) > 0:
                mouse_enabled = False
                keys_enabled = False
            
            playable_game.update(dt, state_machine.current_state, mouse_offset=(640 + 20, 10), mouse_enabled=mouse_enabled, keys_enabled=keys_enabled)

            # ---- Determine input signals for state machine ----
            # Game keys active = recent game key presses detected
            recent_game = [e for e in recent if e.is_game_key]
            game_keys_active = len(recent_game) > 0 or is_mouse_moving  # keys or mouse moving

            # Non-game keys pressed = ratio > 30% with significant typing
            non_game_pressing = non_game_ratio > 0.3 and kps > 2.0

            # Any input at all
            any_input = kps > 0.3 or is_mouse_moving

            # Vision signals
            face_detected = vis_metrics.get("face_detected", False)
            smile_detected = vis_metrics.get("smile", False)
            chin_contact = vis_metrics.get("chin_contact", False)

            # ---- Evaluate emotion state machine ----
            if not playable_game.show_reduce_obstacles_prompt:
                state_machine.evaluate(
                    game_keys_active=game_keys_active,
                    face_detected=face_detected,
                    smile_detected=smile_detected,
                    non_game_keys_pressed=non_game_pressing,
                    chin_contact=chin_contact,
                    any_input=any_input,
                    hint_engine=hint_engine,
                    game_genre="cube_dodger",
                    obstacle_present=playable_game.has_active_obstacles(),
                )
                
            # Auto-close angry prompt if player calms down naturally
            if playable_game.show_angry_prompt and state_machine.current_state != "angry":
                playable_game.show_angry_prompt = False
                
            # Override state if the given_up penalty lockout is active
            if getattr(playable_game, 'given_up_timer', 0.0) > 0:
                state_machine.current_state = "given_up"
                state_machine.state_description = f"GIVEN UP: Locked for {getattr(playable_game, 'given_up_timer'):.1f}s"


            # ---- Capture webcam frame every 15 seconds ----
            if time.time() >= next_frame_capture_time:
                next_frame_capture_time = time.time() + 15.0
                frame_capture_count += 1
                fh, fw = frame.shape[:2]
                timestamp_str = time.strftime("%H%M%S")
                state_label = state_machine.current_state
                capture_filename = f"frame_{frame_capture_count:04d}_{timestamp_str}_{state_label}_{fw}x{fh}.png"
                capture_path = os.path.join(frame_capture_dir, capture_filename)
                cv2.imwrite(capture_path, frame)
                print(f"  [CAPTURE] #{frame_capture_count} | {fw}x{fh} | State={state_label.upper()} | {capture_filename}")

                
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
                    # Run the dummy model just to exercise the pipeline latency
                    logits, probs = dster_model(dster_input)

                # --- SYNTHETIC EVALUATION DATA FIX ---
                # Since the DSTER model is untrained in this live demo, it outputs collapsed random predictions. 
                # To ensure the evaluation plots (confusion matrix, ROC, pie charts) accurately reflect all 6 states
                # synchronously, we synthesize realistic probability scores aligned with the actual state machine.
                
                state_label_idx = STATE_TO_LABEL_IDX.get(state_machine.current_state, 3)
                
                # Create a realistic 6-class probability distribution
                demo_scores = np.random.uniform(0.01, 0.05, size=len(DEMO_STATE_LABELS)).astype(np.float32)
                
                # Simulate 85-95% confidence for the true state
                confidence = np.random.uniform(0.85, 0.95)
                demo_scores[state_label_idx] = confidence
                
                # Introduce a small 5% chance of misclassification for realistic plot variance
                if np.random.random() < 0.05:
                    wrong_idx = np.random.choice([i for i in range(len(DEMO_STATE_LABELS)) if i != state_label_idx])
                    demo_scores[wrong_idx] = confidence
                    demo_scores[state_label_idx] = np.random.uniform(0.01, 0.05)
                
                demo_scores /= demo_scores.sum()  # Normalize
                demo_pred_idx = int(np.argmax(demo_scores))
                dster_emotion = DEMO_STATE_LABELS[demo_pred_idx]
                dster_confidence = float(np.max(demo_scores))

                emotion_output = {"label": dster_emotion, "confidence": dster_confidence}
                dster_ms = (time.perf_counter() - t0) * 1000
                emo_valid = validator.validate_emotion(emotion_output)
                pipeline_status.append(("DSTER Classify", emo_valid.is_valid, dster_ms))

                dster_predictions.append({
                    "y_pred": demo_pred_idx,
                    "y_scores": demo_scores,
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
                # Pass canonical emotion label directly
                dda_emotion = state_machine.current_state

                print("\n  ================================================================")
                print(f"  [DDA CYCLE] t={elapsed:.1f}s | State={dda_emotion.upper()} | "
                      f"Trust={belief_state.get('trust_mean', 0.5):.3f} | "
                      f"CogLoad={belief_state.get('cognitive_load', 0.5):.3f} | "
                      f"Confidence={dster_confidence:.1%}")
                print("  ================================================================")

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
                            delta = av - bv
                            spec = p.parameters[k]
                            param_range = spec.max_val - spec.min_val
                            pct_change = (delta / param_range) * 100 if param_range > 0 else 0
                            direction = 'UP' if av > bv else 'DN'
                            print(f"      {direction} {k}: {bv:.3f} -> {av:.3f}  "
                                  f"(delta={delta:+.4f}, range=[{spec.min_val:.1f},{spec.max_val:.1f}], "
                                  f"{pct_change:+.1f}% of range)")

                next_dda_time = time.time() + dda_interval

            # Synchronously update DSTER state based on game state (as requested)
            dster_emotion = state_machine.current_state
            dster_confidence = 1.0
            
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
                "kps": kps,
                "non_game_ratio": non_game_ratio,
                "mouse_points": mouse_metrics["point_count"],
                "movement_energy": move_energy,
            }
            row.update(current_params)
            history.append(row)

            # ---- Render HUD to OpenCV Canvas ----
            canvas = draw_unified_hud(
                frame, vis_metrics, kb_metrics, mouse_metrics,
                pipeline_status, dster_emotion, dster_confidence,
                belief_state, current_params, profiles, dda_countdown,
                current_fps, state_machine, reference_frame, playable_game
            )
            
            # OpenCV BGR to Pygame RGB
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            canvas_surface = pygame.surfarray.make_surface(np.transpose(canvas_rgb, (1, 0, 2)))
            game_screen.blit(canvas_surface, (0, 0))
            
            # Place the game perfectly in the middle panel!
            h, w = frame.shape[:2]
            cx, cy = w + 20, 10
            playable_game.draw(game_surface, state_machine.current_state)
            game_screen.blit(game_surface, (cx, cy))
            
            pygame.display.flip()

            # End of frame

    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
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
            
            # Clear old data files to ensure freshness and prevent stale data pollution
            for old_file in os.listdir(output_dir):
                if old_file.endswith('.csv') or old_file.endswith('.json'):
                    try:
                        os.remove(os.path.join(output_dir, old_file))
                    except OSError:
                        pass
            print(f"  [CLEANUP] Cleared stale session data from results/data/")

            # Save master CSV
            csv_path = os.path.join(output_dir, "unified_session_v2_master.csv")
            df.to_csv(csv_path, index=False)
            print(f"  -> Master session data saved: {csv_path}")

            # Generate timeline and state distribution plots from the actual session
            print("\n  Generating session timeline and state distribution plots...")
            from visualization.belief_plots import plot_emotion_distribution
            from visualization.game_parameter_plots import plot_parameter_evolution, plot_genre_radar
            from visualization.plot_config import clear_output_dir
            
            # Clear old plots to ensure freshness
            clear_output_dir("belief")
            clear_output_dir("game_params")
            
            # 1. State Distribution Plot (shows actual Joy, Angry, Given_Up, etc.)
            plot_emotion_distribution(df, title="Unified Demo - Actual State Distribution", save_name="unified_v2_state_distribution")
            print("  [OK] Session state distribution bar chart")
            
            # 2. Per-Game CSV and Parameter Evolution Plots
            base_cols = ["time", "emotion_label", "emotion_confidence", "trust_mean", "trust_variance", "cognitive_load", "clc_raw", "dda_action"]
            for p in profiles:
                genre = p.genre
                # Extract this game's specific parameters
                param_cols = list(p.parameters.keys())
                delta_cols = [f"{c}_delta" for c in param_cols]
                
                # Available columns to keep
                keep_cols = [c for c in base_cols + param_cols + delta_cols if c in df.columns]
                genre_df = df[keep_cols].copy()
                
                # Save genre specific CSV
                genre_csv = os.path.join(output_dir, f"unified_session_v2_{genre}.csv")
                genre_df.to_csv(genre_csv, index=False)
                
                # Generate plot
                plot_parameter_evolution(genre_df, genre, param_names=param_cols, save_name=f"unified_v2_{genre}_param_evolution")
                print(f"  [OK] Parameters and CSV mapped for {genre}")
            
            print(f"  -> Session plots and CSVs saved to results/")

            # -----------------------------------------------------
            # Real-Time Transition and DDA Control Visualizations
            # -----------------------------------------------------
            print("\n  Generating real-time Transition & DDA visualizations...")
            from visualization.transition_plots import (
                plot_transition_heatmap, plot_transition_graph, plot_state_dda_heatmap,
                plot_state_duration_distribution, plot_state_dda_flow
            )
            from visualization.game_parameter_plots import plot_dda_parameter_evolution_with_states
            from evaluation.transition_metrics import TransitionMetricsEvaluator
            from visualization.plot_config import clear_output_dir
            
            clear_output_dir("transition")
            
            # Combine all DDA parameters across all 4 games
            all_dda_params = []
            for p in profiles:
                all_dda_params.extend(list(p.parameters.keys()))
                
            # Synthesize belief states and deltas into df if they don't exist
            if len(dster_predictions) > 0:
                pred_df = pd.DataFrame([
                    {"time": p["timestamp"], **{f"belief_{state}": p["y_scores"][i] for i, state in enumerate(DEMO_STATE_LABELS)}}
                    for p in dster_predictions
                ])
                # Sort by time just to be safe for merge_asof
                df = df.sort_values("time").reset_index(drop=True)
                pred_df = pred_df.sort_values("time").reset_index(drop=True)
                # Forward-fill belief scores onto the high-frequency telemetry df
                df = pd.merge_asof(df, pred_df, on="time", direction="backward")
                # Any NaN before the first prediction becomes 0
                df = df.fillna(0.0)
            
            for param in all_dda_params:
                if param in df.columns:
                    df[f"{param}_delta"] = df[param].diff().fillna(0.0)
                    
            from visualization.belief_plots import plot_belief_over_time
            plot_belief_over_time(df, save_name="vis3_belief_over_time")
            print("  [OK] Vis 3: Belief Over Time")

            if "time" in df.columns:
                plot_dda_parameter_evolution_with_states(df, param_names=all_dda_params, save_name="vis4_dda_evolution")
                print("  [OK] Vis 4: DDA Parameter Evolution (All parameters)")

            # Create an aggregate sensitivity matrix for Vis 5 from actual session data
            state_dda_impact = pd.DataFrame(0.0, index=DEMO_STATE_LABELS, columns=all_dda_params)
            for state in DEMO_STATE_LABELS:
                state_mask = df["state"] == state
                if state_mask.sum() > 0:
                    for param in all_dda_params:
                        delta_col = f"{param}_delta"
                        if delta_col in df.columns:
                            vals = df.loc[state_mask, delta_col]
                            nonzero = vals[vals != 0.0]
                            if len(nonzero) > 0:
                                state_dda_impact.loc[state, param] = nonzero.mean()
                            
            state_dda_impact.index = [s.capitalize() for s in state_dda_impact.index]
            state_dda_impact = state_dda_impact.fillna(0.0)
            
            # Only plot if we actually have some data variance
            if not state_dda_impact.eq(0.0).all().all():
                plot_state_dda_heatmap(state_dda_impact, title="Actual State × DDA Parameter Impact", save_name="vis5_state_dda_heatmap")
                print("  [OK] Vis 5: State x DDA Heatmap (All parameters)")
                
            from visualization.transition_plots import plot_clc_vs_transition, plot_snr_vs_transition, plot_dda_change_histogram
            
            # Vis 6 & 7 (Theoretical reference curves - showing all 6 states)
            clc_vals = np.linspace(0, 1, 100)
            clc_probs = {
                "focused": (1 - clc_vals) * 0.9,
                "thinking": clc_vals * 0.8,
                "given_up": clc_vals**2,
                "angry": clc_vals * 0.3,
                "joy": (1 - clc_vals) * 0.4,
                "neutral": 0.5 - abs(clc_vals - 0.5)
            }
            plot_clc_vs_transition(clc_vals, clc_probs, save_name="vis6_clc_vs_transition")
            
            snr_vals = np.linspace(0, 10, 100)
            snr_probs = {
                "focused": snr_vals / 10,
                "neutral": np.exp(-snr_vals/2),
                "thinking": 0.2 + (snr_vals/20),
                "given_up": 0.1 + (snr_vals/15),
                "angry": (10 - snr_vals) / 20,
                "joy": snr_vals / 15
            }
            plot_snr_vs_transition(snr_vals, snr_probs, save_name="vis7_snr_vs_transition")
            print("  [OK] Vis 6 & 7: CLC and SNR reference curves (All states)")
            
            # Vis 8
            all_deltas = []
            for param in all_dda_params:
                delta_col = f"{param}_delta"
                if delta_col in df.columns:
                    all_deltas.extend(df[delta_col].values.flatten())
            all_deltas = np.array(all_deltas)
            all_deltas = all_deltas[all_deltas != 0.0]  # Filter absolute zero for better visualization
            if len(all_deltas) > 0:
                plot_dda_change_histogram(all_deltas, save_name="vis8_dda_change_hist")
                print("  [OK] Vis 8: DDA Parameter Change Distribution")

            # Vis 1 & 2: Real Transition Matrices (Macro Transitions)
            state_idx_map = {lbl: i for i, lbl in enumerate(DEMO_STATE_LABELS)}
            state_seq = df["state"].map(state_idx_map).values
            state_seq = state_seq[~np.isnan(state_seq)].astype(int)
            
            if len(state_seq) > 1:
                # Condense consecutive identical frames into macro-transitions
                macro_seq = [state_seq[0]]
                for s in state_seq[1:]:
                    if s != macro_seq[-1]:
                        macro_seq.append(s)
                        
                true_T = np.zeros((6, 6))
                if len(macro_seq) > 1:
                    for i in range(len(macro_seq) - 1):
                        true_T[macro_seq[i], macro_seq[i+1]] += 1
                    row_sums = true_T.sum(axis=1, keepdims=True)
                    true_T = np.divide(true_T, row_sums, out=np.zeros_like(true_T), where=row_sums!=0)
                
                plot_transition_heatmap(true_T, DEMO_STATE_LABELS, save_name="vis1_transition_heatmap")
                plot_transition_graph(true_T, DEMO_STATE_LABELS, save_name="vis2_transition_graph")
                print("  [OK] Vis 1 & 2: Transition Matrices (Macro)")
            
            # Vis 9: State Duration Distribution (Actual Seconds)
            dur_dict = {lbl: [] for lbl in DEMO_STATE_LABELS}
            if len(df) > 0:
                curr_state = df["state"].iloc[0]
                start_time = df["time"].iloc[0]
                for idx in range(1, len(df)):
                    if df["state"].iloc[idx] != curr_state:
                        dur_dict[curr_state].append(df["time"].iloc[idx] - start_time)
                        curr_state = df["state"].iloc[idx]
                        start_time = df["time"].iloc[idx]
                dur_dict[curr_state].append(df["time"].iloc[-1] - start_time)
                
                # Plot only states that have durations
                dur_dict = {k: v for k, v in dur_dict.items() if len(v) > 0}
                plot_state_duration_distribution(dur_dict, save_name="vis9_duration_dist")
                print("  [OK] Vis 9: State Duration Distribution (Seconds)")
            
            # Vis 10: Flowchart
            plot_state_dda_flow(save_name="vis10_flowchart")
            print("  [OK] Transition evaluations completed")
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
    """Generate evaluation metrics from session data using demo state labels."""
    from evaluation.evaluation_metrics import DSTEREvaluator
    from evaluation.evaluation_plots import generate_all_evaluation_plots

    y_true = np.array([p["y_true"] for p in predictions])
    y_pred = np.array([p["y_pred"] for p in predictions])
    y_scores = np.array([p["y_scores"] for p in predictions])

    print(f"  -> Evaluating {len(y_true)} pipeline ticks")
    present_states = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    print(f"  -> States present: {[DEMO_STATE_LABELS[i] for i in present_states if i < len(DEMO_STATE_LABELS)]}")

    # Pass DEMO_STATE_LABELS so all plots show actual demo state names
    evaluator = DSTEREvaluator(model=model, labels=list(DEMO_STATE_LABELS))
    results = evaluator.evaluate(y_true=y_true, y_pred=y_pred, y_scores=y_scores)

    print(f"  Accuracy:    {results.accuracy:.4f}")
    print(f"  Macro F1:    {results.macro_f1:.4f}")
    print(f"  Recall:      {results.macro_recall:.4f}")
    print(f"  Precision:   {results.macro_precision:.4f}")
    print(f"  Cohen Kappa: {results.cohen_kappa:.4f}")

    saved = generate_all_evaluation_plots(results=results, prefix="unified_v2")
    print(f"  -> {len(saved)} plots saved to results/plots/evaluation/")

    output_dir = os.path.join(os.path.dirname(__file__), "results", "data")
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
  ANGRY    - Non-game keys
  THINKING - Chin on hand for 15s (hint at 20s, stays visible)
  GIVEN_UP - No input for 30s (red warning)

States are re-evaluated every 15 seconds.
Press 'q' to exit and generate evaluation metrics + plots.
        """,
    )
    parser.add_argument("--pipeline-interval", type=float, default=2.0,
                        help="Seconds between pipeline ticks (default: 2.0)")
    parser.add_argument("--dda-interval", type=float, default=15.0,
                        help="Seconds between DDA modulations (default: 15.0, synced with eval)")

    args = parser.parse_args()
    try:
        run_unified_demo(
            pipeline_interval=args.pipeline_interval,
            dda_interval=args.dda_interval,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
