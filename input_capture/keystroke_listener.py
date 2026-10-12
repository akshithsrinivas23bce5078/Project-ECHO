"""
Project ECHO — Keystroke Listener
=================================
Captures key-press and key-release events via pynput.
Computes inter-keystroke intervals (IKI) and stores digraph pairs
for downstream CLC estimation and DSTER feature extraction.

Phase 1 Upgrade: Dynamic Overlapping Sliding Windows
-----------------------------------------------------
Adds a SlidingWindowBuffer that produces overlapping temporal windows
with configurable overlap ratio and minimum density gating. This
enables dense temporal sampling for the DSTER multi-head self-attention
to capture transient micro-behavioral anomalies (hesitations, rapid
backspace corrections) that signify cognitive load spikes.

This module is non-intrusive: it records timing metadata only,
never the content of the keys themselves, preserving user privacy.
"""

import time
import threading
from collections import deque
from typing import Callable, Optional, List, Tuple

try:
    from pynput import keyboard
except ImportError:
    keyboard = None  # Graceful fallback for environments without pynput


# ---------------------------------------------------------------
# Explicit Game Key Definitions
# ---------------------------------------------------------------
# WASD, Space, Shift (L/R), Ctrl (L/R), Enter, Escape, Tab,
# Arrow keys, and common game modifier letters (R, E, F, Q).
# Any key NOT in this set is a "non-game key" — a primary
# signal for frustration / anger detection.
# ---------------------------------------------------------------
GAME_KEY_CHARS = frozenset([
    'w', 'a', 's', 'd',           # Movement
    'W', 'A', 'S', 'D',
    'r', 'R', 'e', 'E',           # Reload / Interact
    'f', 'F', 'q', 'Q',           # Melee / Quick-switch
    'c', 'C',                     # Crouch
    '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Weapon slots
    # Note: Arrow keys (Up/Down/Left/Right), Space, Shift, Tab, Enter 
    # are handled in GAME_KEY_SPECIALS below!
])

# pynput special keys that are valid game controls
GAME_KEY_SPECIALS: frozenset = frozenset()   # populated at import


def _build_game_key_specials():
    """Build the set of pynput special keys considered game controls."""
    if keyboard is None:
        return frozenset()
    keys = []
    for attr in [
        'space', 'shift', 'shift_l', 'shift_r',
        'ctrl', 'ctrl_l', 'ctrl_r',
        'enter', 'esc', 'tab',
        'up', 'down', 'left', 'right',
    ]:
        k = getattr(keyboard.Key, attr, None)
        if k is not None:
            keys.append(k)
    return frozenset(keys)


if keyboard is not None:
    GAME_KEY_SPECIALS = _build_game_key_specials()


class KeystrokeEvent:
    """A single timestamped keystroke event."""

    __slots__ = ("key_id", "event_type", "timestamp", "is_game_key")

    def __init__(self, key_id: str, event_type: str, timestamp: float, is_game_key: bool = True):
        self.key_id = key_id            # Anonymized key identifier
        self.event_type = event_type    # "press" or "release"
        self.timestamp = timestamp      # time.perf_counter() value
        self.is_game_key = is_game_key  # True if WASD/Space/Shift/etc.


# ---------------------------------------------------------------
# Sliding Window Buffer (Phase 1)
# ---------------------------------------------------------------

class SlidingWindowBuffer:
    """
    Manages dynamic, overlapping temporal windows over keystroke events.

    Instead of providing a flat list of events, this buffer produces
    overlapping windows that ensure dense temporal sampling for the
    DSTER Transformer model. The overlap guarantees that transient
    micro-behavioral anomalies at window boundaries are never lost.

    Parameters:
        window_size: Number of events per window (matches DSTER seq_len).
        overlap_ratio: Fraction of overlap between consecutive windows
                       (0.5 = 50% overlap, producing 2× temporal slices).
        min_density: Minimum number of press events required within a
                     window for it to be considered valid.
    """

    def __init__(
        self,
        window_size: int = 50,
        overlap_ratio: float = 0.5,
        min_density: int = 20,
    ):
        if not 0.0 < overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in (0, 1), got {overlap_ratio}")
        if min_density > window_size:
            raise ValueError(
                f"min_density ({min_density}) cannot exceed window_size ({window_size})"
            )

        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.min_density = min_density
        self.stride = max(1, int(window_size * (1.0 - overlap_ratio)))

        # Completed windows ready for consumption
        self._completed_windows: deque = deque(maxlen=50)
        # Window-ready callback
        self._on_window_ready: Optional[Callable] = None

    @property
    def on_window_ready(self) -> Optional[Callable]:
        """Callback invoked when a new complete window becomes available."""
        return self._on_window_ready

    @on_window_ready.setter
    def on_window_ready(self, callback: Optional[Callable]):
        self._on_window_ready = callback

    def update(self, events: List[KeystrokeEvent]) -> List[List[KeystrokeEvent]]:
        """
        Given the full event buffer, extract any new complete overlapping
        windows and return them.

        Args:
            events: The full event buffer (list of KeystrokeEvent).

        Returns:
            List of newly completed windows (each a list of KeystrokeEvent).
        """
        if len(events) < self.window_size:
            return []

        new_windows = []
        n = len(events)

        # Generate all possible windows with the configured stride
        for start in range(0, n - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = events[start:end]

            # Density gate: ensure minimum press events
            press_count = sum(1 for e in window if e.event_type == "press")
            if press_count >= self.min_density:
                new_windows.append(window)

        # Determine truly new windows (not already emitted)
        prev_count = len(self._completed_windows)
        for w in new_windows[prev_count:]:
            self._completed_windows.append(w)
            if self._on_window_ready is not None:
                self._on_window_ready(w)

        return new_windows

    def get_all_windows(self, events: List[KeystrokeEvent]) -> List[List[KeystrokeEvent]]:
        """
        Generate all valid overlapping windows from the event buffer.

        This is a stateless extraction — does not track previously
        emitted windows. Use for batch processing.

        Args:
            events: The full event buffer.

        Returns:
            List of overlapping windows passing the density gate.
        """
        if len(events) < self.window_size:
            return []

        windows = []
        n = len(events)

        for start in range(0, n - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = events[start:end]

            press_count = sum(1 for e in window if e.event_type == "press")
            if press_count >= self.min_density:
                windows.append(window)

        return windows

    def get_latest_window(self, events: List[KeystrokeEvent]) -> Optional[List[KeystrokeEvent]]:
        """
        Return the most recent complete window from the event buffer,
        or None if insufficient data.

        Args:
            events: The full event buffer.

        Returns:
            The latest window passing the density gate, or None.
        """
        if len(events) < self.window_size:
            return None

        window = events[-self.window_size:]
        press_count = sum(1 for e in window if e.event_type == "press")

        if press_count >= self.min_density:
            return window
        return None


class KeystrokeListener:
    """
    Real-time keystroke capture engine.

    Maintains a rolling buffer of keystroke events and computes
    inter-keystroke intervals (IKI) and digraph latencies on the fly.

    Key mapping:
        WASD, Space, Shift (L/R), Ctrl, Enter, Esc, Tab, Arrows
        → game keys (is_game_key=True).
        Everything else → non-game keys (frustration signal).
    """

    def __init__(self, buffer_size: int = 500,
                 on_digraph: Optional[Callable] = None,
                 window_size: int = 50,
                 overlap_ratio: float = 0.5,
                 min_density: int = 20):
        """
        Args:
            buffer_size: Maximum number of events to retain in the ring buffer.
            on_digraph: Optional callback invoked with (digraph_pair, latency_ms)
                        each time a new digraph is recorded.
            window_size: Events per sliding window (matches DSTER seq_len).
            overlap_ratio: Overlap fraction between consecutive windows.
            min_density: Minimum press events required per valid window.
        """
        self._buffer: deque = deque(maxlen=buffer_size)
        self._digraphs: deque = deque(maxlen=buffer_size)
        self._on_digraph = on_digraph
        self._last_press_time: Optional[float] = None
        self._last_key_id: Optional[str] = None
        self._listener = None
        self._lock = threading.Lock()
        self._running = False

        # Idle tracking — timestamp of most recent key press
        self._last_any_press_time: float = time.perf_counter()
        
        # Track currently held game keys for continuous 'focused' state
        self._held_game_keys: set = set()

        # Phase 1: Sliding window buffer for DSTER integration
        self._window_buffer = SlidingWindowBuffer(
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            min_density=min_density,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    def start(self) -> None:
        """Begin capturing keystrokes and mouse activity in background threads."""
        if keyboard is None:
            raise RuntimeError(
                "pynput is not installed. Install with: pip install pynput"
            )
        if self._running:
            return
        self._running = True
        self._last_any_press_time = time.perf_counter()
        
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        
        # Add mouse listener to track mouse activity
        try:
            from pynput import mouse
            self._mouse_listener = mouse.Listener(
                on_move=self._on_mouse_activity,
                on_click=self._on_mouse_activity,
                on_scroll=self._on_mouse_activity
            )
            self._mouse_listener.start()
        except ImportError:
            self._mouse_listener = None

    def stop(self) -> None:
        """Stop capturing and join the listener threads."""
        self._running = False
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if hasattr(self, '_mouse_listener') and self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None

    def get_recent_events(self, n: int = 50) -> List[KeystrokeEvent]:
        """Return the last *n* keystroke events."""
        with self._lock:
            items = list(self._buffer)
        return items[-n:]

    def get_digraph_latencies(self, n: int = 50) -> List[Tuple[str, float]]:
        """
        Return the last *n* digraph latencies.

        Each entry is (digraph_pair, latency_ms) where digraph_pair
        is a string like "a->b" and latency_ms is the inter-key interval.
        """
        with self._lock:
            items = list(self._digraphs)
        return items[-n:]

    def get_iki_series(self, n: int = 50) -> List[float]:
        """Return raw IKI values (ms) for the last *n* digraphs."""
        return [lat for _, lat in self.get_digraph_latencies(n)]

    def get_idle_seconds(self) -> float:
        """Return seconds since the last key press of any kind."""
        with self._lock:
            return time.perf_counter() - self._last_any_press_time

    def get_non_game_ratio(self, n: int = 30) -> float:
        """
        Return the fraction of non-game-key presses in the last *n* press events.

        Returns 0.0 if there are no recent press events.
        A high ratio (> 0.4) is a strong frustration/anger signal.
        """
        presses = [e for e in self.get_recent_events(n * 2) if e.event_type == 'press'][-n:]
        if not presses:
            return 0.0
        non_game = sum(1 for e in presses if not e.is_game_key)
        return non_game / len(presses)

    def is_game_key_held(self) -> bool:
        """Return True if any game key (WASD/Space/etc.) is currently held down."""
        with self._lock:
            return len(self._held_game_keys) > 0

    # ------------------------------------------------------------------
    # Sliding Window API (Phase 1)
    # ------------------------------------------------------------------

    @property
    def window_buffer(self) -> SlidingWindowBuffer:
        """Access the underlying SlidingWindowBuffer for advanced config."""
        return self._window_buffer

    def get_sliding_windows(self) -> List[List[KeystrokeEvent]]:
        """
        Generate all valid overlapping windows from the current event buffer.

        Returns:
            List of overlapping windows, each containing `window_size`
            KeystrokeEvent objects. Windows that fail the density gate
            (too few press events) are excluded.
        """
        with self._lock:
            events = list(self._buffer)
        return self._window_buffer.get_all_windows(events)

    def get_current_window(self) -> Optional[List[KeystrokeEvent]]:
        """
        Return the most recent complete window for real-time inference.

        Returns:
            The latest window of `window_size` events passing the
            density gate, or None if insufficient data.
        """
        with self._lock:
            events = list(self._buffer)
        return self._window_buffer.get_latest_window(events)

    # ------------------------------------------------------------------
    # Private handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _anonymize_key(key) -> str:
        """
        Convert a pynput key object into a privacy-safe identifier.
        We hash the key rather than storing the literal character.
        """
        try:
            return f"k_{hash(key.char) % 10000:04d}"
        except AttributeError:
            return f"s_{hash(key) % 10000:04d}"

    @staticmethod
    def _is_game_key(key) -> bool:
        """Check if the key belongs to the explicit GAME_KEYS set."""
        try:
            if hasattr(key, 'char') and key.char in GAME_KEY_CHARS:
                return True
        except Exception:
            pass
        try:
            if key in GAME_KEY_SPECIALS:
                return True
        except Exception:
            pass
        return False

    def _on_press(self, key) -> None:
        now = time.perf_counter()
        key_id = self._anonymize_key(key)
        is_game = self._is_game_key(key)
        event = KeystrokeEvent(key_id, "press", now, is_game)

        with self._lock:
            self._buffer.append(event)
            self._last_any_press_time = now
            if is_game:
                self._held_game_keys.add(key_id)

            # Compute digraph latency from the previous press
            if self._last_press_time is not None and self._last_key_id is not None:
                latency_ms = (now - self._last_press_time) * 1000.0
                pair = f"{self._last_key_id}->{key_id}"
                self._digraphs.append((pair, latency_ms))

                if self._on_digraph is not None:
                    self._on_digraph(pair, latency_ms)

            self._last_press_time = now
            self._last_key_id = key_id

    def _on_release(self, key) -> None:
        now = time.perf_counter()
        key_id = self._anonymize_key(key)
        is_game = self._is_game_key(key)
        event = KeystrokeEvent(key_id, "release", now, is_game)

        with self._lock:
            self._buffer.append(event)
            if is_game:
                self._held_game_keys.discard(key_id)

    def _on_mouse_activity(self, *args) -> None:
        """Handler for mouse movement, clicks, and scrolls."""
        now = time.perf_counter()
        with self._lock:
            self._last_any_press_time = now
