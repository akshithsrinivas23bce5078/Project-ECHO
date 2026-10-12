"""
Project ECHO — Mouse Trajectory Listener
=========================================
Captures mouse movement and click events via pynput.
Stores trajectory points (x, y, t) for downstream Sigma-Lognormal
neuromotor kinematics processing.

Phase 1 Upgrade: Trajectory Segmentation
------------------------------------------
Adds movement pause detection and segment extraction for the IDeLog
real-time neuromotor parameter extraction algorithm. Continuous
trajectories are split at velocity pauses (< threshold for > pause_ms)
into discrete ballistic segments.
"""

import time
import threading
from collections import deque
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

try:
    from pynput import mouse
except ImportError:
    mouse = None  # Graceful fallback


class TrajectoryPoint:
    """A single mouse position sample."""

    __slots__ = ("x", "y", "t")

    def __init__(self, x: int, y: int, t: float):
        self.x = x
        self.y = y
        self.t = t  # time.perf_counter()


class MouseClickEvent:
    """A single mouse click event."""

    __slots__ = ("x", "y", "t", "button", "pressed")

    def __init__(self, x: int, y: int, t: float, button: str, pressed: bool):
        self.x = x
        self.y = y
        self.t = t
        self.button = button
        self.pressed = pressed


# ---------------------------------------------------------------
# Trajectory Segment (Phase 1)
# ---------------------------------------------------------------

@dataclass
class TrajectorySegmentData:
    """
    A continuous ballistic movement segment between pauses.

    Produced by MouseListener.get_trajectory_segments() for
    downstream IDeLog processing.
    """
    segment_id: int
    xs: np.ndarray
    ys: np.ndarray
    ts: np.ndarray
    t_start: float  # Temporal bound: start time
    t_end: float    # Temporal bound: end time
    point_count: int


class MouseListener:
    """
    Real-time mouse trajectory capture engine.

    Maintains a rolling buffer of trajectory points and click events
    for Sigma-Lognormal analysis of neuromotor kinematics.
    """

    def __init__(self, buffer_size: int = 2000,
                 sample_interval_ms: float = 5.0,
                 pause_velocity_threshold: float = 5.0,
                 pause_duration_ms: float = 100.0):
        """
        Args:
            buffer_size: Maximum trajectory points to retain.
            sample_interval_ms: Minimum ms between recorded move samples
                                (rate-limiting to avoid flooding).
            pause_velocity_threshold: Velocity (px/s) below which movement
                                      is considered a pause.
            pause_duration_ms: Duration (ms) of low velocity to confirm
                               a segment boundary.
        """
        self._trajectory: deque = deque(maxlen=buffer_size)
        self._clicks: deque = deque(maxlen=500)
        self._sample_interval = sample_interval_ms / 1000.0
        self._last_sample_time: float = 0.0
        self._listener = None
        self._lock = threading.Lock()
        self._running = False

        # Phase 1: Segmentation parameters
        self._pause_velocity_threshold = pause_velocity_threshold
        self._pause_duration_ms = pause_duration_ms
        self._segment_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin capturing mouse events in a background thread."""
        if mouse is None:
            raise RuntimeError(
                "pynput is not installed. Install with: pip install pynput"
            )
        if self._running:
            return
        self._running = True
        self._listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
        )
        self._listener.start()

    def stop(self) -> None:
        """Stop capturing and join the listener thread."""
        self._running = False
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def get_trajectory(self, n: int = 200) -> List[TrajectoryPoint]:
        """Return the last *n* trajectory points."""
        with self._lock:
            items = list(self._trajectory)
        return items[-n:]

    def get_trajectory_arrays(self, n: int = 200) -> Tuple[List[float], List[float], List[float]]:
        """
        Return trajectory as three parallel lists (xs, ys, ts)
        suitable for NumPy array conversion.
        """
        points = self.get_trajectory(n)
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        ts = [p.t for p in points]
        return xs, ys, ts

    def get_recent_clicks(self, n: int = 20) -> List[MouseClickEvent]:
        """Return the last *n* click events."""
        with self._lock:
            items = list(self._clicks)
        return items[-n:]

    # ------------------------------------------------------------------
    # Trajectory Segmentation API (Phase 1)
    # ------------------------------------------------------------------

    def get_trajectory_segments(
        self,
        n: int = 500,
        min_points: int = 10,
    ) -> List[TrajectorySegmentData]:
        """
        Split the trajectory buffer into continuous ballistic segments.

        Segments are separated at movement pauses — regions where the
        instantaneous velocity drops below `pause_velocity_threshold`
        for longer than `pause_duration_ms`.

        Args:
            n: Number of trajectory points to consider.
            min_points: Minimum points per segment (discard shorter ones).

        Returns:
            List of TrajectorySegmentData objects ordered chronologically.
        """
        points = self.get_trajectory(n)
        if len(points) < min_points:
            return []

        xs = np.array([p.x for p in points], dtype=np.float64)
        ys = np.array([p.y for p in points], dtype=np.float64)
        ts = np.array([p.t for p in points], dtype=np.float64)

        # Compute instantaneous velocity
        dx = np.diff(xs)
        dy = np.diff(ys)
        dt = np.diff(ts)
        dt = np.where(dt == 0, 1e-6, dt)
        velocity = np.sqrt(dx ** 2 + dy ** 2) / dt

        # Detect pause regions
        pause_threshold_s = self._pause_duration_ms / 1000.0
        is_slow = velocity < self._pause_velocity_threshold

        # Find segment boundaries (transitions from slow→fast or fast→slow)
        segment_boundaries = [0]
        in_pause = False
        pause_start = 0

        for i in range(len(is_slow)):
            if is_slow[i] and not in_pause:
                in_pause = True
                pause_start = i
            elif not is_slow[i] and in_pause:
                pause_duration = ts[i + 1] - ts[pause_start + 1] if pause_start + 1 < len(ts) else 0
                if pause_duration >= pause_threshold_s:
                    segment_boundaries.append(i + 1)  # +1 for position array offset
                in_pause = False

        segment_boundaries.append(len(xs))

        # Build segment objects
        segments = []
        for i in range(len(segment_boundaries) - 1):
            start = segment_boundaries[i]
            end = segment_boundaries[i + 1]

            if end - start < min_points:
                continue

            seg_xs = xs[start:end]
            seg_ys = ys[start:end]
            seg_ts = ts[start:end]

            self._segment_counter += 1
            segments.append(TrajectorySegmentData(
                segment_id=self._segment_counter,
                xs=seg_xs,
                ys=seg_ys,
                ts=seg_ts,
                t_start=float(seg_ts[0]),
                t_end=float(seg_ts[-1]),
                point_count=len(seg_xs),
            ))

        return segments

    # ------------------------------------------------------------------
    # Private handlers
    # ------------------------------------------------------------------

    def _on_move(self, x: int, y: int) -> None:
        now = time.perf_counter()
        # Rate-limit sampling
        if (now - self._last_sample_time) < self._sample_interval:
            return
        self._last_sample_time = now

        point = TrajectoryPoint(x, y, now)
        with self._lock:
            self._trajectory.append(point)

    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        now = time.perf_counter()
        event = MouseClickEvent(x, y, now, str(button), pressed)
        with self._lock:
            self._clicks.append(event)

