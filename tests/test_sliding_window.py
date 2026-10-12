"""
Project ECHO — Sliding Window Tests
======================================
Validates the dynamic overlapping sliding window buffer for
DSTER integration (Phase 1).

Tests:
    1. Overlap ratio produces correct number of windows
    2. Minimum density gating excludes sparse windows
    3. Backward compatibility of get_recent_events()
    4. get_current_window() returns valid latest window
    5. SlidingWindowBuffer callback mechanism
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from input_capture.keystroke_listener import (
    KeystrokeEvent, SlidingWindowBuffer, KeystrokeListener,
)


class TestSlidingWindowBuffer:
    """Tests for the SlidingWindowBuffer class."""

    def test_overlap_produces_correct_window_count(self):
        """
        With 100 events, window_size=50, overlap=0.5 (stride=25):
        windows start at 0, 25, 50 → 3 windows.
        (All press events to pass density gate.)
        """
        events = [
            KeystrokeEvent(f"k_{i:04d}", "press", float(i), True)
            for i in range(100)
        ]
        buf = SlidingWindowBuffer(window_size=50, overlap_ratio=0.5, min_density=1)
        windows = buf.get_all_windows(events)

        assert len(windows) == 3
        assert len(windows[0]) == 50
        assert len(windows[1]) == 50
        assert len(windows[2]) == 50

    def test_overlap_ratio_75_percent(self):
        """
        With 100 events, window_size=40, overlap=0.75 (stride=10):
        windows start at 0, 10, 20, 30, 40, 50, 60 → 7 windows.
        """
        events = [
            KeystrokeEvent(f"k_{i:04d}", "press", float(i), True)
            for i in range(100)
        ]
        buf = SlidingWindowBuffer(window_size=40, overlap_ratio=0.75, min_density=1)
        windows = buf.get_all_windows(events)

        assert len(windows) == 7

    def test_density_gate_excludes_sparse_windows(self):
        """Windows with fewer press events than min_density should be excluded."""
        # Create events: first 30 are presses, rest are releases
        events = []
        for i in range(100):
            evt_type = "press" if i < 30 else "release"
            events.append(KeystrokeEvent(f"k_{i:04d}", evt_type, float(i), True))

        # Require 25 presses per window — only first window (0–49) has 30 presses
        buf = SlidingWindowBuffer(window_size=50, overlap_ratio=0.5, min_density=25)
        windows = buf.get_all_windows(events)

        assert len(windows) == 1  # Only the first window passes density gate

    def test_insufficient_data_returns_empty(self):
        """Buffer smaller than window_size returns no windows."""
        events = [
            KeystrokeEvent(f"k_{i:04d}", "press", float(i), True)
            for i in range(10)
        ]
        buf = SlidingWindowBuffer(window_size=50, overlap_ratio=0.5, min_density=1)
        windows = buf.get_all_windows(events)

        assert len(windows) == 0

    def test_latest_window_returns_most_recent(self):
        """get_latest_window should return the last window_size events."""
        events = [
            KeystrokeEvent(f"k_{i:04d}", "press", float(i), True)
            for i in range(100)
        ]
        buf = SlidingWindowBuffer(window_size=50, overlap_ratio=0.5, min_density=1)
        window = buf.get_latest_window(events)

        assert window is not None
        assert len(window) == 50
        # Should be the last 50 events
        assert window[0].key_id == "k_0050"
        assert window[-1].key_id == "k_0099"

    def test_latest_window_density_gate(self):
        """get_latest_window returns None if density gate fails."""
        events = [
            KeystrokeEvent(f"k_{i:04d}", "release", float(i), True)
            for i in range(100)
        ]
        buf = SlidingWindowBuffer(window_size=50, overlap_ratio=0.5, min_density=20)
        window = buf.get_latest_window(events)

        assert window is None

    def test_callback_fires_on_new_windows(self):
        """on_window_ready callback should fire for new windows."""
        events = [
            KeystrokeEvent(f"k_{i:04d}", "press", float(i), True)
            for i in range(100)
        ]
        received = []
        buf = SlidingWindowBuffer(window_size=50, overlap_ratio=0.5, min_density=1)
        buf.on_window_ready = lambda w: received.append(len(w))

        buf.update(events)

        assert len(received) == 3
        assert all(n == 50 for n in received)

    def test_invalid_overlap_ratio_raises(self):
        """Overlap ratio outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError):
            SlidingWindowBuffer(window_size=50, overlap_ratio=0.0)
        with pytest.raises(ValueError):
            SlidingWindowBuffer(window_size=50, overlap_ratio=1.0)

    def test_min_density_exceeds_window_raises(self):
        """min_density > window_size should raise ValueError."""
        with pytest.raises(ValueError):
            SlidingWindowBuffer(window_size=50, min_density=60)


class TestKeystrokeListenerWindows:
    """Tests for sliding window integration in KeystrokeListener."""

    def test_backward_compat_get_recent_events(self):
        """get_recent_events() must still work after Phase 1 upgrade."""
        listener = KeystrokeListener(buffer_size=100)
        # Manually inject events
        for i in range(20):
            evt = KeystrokeEvent(f"k_{i:04d}", "press", float(i), True)
            listener._buffer.append(evt)

        events = listener.get_recent_events(10)
        assert len(events) == 10
        assert events[-1].key_id == "k_0019"

    def test_get_sliding_windows_integration(self):
        """get_sliding_windows() should delegate to SlidingWindowBuffer."""
        listener = KeystrokeListener(
            buffer_size=200, window_size=50, overlap_ratio=0.5, min_density=1
        )
        for i in range(100):
            evt = KeystrokeEvent(f"k_{i:04d}", "press", float(i), True)
            listener._buffer.append(evt)

        windows = listener.get_sliding_windows()
        assert len(windows) == 3

    def test_get_current_window_integration(self):
        """get_current_window() should return the latest window."""
        listener = KeystrokeListener(
            buffer_size=200, window_size=50, overlap_ratio=0.5, min_density=1
        )
        for i in range(80):
            evt = KeystrokeEvent(f"k_{i:04d}", "press", float(i), True)
            listener._buffer.append(evt)

        window = listener.get_current_window()
        assert window is not None
        assert len(window) == 50

    def test_window_buffer_property(self):
        """window_buffer property should expose the SlidingWindowBuffer."""
        listener = KeystrokeListener(buffer_size=100, window_size=30)
        assert isinstance(listener.window_buffer, SlidingWindowBuffer)
        assert listener.window_buffer.window_size == 30
