"""
Project ECHO — Hint Engine
==============================
Game-specific tiered hint system triggered by the Thinking
state modifier. Based on Vygotsky's Zone of Proximal Development
applied to gaming — scaffolding the player rather than just
adjusting difficulty numbers.

Hint Tiers:
    Tier 1 (Nudge)    — 10s thinking: vague directional nudge
    Tier 2 (Guide)    — 20s thinking: actionable tactical advice
    Tier 3 (Solution) — 30s thinking: explicit solution/path

Hints rotate within each tier to avoid repetition.
"""

import random
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------
# Game-Specific Hint Pools
# ------------------------------------------------------------------
HINT_POOLS: Dict[str, Dict[str, List[str]]] = {

    "left_4_dead": {
        "nudge": [
            "Check nearby rooms for health packs",
            "Listen for audio cues - special infected are close",
            "Stick together with your team for safety",
            "Try a different route through the map",
        ],
        "guide": [
            "The safe room is usually in the direction of fewer zombies",
            "Pipe bombs can clear a horde instantly - check your inventory",
            "Crouch behind cover during horde events to reduce damage",
            "Bile jars attract zombies away from your position",
        ],
        "solution": [
            "Head north - the safe room door has a red light",
            "Use the mounted gun to clear the courtyard, then rush",
            "The shortcut is through the building on your left",
            "Throw a pipe bomb, then sprint past the horde",
        ],
    },

    "alien_isolation": {
        "nudge": [
            "The alien reacts to sound - try crouching",
            "Check your motion tracker for movement",
            "There might be a hiding spot nearby",
            "Conserve your flamethrower fuel",
        ],
        "guide": [
            "Lockers and vents provide safe hiding spots",
            "The rewire panel can disable cameras and alarms",
            "Move slowly - the alien tracks footstep noise",
            "Use the environment to distract the alien",
        ],
        "solution": [
            "Use the noisemaker to lure the alien away",
            "The vent behind you leads to a safe area",
            "Wait for the alien to leave, then sprint to the door",
            "The access code is on the note beside the terminal",
        ],
    },

    "resident_evil_4": {
        "nudge": [
            "Aim for weak points for bonus damage",
            "Check your map for unexplored areas",
            "Some enemies drop valuable items when defeated",
            "Try upgrading your current weapon",
        ],
        "guide": [
            "Conserve ammo - use the knife on downed enemies",
            "Environmental objects like barrels can deal area damage",
            "The merchant has new stock - check for upgrades",
            "Combine herbs for stronger healing effects",
        ],
        "solution": [
            "Shoot the red barrel near the enemy cluster",
            "The key item is in the locked cabinet upstairs",
            "Use a flash grenade to stun the group, then knife them",
            "Sell the old weapon to afford the rifle upgrade",
        ],
    },

    "silent_hill": {
        "nudge": [
            "Examine objects closely - some have hidden details",
            "Your flashlight can reveal hidden writing",
            "Pay attention to the radio static pattern",
            "The environment changes based on your actions",
        ],
        "guide": [
            "The symbols on the wall match the puzzle pattern",
            "Try comparing the map to the environment layout",
            "Some doors only open after specific events trigger",
            "Backtrack to the room where the clock was ticking",
        ],
        "solution": [
            "Rotate the third dial to match the mirror image",
            "The answer is written on the back of the photo",
            "Enter the code from the poem: 3-1-4-1",
            "The key is hidden under the piano bench",
        ],
    },

    "cube_dodger": {
        "nudge": [
            "Where should you focus your eyes?",
            "Are your movements too large?",
            "What should you be looking at?",
        ],
        "guide": [
            "Where is the safest place to stand?",
            "Can you anticipate where the blocks will fall?",
            "Should you be moving constantly?",
        ],
        "solution": [
            "Focus on the bottom half and make smaller movements.",
            "Stay near the middle, avoid corners, and anticipate drops.",
            "Only move when absolutely necessary to dodge.",
        ],
    },
}


class HintEngine:
    """
    Manages game-specific, tiered hint delivery.

    Hints are triggered by the Thinking state modifier and
    escalate from vague nudges to explicit solutions based
    on how long the player has been idle.

    Usage:
        engine = HintEngine()
        hint = engine.get_hint("left_4_dead", thinking_seconds=12.0)
        # Returns ("Check nearby rooms for health packs", "nudge")
    """

    # Tier thresholds (seconds of continuous thinking)
    TIER_THRESHOLDS = {
        "nudge":    0.0,    # instant → vague hint
        "guide":    10.0,   # 10s idle → tactical advice
        "solution": 20.0,   # 20s idle → explicit answer
    }

    def __init__(self):
        # Track last hint index per (genre, tier) to avoid repeats
        self._last_index: Dict[Tuple[str, str], int] = {}

    def get_tier(self, thinking_seconds: float) -> Optional[str]:
        """
        Determine the hint tier based on thinking duration.

        Returns None if thinking_seconds < 10 (no hint yet).
        """
        if thinking_seconds >= self.TIER_THRESHOLDS["solution"]:
            return "solution"
        elif thinking_seconds >= self.TIER_THRESHOLDS["guide"]:
            return "guide"
        elif thinking_seconds >= self.TIER_THRESHOLDS["nudge"]:
            return "nudge"
        return None

    def get_hint(
        self, genre: str, thinking_seconds: float
    ) -> Optional[Tuple[str, str]]:
        """
        Get a hint for the given game genre and thinking duration.

        Args:
            genre: Game genre key (e.g., "left_4_dead").
            thinking_seconds: How long the player has been in thinking state.

        Returns:
            Tuple of (hint_text, tier_name) or None if no hint is due yet.
        """
        tier = self.get_tier(thinking_seconds)
        if tier is None:
            return None

        pool = HINT_POOLS.get(genre, {}).get(tier, [])
        if not pool:
            # Fallback generic hints
            pool = self._generic_hints(tier)

        # Rotate: pick next hint, avoid repeating the last one
        key = (genre, tier)
        last_idx = self._last_index.get(key, -1)
        next_idx = (last_idx + 1) % len(pool)
        self._last_index[key] = next_idx

        return (pool[next_idx], tier)

    def reset(self):
        """Reset hint rotation state (e.g., at session start)."""
        self._last_index.clear()

    @staticmethod
    def _generic_hints(tier: str) -> List[str]:
        """Fallback hints for unknown game genres."""
        generic = {
            "nudge": [
                "Try exploring a different area",
                "Check your inventory for useful items",
            ],
            "guide": [
                "Look for environmental clues nearby",
                "Sometimes backtracking reveals new paths",
            ],
            "solution": [
                "The objective is marked on your map",
                "Try interacting with the highlighted object",
            ],
        }
        return generic.get(tier, ["Keep going!"])
