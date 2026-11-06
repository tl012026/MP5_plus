"""Rule policy. Turn toward a visible target, leave when conditions fail, else cover the least-visited sector."""

from __future__ import annotations

from lfnav.policy.base import Policy
from lfnav.types import PolicyInput, PolicyOutput


def _sector_counts(grid: list[list[float]]) -> dict[str, float]:
    if not grid:
        return {"left": 0.0, "center": 0.0, "right": 0.0}
    size = len(grid)
    half = size // 2
    counts = {"left": 0.0, "center": 0.0, "right": 0.0}
    for row in range(half):
        for col, value in enumerate(grid[row]):
            if col < half:
                counts["left"] += value
            elif col > half:
                counts["right"] += value
            else:
                counts["center"] += value
    return counts


class RuleBased(Policy):
    name = "c1"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def act(self, policy_input: PolicyInput) -> PolicyOutput:
        match = policy_input.match or {}
        if match.get("status") == "found":
            return PolicyOutput([self._steer(match.get("target_direction"))])
        satisfied = match.get("satisfied") or {}
        if satisfied.get("target") and match.get("missing"):
            return PolicyOutput(["turn_left_45", "forward"])
        grid = (policy_input.memory or {}).get("local_grid") or []
        counts = _sector_counts(grid)
        direction = min(counts, key=lambda name: (counts[name], name))
        return PolicyOutput([self._steer(direction)])

    @staticmethod
    def _steer(direction: str | None) -> str:
        if direction == "left":
            return "turn_left_15"
        if direction == "right":
            return "turn_right_15"
        return "forward"
