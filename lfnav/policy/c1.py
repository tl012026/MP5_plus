"""Rule policy. Turn toward a visible target, leave when conditions fail, else cover the least-visited sector."""

from __future__ import annotations

import math
import random

from lfnav.policy.base import Policy
from lfnav.types import PolicyInput, PolicyOutput

SCAN_TURNS = 24
LOST_HORIZON = 20
N_SECTORS = 8


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


def sector8_counts(grid: list[list[float]]) -> list[float]:
    """8 yaw sectors of 45°. Index 0 is forward, 4 is behind."""
    counts = [0.0] * N_SECTORS
    if not grid:
        return counts
    size = len(grid)
    half = size // 2
    for row, line in enumerate(grid):
        for col, value in enumerate(line):
            ahead = half - row
            side = col - half
            if ahead == 0 and side == 0:
                continue
            ang = math.degrees(math.atan2(side, ahead)) % 360.0
            idx = int((ang + 22.5) // 45.0) % N_SECTORS
            counts[idx] += float(value)
    return counts


def _face_sector(k: int) -> list[str]:
    k = int(k) % N_SECTORS
    if k == 0:
        return ["forward"]
    if k <= 4:
        return ["turn_right_45"] * k + ["forward"]
    return ["turn_left_45"] * (N_SECTORS - k) + ["forward"]


def _target_visible(match: dict) -> bool:
    if match.get("target_direction") is not None:
        return True
    if match.get("status") == "found":
        return True
    satisfied = match.get("satisfied") or {}
    return bool(satisfied.get("target"))


class RuleBased(Policy):
    name = "c1"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        pol = self.cfg.get("policy") or {}
        self.opening_scan = bool(pol.get("c1_opening_scan", True))
        self.eight_sectors = bool(pol.get("c1_eight_sectors", True))
        self.mid_cap = int(pol.get("c1_mid_cap", 2))
        self.lost_horizon = int(pol.get("c1_lost_horizon", LOST_HORIZON))
        self.recover_scan = bool(pol.get("c1_recover_scan", True))
        self.rng = random.Random(int(self.cfg.get("seed", 0)))
        self.reset()

    def reset(self) -> None:
        self._scan_left = SCAN_TURNS if self.opening_scan else 0
        self._recover_scan = 0
        self._recover_back = False
        self.last_seen = None
        self.last_branch = None

    def act(self, policy_input: PolicyInput) -> PolicyOutput:
        match = policy_input.match or {}
        memory = policy_input.memory or {}
        env_steps = int(memory.get("env_steps") or 0)
        visible = _target_visible(match)
        distance = match.get("target_distance")
        direction = match.get("target_direction")
        if visible:
            self.last_seen = {
                "step": env_steps,
                "direction": direction,
                "distance": distance,
            }
            self._scan_left = 0
            self._recover_scan = 0
            self._recover_back = False
            if match.get("status") == "found":
                self.last_branch = "approach"
                return PolicyOutput(self._approach(direction, distance))
            if match.get("missing"):
                self.last_branch = "approach"
                return PolicyOutput(["turn_left_45", "forward"])
        if self._scan_left == SCAN_TURNS and env_steps > 0:
            self._scan_left = 0
        if self._scan_left > 0:
            self._scan_left -= 1
            self.last_branch = "scan"
            return PolicyOutput(["turn_right_15"])
        if self.recover_scan and self._recover_back:
            self._recover_back = False
            self._recover_scan = SCAN_TURNS
            self.last_branch = "reacquire"
            return PolicyOutput(["backward"])
        if self.recover_scan and self._recover_scan > 0:
            self._recover_scan -= 1
            self.last_branch = "reacquire"
            return PolicyOutput(["turn_right_15"])
        if (
            self.recover_scan
            and not visible
            and self.last_seen is not None
            and env_steps - int(self.last_seen.get("step") or 0) <= self.lost_horizon
            and self.last_seen.get("distance") in ("near", "mid")
        ):
            self.last_seen = None
            self._recover_scan = SCAN_TURNS
            self.last_branch = "reacquire"
            return PolicyOutput(["backward"])
        grid = memory.get("local_grid") or []
        if self.eight_sectors:
            counts = sector8_counts(grid)
            lo = min(counts)
            tied = [i for i, n in enumerate(counts) if n == lo]
            k = self.rng.choice(tied)
            self.last_branch = "explore"
            return PolicyOutput(_face_sector(k))
        counts = _sector_counts(grid)
        direction = min(counts, key=lambda name: (counts[name], name))
        self.last_branch = "explore"
        return PolicyOutput([self._steer(direction)])

    def _approach(self, direction: str | None, distance: str | None) -> list[str]:
        turn = []
        if direction == "left":
            turn = ["turn_left_15"]
        elif direction == "right":
            turn = ["turn_right_15"]
        if distance == "mid":
            return turn + ["forward"] * max(self.mid_cap, 0)
        if turn:
            return turn
        return ["forward"]

    @staticmethod
    def _steer(direction: str | None) -> str:
        if direction == "left":
            return "turn_left_15"
        if direction == "right":
            return "turn_right_15"
        return "forward"
