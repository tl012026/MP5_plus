"""Visit counts on a 1-block grid, plus a heading-aligned crop.

Position is odometry from the environment's position field.
"""

from __future__ import annotations

import math


def _basis(yaw_deg: float):
    yaw = math.radians(yaw_deg)
    forward = (math.sin(yaw), math.cos(yaw))
    right = (math.cos(yaw), -math.sin(yaw))
    return forward, right


class Memory:
    def __init__(self, size: int = 15, recent_k: int = 8):
        self.size = int(size)
        self.recent_k = int(recent_k)
        self.visits: dict[tuple[int, int], int] = {}
        self.recent_actions: list[str] = []
        self.recent_matches: list[dict] = []

    def reset(self) -> None:
        self.visits.clear()
        self.recent_actions.clear()
        self.recent_matches.clear()

    def observe(self, pos, yaw: float, action: str | None = None, match: dict | None = None) -> None:
        if pos is not None:
            cell = (int(math.floor(pos[0])), int(math.floor(pos[2] if len(pos) > 2 else pos[1])))
            self.visits[cell] = self.visits.get(cell, 0) + 1
        if action:
            self.recent_actions.append(action)
            self.recent_actions = self.recent_actions[-self.recent_k:]
        if match is not None:
            self.recent_matches.append(match)
            self.recent_matches = self.recent_matches[-self.recent_k:]

    def local_grid(self, pos, yaw: float) -> list[list[float]]:
        """Rows point backward to forward (forward is the last rows' opposite: row 0 is ahead)."""
        size = self.size
        half = size // 2
        grid = [[0.0 for _ in range(size)] for _ in range(size)]
        if pos is None:
            return grid
        ax = float(pos[0])
        az = float(pos[2] if len(pos) > 2 else pos[1])
        forward, right = _basis(yaw)
        for (cx, cz), count in self.visits.items():
            dx = (cx + 0.5) - ax
            dz = (cz + 0.5) - az
            ahead = dx * forward[0] + dz * forward[1]
            side = dx * right[0] + dz * right[1]
            col = int(round(side)) + half
            row = half - int(round(ahead))
            if 0 <= row < size and 0 <= col < size:
                grid[row][col] += float(count)
        return grid

    def to_dict(self, pos, yaw: float) -> dict:
        return {
            "local_grid": self.local_grid(pos, yaw),
            "recent_actions": list(self.recent_actions),
            "recent_matches": list(self.recent_matches),
        }
