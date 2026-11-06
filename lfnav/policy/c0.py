"""Random-walk exploration: walk forward, and turn when the controller reports a block."""

from __future__ import annotations

from lfnav.policy.base import Policy
from lfnav.types import PolicyInput, PolicyOutput


class RandomWalk(Policy):
    name = "c0"

    def __init__(self, cfg: dict | None = None):
        self._left = True

    def reset(self) -> None:
        self._left = True

    def act(self, policy_input: PolicyInput) -> PolicyOutput:
        memory = policy_input.memory or {}
        if memory.get("stuck") or memory.get("blocked"):
            turn = "turn_left_45" if self._left else "turn_right_45"
            self._left = not self._left
            return PolicyOutput([turn])
        return PolicyOutput(["forward"])
