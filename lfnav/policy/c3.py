"""C3 diffusion policy. The interface matches C2; the model is not implemented."""

from __future__ import annotations

from lfnav.policy.base import Policy
from lfnav.types import PolicyInput, PolicyOutput


class DiffusionPolicy(Policy):
    name = "c3"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def act(self, policy_input: PolicyInput) -> PolicyOutput:
        raise NotImplementedError(
            "C3 diffusion waypoints are reserved. Train and run C0, C1, or C2."
        )
