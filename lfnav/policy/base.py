"""Shared policy interface."""

from __future__ import annotations

from lfnav.types import PolicyInput, PolicyOutput


class Policy:
    name = "base"

    def reset(self) -> None:
        return None

    def act(self, policy_input: PolicyInput) -> PolicyOutput:
        raise NotImplementedError
