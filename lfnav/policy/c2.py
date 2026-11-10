"""C2 behavior cloning: GT scene features → K discrete action heads (PyTorch MLP).

The numpy linear placeholder is lfnav.policy.c2_linear. DINOv2 inputs come later.
"""

from __future__ import annotations

import numpy as np

from lfnav.policy.base import Policy
from lfnav.policy.c2_data import FeatureSpec, encode_policy_input
from lfnav.policy.c2_linear import ACTIONS, featurize
from lfnav.policy.c2_linear import BehaviorCloning as LinearBehaviorCloning
from lfnav.types import PolicyInput, PolicyOutput

c2_linear = LinearBehaviorCloning


def _visible(match: dict) -> bool:
    if (match or {}).get("target_direction") is not None:
        return True
    return (match or {}).get("status") == "found"


def ChunkMLP(dim: int, n_actions: int, horizon: int, hidden: int = 256):
    from torch import nn

    class _ChunkMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.horizon = int(horizon)
            self.n_actions = int(n_actions)
            self.fc1 = nn.Linear(int(dim), int(hidden))
            self.fc2 = nn.Linear(int(hidden), int(hidden))
            self.heads = nn.ModuleList(nn.Linear(int(hidden), int(n_actions)) for _ in range(int(horizon)))

        def forward(self, x):
            h = nn.functional.relu(self.fc1(x))
            h = nn.functional.relu(self.fc2(h))
            return [head(h) for head in self.heads]

    return _ChunkMLP()


class BehaviorCloning(Policy):
    name = "c2"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        pol = self.cfg.get("policy") or {}
        c2 = self.cfg.get("c2") or {}
        self.spec = FeatureSpec.from_cfg(self.cfg)
        self.horizon = int(c2.get("chunk") or pol.get("chunk") or self.spec.horizon)
        self.spec.horizon = self.horizon
        self.hidden = int(c2.get("hidden") or 256)
        self.dagger = bool(pol.get("dagger", False))
        self._fallback = None
        self.model = None
        self._torch = None
        try:
            import torch
        except ImportError:
            self._fallback = LinearBehaviorCloning(self.cfg)
        else:
            self._torch = torch
            self.model = ChunkMLP(self.spec.dim, len(self.spec.actions), self.horizon, self.hidden)
            ckpt = c2.get("checkpoint")
            if ckpt:
                payload = torch.load(ckpt, map_location="cpu")
                state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
                self.model.load_state_dict(state)
            self.model.eval()
        self.reset()

    def reset(self) -> None:
        self.last_seen = None
        if self._fallback is not None:
            self._fallback.reset()

    def n_parameters(self) -> int:
        if self._fallback is not None:
            return int(self._fallback.weight.size + self._fallback.bias.size)
        return int(sum(p.numel() for p in self.model.parameters()))

    def act(self, policy_input: PolicyInput) -> PolicyOutput:
        if self._fallback is not None:
            return self._fallback.act(policy_input)
        match = policy_input.match or {}
        env_step = int((policy_input.memory or {}).get("env_steps") or 0)
        if _visible(match):
            self.last_seen = {
                "step": env_step,
                "direction": match.get("target_direction"),
                "distance": match.get("target_distance"),
            }
        vector = encode_policy_input(policy_input, self.spec, last_seen=self.last_seen)
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(vector, dtype=np.float32)).unsqueeze(0)
            logits = self.model(x)
            idx = [int(row.argmax(dim=-1).item()) for row in logits]
        chunk = [self.spec.actions[i] for i in idx]
        return PolicyOutput(chunk)
