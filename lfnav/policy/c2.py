"""Behavior cloning over a short action chunk. Weights stay in numpy unless torch is present.

DAgger is a config switch: run the learner, ask the expert for the action, and append those rows.
"""

from __future__ import annotations

import numpy as np

from lfnav.policy.base import Policy
from lfnav.types import PolicyInput, PolicyOutput

ACTIONS = ["forward", "turn_left_15", "turn_right_15", "turn_left_45", "turn_right_45", "look_reset"]


def featurize(policy_input: PolicyInput, dim: int = 16) -> np.ndarray:
    """Fixed-width input. Unseen targets are zeros in [0:3] plus visible=0 at [3]."""
    vector = np.zeros(dim, dtype=np.float64)
    match = policy_input.match or {}
    direction = match.get("target_direction")
    if direction == "left":
        vector[0] = 1.0
    elif direction == "center":
        vector[1] = 1.0
    elif direction == "right":
        vector[2] = 1.0
    vector[3] = 1.0 if direction is not None else 0.0
    vector[4] = 1.0 if match.get("status") == "found" else 0.0
    missing = match.get("missing") or []
    vector[5] = min(len(missing), 4) / 4.0
    actions = policy_input.recent_actions or []
    for i, name in enumerate(actions[-4:]):
        if name in ACTIONS:
            vector[6 + i] = ACTIONS.index(name) / max(len(ACTIONS) - 1, 1)
    grid = (policy_input.memory or {}).get("local_grid") or []
    if grid:
        flat = np.asarray(grid, dtype=np.float64).ravel()
        vector[10] = float(flat.mean()) if flat.size else 0.0
    feats = policy_input.frame_features
    if isinstance(feats, dict) and feats.get("cls") is not None:
        cls = np.asarray(feats["cls"], dtype=np.float64).ravel()
        vector[11] = float(cls.mean()) if cls.size else 0.0
    return vector


class BehaviorCloning(Policy):
    name = "c2"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        pol = cfg.get("policy") or {}
        self.horizon = int(pol.get("chunk", 4))
        self.dim = 16
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self.weight = rng.normal(scale=0.01, size=(self.horizon, len(ACTIONS), self.dim))
        self.bias = np.zeros((self.horizon, len(ACTIONS)))
        self.dagger = bool(pol.get("dagger", False))

    def act(self, policy_input: PolicyInput) -> PolicyOutput:
        scores = self._scores(featurize(policy_input, self.dim))
        chunk = [ACTIONS[int(np.argmax(row))] for row in scores]
        return PolicyOutput(chunk)

    def _scores(self, vector: np.ndarray) -> np.ndarray:
        return np.einsum("had,d->ha", self.weight, vector) + self.bias

    def loss(self, vector: np.ndarray, labels: list[int]) -> float:
        scores = self._scores(vector)
        total = 0.0
        for step, label in enumerate(labels[: self.horizon]):
            shifted = scores[step] - scores[step].max()
            probs = np.exp(shifted)
            probs /= probs.sum()
            total += -float(np.log(probs[int(label)] + 1e-8))
        return total / max(len(labels[: self.horizon]), 1)

    def train_step(self, vector: np.ndarray, labels: list[int], lr: float = 0.05) -> float:
        before = self.loss(vector, labels)
        scores = self._scores(vector)
        for step, label in enumerate(labels[: self.horizon]):
            shifted = scores[step] - scores[step].max()
            probs = np.exp(shifted)
            probs /= probs.sum()
            grad = probs
            grad[int(label)] -= 1.0
            self.weight[step] -= lr * grad[:, None] * vector[None, :]
            self.bias[step] -= lr * grad
        return before
