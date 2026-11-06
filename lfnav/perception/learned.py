"""Learned perception. The frozen encoder runs once per frame and is shared with C2."""

from __future__ import annotations

import numpy as np

from lfnav.aggregate import aggregate_object
from lfnav.perception.encoder import FakeEncoder, load_encoder
from lfnav.types import SceneState


class LearnedPerception:
    name = "learned"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        perc = cfg.get("perception") or {}
        name = perc.get("encoder", "fake")
        if cfg.get("dry_run"):
            name = "fake"
        self.encoder = load_encoder(name, cfg)
        self._cache = None
        self.checkpoint = perc.get("checkpoint")

    def features(self, frame) -> dict:
        rgb = frame.get("rgb") if isinstance(frame, dict) and "rgb" in frame else frame
        encoded = self.encoder(rgb)
        self._cache = encoded
        return encoded

    def perceive(self, frame) -> SceneState:
        feats = self._cache if self._cache is not None else self.features(frame)
        labels = self.cfg.get("labels") or {}
        objects = list(labels.get("objects") or [])
        tokens = np.asarray(feats["tokens"])
        gh, gw = tokens.shape[:2]
        beliefs = {}
        for name in objects:
            # Untrained head: a fixed slice of the token is a probability in [0, 1].
            raw = tokens[:, :, 0]
            present = np.clip(raw, 0.0, 1.0)
            dist = [[{"mid": 1.0} for _ in range(gw)] for _ in range(gh)]
            beliefs[name] = aggregate_object(present.tolist(), dist)
        return SceneState(objects=beliefs)


def train_step_numpy(logits, target) -> float:
    """One BCE-with-logits step on numpy arrays, used when torch is absent."""
    preds = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    target = np.asarray(target, dtype=np.float64)
    eps = 1e-6
    loss = -(target * np.log(preds + eps) + (1.0 - target) * np.log(1.0 - preds + eps))
    return float(loss.mean())
