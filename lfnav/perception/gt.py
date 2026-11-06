"""Upper bound: the scene state is the ground truth derived from the observation."""

from __future__ import annotations

from lfnav.gt.scene_state import scene_state_from_obs
from lfnav.types import SceneState


class GTPerception:
    name = "gt"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def perceive(self, frame) -> SceneState:
        if not isinstance(frame, dict):
            raise TypeError("GTPerception expects the full observation, not a bare image")
        return scene_state_from_obs(frame, self.cfg)

    def features(self, frame):
        return None
