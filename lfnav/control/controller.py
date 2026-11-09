"""Turn named actions into environment steps.

The voxel crop (MineDojo voxel_radius, default 5 → 11^3) is a local collision
sensor for this controller only. Policies never read it. Jumping, stuck
recovery, and refusing water or lava live here. Waypoints are yaw-then-forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lfnav.obsutil import is_empty, position_of, scalar, voxels_of, yaw_of

DEFAULT_PASSABLE = (
    "air",
    "tallgrass",
    "double_plant",
    "red_flower",
    "yellow_flower",
    "deadbush",
    "snow_layer",
    "reeds",
)
DEFAULT_HAZARD = (
    "water",
    "flowing_water",
    "lava",
    "flowing_lava",
)


def _front_delta(yaw: float) -> tuple[int, int]:
    """Integer step in voxel x/z for the current look direction.

    lfnav yaw 0 looks +z; yaw increases when turning right (Minecraft).
    0° → (dx, dz)=(0, +1); 90° → (+1, 0); 180° → (0, -1); 270° → (-1, 0).
    Confirm in the MineDojo image with lfnav/control/check_heading.py
    (heading check 2026-11-08).
    """
    y = float(yaw) % 360.0
    if y < 45.0 or y >= 315.0:
        return 0, 1
    if y < 135.0:
        return 1, 0
    if y < 225.0:
        return 0, -1
    return -1, 0


def block_key(value) -> str:
    raw = scalar(value)
    if raw is None:
        return "air"
    text = str(raw).strip().lower().replace(" ", "_")
    return text or "air"


def _nameset(cfg: dict, key: str, default) -> set[str]:
    actions = cfg.get("actions") or {}
    raw = actions.get(key)
    if raw is None:
        raw = cfg.get(key)
    if not raw:
        raw = default
    return {block_key(name) for name in raw}


def choose_forward(voxels, yaw: float, cfg: dict | None = None) -> str:
    """Pick forward / jump_forward / turn when the cell ahead is blocked or a hazard."""
    cfg = cfg or {}
    actions = cfg.get("actions") or {}
    if voxels is None or is_empty(voxels):
        return "forward"
    arr = np.asarray(voxels)
    if arr.ndim != 3:
        return "forward"
    c = int(arr.shape[0] // 2)
    dx, dz = _front_delta(yaw)
    ix, iy_foot, iz = c + dx, c, c + dz
    if not (0 <= ix < arr.shape[0] and 0 <= iz < arr.shape[2]):
        return "forward"
    foot = block_key(arr[ix, iy_foot, iz])
    head = block_key(arr[ix, iy_foot + 1, iz]) if iy_foot + 1 < arr.shape[1] else "air"
    below = block_key(arr[ix, iy_foot - 1, iz]) if iy_foot - 1 >= 0 else "air"
    below2 = block_key(arr[ix, iy_foot - 2, iz]) if iy_foot - 2 >= 0 else "air"
    passable = _nameset(cfg, "passable_blocks", DEFAULT_PASSABLE)
    hazard = _nameset(cfg, "hazard_blocks", DEFAULT_HAZARD)
    can_jump = "jump_forward" in (actions.get("vectors") or {})
    if foot in hazard or head in hazard:
        return "turn_right_45"
    foot_solid = foot not in passable
    head_ok = head in passable
    if foot_solid and not head_ok:
        return "turn_left_45"
    if foot_solid and head_ok:
        return "jump_forward" if can_jump else "forward"
    avoid_cliffs = bool(actions.get("avoid_cliffs", cfg.get("avoid_cliffs", True)))
    if avoid_cliffs and (not foot_solid) and below in passable and below2 in passable:
        return "turn_right_45"
    return "forward"


@dataclass
class ControlResult:
    env_steps: int
    stuck: bool
    jumped: bool
    obs: dict

    def to_dict(self) -> dict:
        return {"env_steps": self.env_steps, "stuck": self.stuck, "jumped": self.jumped}


class Controller:
    def __init__(self, env, cfg: dict):
        self.env = env
        self.cfg = cfg
        self.actions = cfg.get("actions") or {}
        self.keys = cfg.get("obs_keys") or {}
        self.stuck_threshold = int(self.actions.get("stuck_threshold", 2))
        self._unstuck_toggle = False
        self._last_pos = None
        self._still = 0

    def execute(self, action_name: str) -> ControlResult:
        steps = 0
        jumped = False
        stuck = False
        name = action_name
        if name == "forward":
            name = self._forward_name()
        obs, info = self._one(name)
        steps += 1
        jumped = bool(info.get("jumped"))
        if info.get("blocked") or info.get("stuck") or self._still >= self.stuck_threshold:
            stuck = True
            obs, _info = self._one(self._unstuck())
            steps += 1
        return ControlResult(steps, stuck, jumped, obs)

    def execute_waypoints(self, waypoints) -> ControlResult:
        obs = self.env.observation() if hasattr(self.env, "observation") else {}
        steps = 0
        jumped = False
        stuck = False
        for waypoint in waypoints:
            dyaw, blocks = waypoint[0], waypoint[1]
            turn = "turn_left_15" if dyaw < 0 else "turn_right_15"
            n_turns = int(round(abs(float(dyaw)) / 15.0))
            for _ in range(n_turns):
                result = self.execute(turn)
                obs = result.obs
                steps += result.env_steps
                stuck = stuck or result.stuck
            for _ in range(int(round(float(blocks)))):
                result = self.execute("forward")
                obs = result.obs
                steps += result.env_steps
                jumped = jumped or result.jumped
                stuck = stuck or result.stuck
        return ControlResult(steps, stuck, jumped, obs)

    def _one(self, name: str):
        vectors = self.actions.get("vectors") or {}
        action = vectors.get(name, name)
        before = position_of(self.env.observation(), self.cfg) if hasattr(self.env, "observation") else None
        obs, _reward, _done, info = self.env.step(action)
        after = position_of(obs, self.cfg)
        if before is not None and after is not None and before[0] == after[0] and before[-1] == after[-1] and name in ("forward", "jump_forward"):
            self._still += 1
        else:
            self._still = 0
        info = dict(info or {})
        info["stuck"] = info.get("stuck") or self._still >= self.stuck_threshold
        return obs, info

    def _forward_name(self) -> str:
        obs = self.env.observation()
        voxels = voxels_of(obs, self.cfg)
        if voxels is None:
            return "forward"
        yaw = yaw_of(obs, self.cfg)
        return choose_forward(voxels, yaw, self.cfg)

    def _unstuck(self) -> str:
        self._unstuck_toggle = not self._unstuck_toggle
        return "turn_left_45" if self._unstuck_toggle else "turn_right_45"
