"""Turn named actions into environment steps.

The voxel crop (MineDojo voxel_radius, default 5 → 11^3) is a local collision
sensor for this controller only. Policies never read it. Jumping, stuck
recovery, and refusing water or lava live here. Waypoints are yaw-then-forward.
"""

from __future__ import annotations

import math
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
DEFAULT_HARD_HAZARD = (
    "lava",
    "flowing_lava",
    "fire",
    "cactus",
)
DEFAULT_WATER = (
    "water",
    "flowing_water",
)

ESCAPE_SEQUENCES = (
    ("jump_forward",) * 8,
    ("backward",) * 4 + ("turn_right_45", "turn_right_45"),
    ("turn_right_45",) * 4 + ("jump_forward",) * 8,
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


def _shallow_water_ok(cfg: dict) -> bool:
    actions = cfg.get("actions") or {}
    return bool(actions.get("shallow_water_ok", cfg.get("shallow_water_ok", True)))


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
    here = block_key(arr[c, iy_foot, c])
    passable = _nameset(cfg, "passable_blocks", DEFAULT_PASSABLE)
    hazard = _nameset(cfg, "hazard_blocks", DEFAULT_HAZARD)
    can_jump = "jump_forward" in (actions.get("vectors") or {})
    water = _nameset(cfg, "water_blocks", DEFAULT_WATER)
    if _shallow_water_ok(cfg):
        hard = _nameset(cfg, "hard_hazard_blocks", DEFAULT_HARD_HAZARD)
        if foot in hard or head in hard:
            return "turn_right_45"
        ahead_water = foot in water or head in water
        deep = (foot in water and below in water) or head in water
        if ahead_water and deep:
            return "turn_right_45"
        if ahead_water or here in water:
            return "jump_forward" if can_jump else "forward"
    elif foot in hazard or head in hazard:
        return "turn_right_45"
    foot_solid = foot not in passable and foot not in water
    head_ok = head in passable or head in water
    if foot_solid and not head_ok:
        return "turn_left_45"
    if foot_solid and head_ok:
        return "jump_forward" if can_jump else "forward"
    avoid_cliffs = bool(actions.get("avoid_cliffs", cfg.get("avoid_cliffs", True)))
    if avoid_cliffs and (not foot_solid) and foot not in water and below in passable and below2 in passable:
        return "turn_right_45"
    return "forward"


@dataclass
class ControlResult:
    env_steps: int
    stuck: bool
    jumped: bool
    obs: dict
    escape_k: int | None = None

    def to_dict(self) -> dict:
        out = {"env_steps": self.env_steps, "stuck": self.stuck, "jumped": self.jumped}
        if self.escape_k is not None:
            out["escape_k"] = self.escape_k
        return out


class Controller:
    def __init__(self, env, cfg: dict):
        self.env = env
        self.cfg = cfg
        self.actions = cfg.get("actions") or {}
        self.keys = cfg.get("obs_keys") or {}
        self.stuck_threshold = int(self.actions.get("stuck_threshold", 2))
        self.displacement_stuck = bool(self.actions.get("displacement_stuck", True))
        self.stuck_window = int(self.actions.get("stuck_window", 60))
        self.stuck_displacement = float(self.actions.get("stuck_displacement", 1.0))
        self._unstuck_toggle = False
        self._last_pos = None
        self._still = 0
        self._pos_hist: list[tuple[float, float]] = []
        self.n_stuck = 0
        self.n_escape = 0
        self.stuck_steps_total = 0
        self._escaping = False

    def execute(self, action_name: str) -> ControlResult:
        if self.displacement_stuck and not self._escaping and self._window_stuck():
            return self._run_escape()
        steps = 0
        jumped = False
        stuck = False
        name = action_name
        if name == "forward":
            name = self._forward_name()
        obs, info = self._one(name)
        steps += 1
        jumped = bool(info.get("jumped"))
        if self.displacement_stuck:
            if not self._escaping and self._window_stuck():
                extra = self._run_escape()
                return ControlResult(
                    steps + extra.env_steps,
                    True,
                    jumped or extra.jumped,
                    extra.obs,
                    escape_k=extra.escape_k,
                )
            return ControlResult(steps, False, jumped, obs)
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
        escape_k = None
        for waypoint in waypoints:
            dyaw, blocks = waypoint[0], waypoint[1]
            turn = "turn_left_15" if dyaw < 0 else "turn_right_15"
            n_turns = int(round(abs(float(dyaw)) / 15.0))
            for _ in range(n_turns):
                result = self.execute(turn)
                obs = result.obs
                steps += result.env_steps
                stuck = stuck or result.stuck
                escape_k = result.escape_k if result.escape_k is not None else escape_k
            for _ in range(int(round(float(blocks)))):
                result = self.execute("forward")
                obs = result.obs
                steps += result.env_steps
                jumped = jumped or result.jumped
                stuck = stuck or result.stuck
                escape_k = result.escape_k if result.escape_k is not None else escape_k
        return ControlResult(steps, stuck, jumped, obs, escape_k=escape_k)

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
        if after is not None:
            self._pos_hist.append((float(after[0]), float(after[-1])))
            if len(self._pos_hist) > self.stuck_window:
                self._pos_hist = self._pos_hist[-self.stuck_window:]
        info = dict(info or {})
        info["stuck"] = info.get("stuck") or self._still >= self.stuck_threshold or self._window_stuck()
        return obs, info

    def _window_stuck(self) -> bool:
        if not self.displacement_stuck:
            return False
        if len(self._pos_hist) < self.stuck_window:
            return False
        x0, z0 = self._pos_hist[0]
        x1, z1 = self._pos_hist[-1]
        return math.hypot(x1 - x0, z1 - z0) < self.stuck_displacement

    def _run_escape(self) -> ControlResult:
        self._escaping = True
        self.n_stuck += 1
        origin = self._pos_hist[-1] if self._pos_hist else None
        steps = 0
        jumped = False
        obs = self.env.observation() if hasattr(self.env, "observation") else {}
        used = None
        try:
            for k, seq in enumerate(ESCAPE_SEQUENCES, start=1):
                self.n_escape += 1
                used = k
                for name in seq:
                    obs, info = self._one(name)
                    steps += 1
                    self.stuck_steps_total += 1
                    jumped = jumped or bool(info.get("jumped"))
                after = position_of(obs, self.cfg)
                if origin is not None and after is not None:
                    if math.hypot(after[0] - origin[0], after[-1] - origin[1]) >= self.stuck_displacement:
                        break
                if not self._window_stuck():
                    break
        finally:
            self._escaping = False
        return ControlResult(steps, True, jumped, obs, escape_k=used)

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
