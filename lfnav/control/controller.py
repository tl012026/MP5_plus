"""Turn named actions into environment steps.

The 3x3x3 voxel crop is a local collision sensor for this controller only.
Policies never read it. Jumping, stuck recovery, and refusing water or lava
live here. Waypoints are yaw-then-forward, the stand-in for a later C3 controller.
"""

from __future__ import annotations

from dataclasses import dataclass

from lfnav.obsutil import lookup, position_of, yaw_of


@dataclass
class ControlResult:
    env_steps: int
    stuck: bool
    jumped: bool
    obs: dict

    def to_dict(self) -> dict:
        return {"env_steps": self.env_steps, "stuck": self.stuck, "jumped": self.jumped}


def _front_delta(yaw: float) -> tuple[int, int]:
    y = yaw % 360.0
    if y < 45.0 or y >= 315.0:
        return 0, 1
    if y < 135.0:
        return 1, 0
    if y < 225.0:
        return 0, -1
    return -1, 0


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
        voxels = lookup(obs, self.keys.get("voxels"))
        yaw = yaw_of(obs, self.cfg)
        if not voxels:
            return "forward"
        dx, dz = _front_delta(yaw)
        # Voxels are indexed [x+1][y+1][z+1] over the {-1,0,1} neighborhood.
        ix, iz = dx + 1, dz + 1
        ground = voxels[ix][1][iz]
        above = voxels[ix][2][iz]
        if ground in ("water", "lava"):
            return "turn_right_45"
        if ground not in ("air", None, "grass", "sand") and above not in ("air", None):
            return "turn_left_45"
        if ground not in ("air", None, "grass", "sand"):
            return "jump_forward" if "jump_forward" in (self.actions.get("vectors") or {}) else "forward"
        return "forward"

    def _unstuck(self) -> str:
        self._unstuck_toggle = not self._unstuck_toggle
        return "turn_left_45" if self._unstuck_toggle else "turn_right_45"
