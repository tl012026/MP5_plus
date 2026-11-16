"""A toy world that speaks the MineRL 1.0 observation/action dialect. No Java, no Minecraft.

Flat ground, a few trees, a first-person picture painted with numpy. It exists only so
the MineDreamer performer loop, the instruction mapping, and the result file can be
exercised before any real client is started. Numbers are coarse Minecraft values:
walking 4.3 blocks/s (0.216 per 20 Hz tick), log by hand 3 s (60 ticks), dirt 0.75 s.
"""

from __future__ import annotations

import math
import random

import numpy as np

SKY = (135, 206, 235)
GRASS = (95, 159, 53)
TRUNK = (102, 76, 40)
LEAVES = (48, 110, 38)
DIRT = (134, 96, 67)

WALK_PER_TICK = 0.216
REACH = 3.5
LOG_TICKS = 60
DIRT_TICKS = 15
HALF_FOV = 45.0


def _wrap(deg: float) -> float:
    deg = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if deg == -180.0 else deg


class FakeMinecraft:
    name = "fake_minecraft"

    def __init__(self, seed: int = 0, n_trees: int = 3, size_hw: tuple[int, int] = (160, 256),
                 inventory: dict | None = None):
        self.seed = int(seed)
        self.start_inventory = dict(inventory or {})
        self.n_trees = int(n_trees)
        self.h, self.w = int(size_hw[0]), int(size_hw[1])
        self.reset()

    # ---- env API -------------------------------------------------------
    def reset(self, **_kwargs) -> dict:
        rng = random.Random(self.seed)
        self.pos = [0.0, 64.0, 0.0]
        self.yaw = rng.uniform(-180.0, 180.0)
        self.pitch = 0.0
        self.trees = []
        for _ in range(self.n_trees):
            bearing = math.radians(rng.uniform(-180.0, 180.0))
            dist = rng.uniform(6.0, 12.0)
            self.trees.append({"x": -math.sin(bearing) * dist, "z": math.cos(bearing) * dist, "logs": 4})
        self.inventory: dict[str, int] = dict(self.start_inventory)
        self.commands: list[str] = []
        self.time, self.weather = 6000, "clear"
        self.steps = 0
        self._attack_ticks = 0
        self._dig_ticks = 0
        self._obs = self._make_obs()
        return self._obs

    def step(self, action: dict):
        cam = action.get("camera") or [0.0, 0.0]
        self.pitch = max(-90.0, min(90.0, self.pitch + float(cam[0])))
        self.yaw = _wrap(self.yaw + float(cam[1]))
        if action.get("forward") and not action.get("attack"):
            self._walk(WALK_PER_TICK)
        if action.get("back"):
            self._walk(-WALK_PER_TICK)
        event = None
        if action.get("attack"):
            event = self._attack()
        else:
            self._attack_ticks = self._dig_ticks = 0
        self.steps += 1
        self._obs = self._make_obs()
        return self._obs, 0.0, False, {"event": event}

    def observation(self) -> dict:
        return self._obs

    def execute_cmd(self, cmd: str):
        """The few chat commands mc116 sends: /give, /clear, /time, /weather, /gamerule, summon, setblock."""
        parts = cmd.split()
        self.commands.append(cmd)
        if parts[0] == "/give":
            item, n = parts[2].split(":")[-1], int(parts[3]) if len(parts) > 3 else 1
            self.inventory[item] = self.inventory.get(item, 0) + n
        elif parts[0] == "/clear":
            item, n = parts[2].split(":")[-1], int(parts[3]) if len(parts) > 3 else 10 ** 9
            left = self.inventory.get(item, 0) - n
            if left > 0:
                self.inventory[item] = left
            else:
                self.inventory.pop(item, None)
        elif parts[0] == "/time" and parts[1] == "set":
            self.time = int(parts[2])
        elif parts[0] == "/weather":
            self.weather = parts[1]
        return None

    def settle(self, n: int = 1):
        for _ in range(n):
            self._obs = self._make_obs()
        return self._obs

    def close(self):
        pass

    # ---- ground truth for done-conditions (the performer's "perception oracle") ----
    def visible(self, obj: str) -> bool:
        if obj not in ("wood", "tree", "log"):
            return False
        return any(abs(rel) <= HALF_FOV and dist <= 24.0 for rel, dist, _t in self._trees_rel())

    def distance(self, obj: str):
        if obj not in ("wood", "tree", "log"):
            return None
        dists = [dist for _rel, dist, _t in self._trees_rel()]
        return min(dists) if dists else None

    # ---- internals -----------------------------------------------------
    def _trees_rel(self):
        out = []
        for tree in self.trees:
            if tree["logs"] <= 0:
                continue
            dx, dz = tree["x"] - self.pos[0], tree["z"] - self.pos[2]
            dist = math.hypot(dx, dz)
            bearing = math.degrees(math.atan2(-dx, dz))
            out.append((_wrap(bearing - self.yaw), dist, tree))
        return out

    def _walk(self, step: float):
        yaw = math.radians(self.yaw)
        nx = self.pos[0] - math.sin(yaw) * step
        nz = self.pos[2] + math.cos(yaw) * step
        for tree in self.trees:
            if tree["logs"] > 0 and math.hypot(tree["x"] - nx, tree["z"] - nz) < 0.8:
                return
        self.pos[0], self.pos[2] = nx, nz

    def _attack(self):
        if self.pitch >= 60.0:
            self._attack_ticks = 0
            self._dig_ticks += 1
            if self._dig_ticks >= DIRT_TICKS:
                self._dig_ticks = 0
                self.inventory["dirt"] = self.inventory.get("dirt", 0) + 1
                self.pos[1] -= 1.0
                return "mined:dirt"
            return None
        self._dig_ticks = 0
        if -30.0 <= self.pitch <= 30.0:
            for rel, dist, tree in self._trees_rel():
                if dist <= REACH and abs(rel) <= 12.0:
                    self._attack_ticks += 1
                    if self._attack_ticks >= LOG_TICKS:
                        self._attack_ticks = 0
                        tree["logs"] -= 1
                        self.inventory["oak_log"] = self.inventory.get("oak_log", 0) + 1
                        return "mined:oak_log"
                    return None
        self._attack_ticks = 0
        return None

    def _make_obs(self) -> dict:
        return {
            "pov": self._render(),
            "inventory": dict(self.inventory),
            "location_stats": {"pos": list(self.pos), "yaw": self.yaw, "pitch": self.pitch},
            "life_stats": {"life": 20.0},
        }

    def _render(self) -> np.ndarray:
        h, w = self.h, self.w
        img = np.empty((h, w, 3), dtype=np.uint8)
        horizon = int(h / 2 - (self.pitch / HALF_FOV) * (h / 2))
        top = max(0, min(h, horizon))
        img[:top] = SKY
        img[top:] = GRASS if self.pitch < 60.0 else DIRT
        for rel, dist, _tree in sorted(self._trees_rel(), key=lambda r: -r[1]):
            if abs(rel) > HALF_FOV or dist < 0.3:
                continue
            col = int(w / 2 + rel / HALF_FOV * (w / 2))
            half = max(1, int(w * 0.66 / dist) // 2)
            t0 = horizon - int(h * 1.2 / dist)
            t1 = horizon + int(h * 0.8 / dist)
            x0, x1 = max(0, col - half), min(w, col + half)
            if x0 < x1:
                img[max(0, t0):max(0, min(h, t1)), x0:x1] = TRUNK
                l0 = max(0, t0 - int(h * 1.0 / dist))
                lx0, lx1 = max(0, col - 3 * half), min(w, col + 3 * half)
                img[l0:max(0, min(h, t0)), lx0:lx1] = LEAVES
        return img
