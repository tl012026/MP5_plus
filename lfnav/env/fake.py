"""Grid world with the same observation keys as configs/lfnav/obs_keys.yaml.

Used to test the controller, matcher and eval loop without MineDojo.
"""

from __future__ import annotations

import math

import numpy as np

from lfnav.geom import patch_grid, ray_directions
from lfnav.gt.scene_state import biome_id_for_name


def _assign(obs: dict, path, value) -> None:
    if path is None:
        return
    if isinstance(path, str):
        obs[path] = value
        return
    cur = obs
    for key in path[:-1]:
        cur = cur.setdefault(str(key), {})
    cur[str(path[-1])] = value


def _yaw_basis(yaw_deg: float):
    yaw = math.radians(yaw_deg)
    forward = (math.sin(yaw), math.cos(yaw))
    right = (math.cos(yaw), -math.sin(yaw))
    return forward, right


class FakeMineDojoEnv:
    """2D blocks. Yaw 0 faces +z. Positive yaw turns right. One forward action moves about 1 block."""

    def __init__(self, cfg: dict, layout: dict | None = None):
        self.cfg = cfg
        self.keys = cfg.get("obs_keys") or {}
        self.actions = cfg.get("actions") or {}
        self.image_hw = tuple(cfg.get("image_hw", [28, 42]))
        self.patch = int(cfg.get("patch", 14))
        self.fov_deg = float(cfg.get("fov_deg", 70))
        self.fov_axis = str(cfg.get("fov_axis", "vertical"))
        self.max_ray = float(cfg.get("aligned_ray_range", 16))
        self.wide_range = float(cfg.get("wide_ray_range", 32))
        self.wide_enabled = bool(cfg.get("wide_lidar", True))
        layout = layout or {}
        self.pos = [float(v) for v in layout.get("pos", [0.0, 64.0, 0.0])]
        self.yaw = float(layout.get("yaw", 0.0))
        self.pitch = float(layout.get("pitch", 0.0))
        self.blocks = {tuple(k): v for k, v in layout.get("blocks", {}).items()}
        self.entities = [dict(e) for e in layout.get("entities", [])]
        self.solids = {tuple(p) for p in layout.get("solids", [])}
        self.tall = {tuple(p) for p in layout.get("tall", [])}
        self.water = {tuple(p) for p in layout.get("water", [])}
        self.lava = {tuple(p) for p in layout.get("lava", [])}
        self.biome = layout.get("biome", "plains")
        self.time_name = layout.get("time", "day")
        self.weather = layout.get("weather", "clear")
        self.brightness = layout.get("brightness", "sufficient")
        self._stuck_count = 0
        self.n_h, self.n_w = patch_grid(self.image_hw[0], self.image_hw[1], self.patch)
        self._cam_rays = ray_directions(self.n_h, self.n_w, self.fov_deg, self.fov_axis)
        # Match the 2026-11-08 MineDojo probe: size-(1,) arrays except pos length 3.
        self.numpy_location_stats = bool(cfg.get("numpy_location_stats", True))

    def reset(self, **_kwargs):
        self._stuck_count = 0
        return self.observation()

    def close(self):
        self.closed = True

    def observation(self) -> dict:
        obs: dict = {}
        h, w = self.image_hw
        _assign(obs, self.keys.get("rgb", "rgb"), [[[0, 0, 0] for _ in range(w)] for _ in range(h)])
        _assign(obs, self.keys.get("position"), [self.pos[0], self.pos[1], self.pos[2]])
        _assign(obs, self.keys.get("yaw"), self.yaw)
        _assign(obs, self.keys.get("pitch"), self.pitch)
        _assign(obs, self.keys.get("voxels"), self._voxels())
        rays = self._aligned_rays()
        _assign(obs, self.keys.get("aligned_rays"), rays)
        if self.wide_enabled:
            _assign(obs, self.keys.get("wide_rays"), self._wide_rays())
        _assign(obs, self.keys.get("biome_name"), self.biome)
        _assign(obs, self.keys.get("time_name"), self.time_name)
        _assign(obs, self.keys.get("weather_name"), self.weather)
        _assign(obs, self.keys.get("brightness_name"), self.brightness)
        loc = obs.setdefault("location_stats", {})
        if isinstance(loc, dict):
            mapping = self.keys.get("biome_id_to_name") or {}
            biome_id = biome_id_for_name(self.biome, mapping)
            if biome_id is not None:
                loc["biome_id"] = biome_id
            loc["is_raining"] = str(self.weather).lower() == "rain"
            loc["sun_brightness"] = 1.0 if str(self.time_name).lower() == "day" else 0.0
            loc["sky_light_level"] = loc["sun_brightness"]
            loc["light_level"] = 15.0 if str(self.brightness).lower() == "sufficient" else 3.0
            if self.numpy_location_stats:
                obs["location_stats"] = self._numpy_location_stats(loc)
        return obs

    def _numpy_location_stats(self, loc: dict) -> dict:
        """Same dtypes as real MineDojo location_stats: int biome_id, bool rain, else float."""
        packed = {}
        for key, val in loc.items():
            if key == "pos":
                packed[key] = np.asarray(val, dtype=np.float32)
            elif key == "biome_id":
                packed[key] = np.asarray([int(val)], dtype=np.int64)
            elif key in ("is_raining", "can_see_sky"):
                packed[key] = np.asarray([bool(val)], dtype=bool)
            else:
                packed[key] = np.asarray([float(val)], dtype=np.float32)
        return packed

    def step(self, action):
        name = self._action_name(action)
        before = (self.pos[0], self.pos[2])
        jumped = False
        blocked = False
        if name in ("forward", "jump_forward"):
            jumped, blocked = self._forward()
        elif name == "turn_left_15":
            self.yaw = (self.yaw - 15.0) % 360.0
        elif name == "turn_right_15":
            self.yaw = (self.yaw + 15.0) % 360.0
        elif name == "turn_left_45":
            self.yaw = (self.yaw - 45.0) % 360.0
        elif name == "turn_right_45":
            self.yaw = (self.yaw + 45.0) % 360.0
        elif name == "look_reset":
            self.pitch = 0.0
        elif name in (None, "noop", "stop"):
            pass
        else:
            raise KeyError(f"unknown action {action!r}")
        moved = (self.pos[0], self.pos[2]) != before
        if name == "forward" and not moved:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
        obs = self.observation()
        info = {"stuck": self._stuck_count >= 2, "jumped": jumped, "blocked": blocked, "action": name}
        return obs, 0.0, False, info

    def _action_name(self, action) -> str:
        if isinstance(action, str):
            return action
        table = self.actions.get("vectors") or {}
        for name, vec in table.items():
            if list(vec) == list(action):
                return name
        raise KeyError(f"action vector is not in the configured table: {action}")

    def _cell(self, x, z):
        return (int(math.floor(x)), int(math.floor(z)))

    def _forward(self) -> tuple[bool, bool]:
        forward, _right = _yaw_basis(self.yaw)
        nxt = self._cell(self.pos[0] + forward[0], self.pos[2] + forward[1])
        if nxt in self.lava or nxt in self.water or nxt in self.tall:
            return False, True
        jumped = nxt in self.solids
        self.pos[0] += forward[0]
        self.pos[2] += forward[1]
        return jumped, False

    def _voxels(self):
        cx, cz = self._cell(self.pos[0], self.pos[2])
        grid = []
        for dx in (-1, 0, 1):
            plane = []
            for _dy in (-1, 0, 1):
                row = []
                for dz in (-1, 0, 1):
                    cell = (cx + dx, cz + dz)
                    if _dy > 0 and cell in self.tall:
                        name = "solid"
                    elif _dy == 0 and cell in self.lava:
                        name = "lava"
                    elif _dy == 0 and cell in self.water:
                        name = "water"
                    elif _dy == 0 and cell in self.solids:
                        name = "solid"
                    elif _dy == 0:
                        name = self.blocks.get(cell, "air")
                    else:
                        name = "air"
                    row.append(name)
                plane.append(row)
            grid.append(plane)
        return grid

    def _aligned_rays(self) -> list[dict]:
        forward, right = _yaw_basis(self.yaw)
        hits = []
        for i, row in enumerate(self._cam_rays):
            for j, (rx, ry, rz) in enumerate(row):
                wx = rz * forward[0] + rx * right[0]
                wz = rz * forward[1] + rx * right[1]
                hit = self._march(wx, wz, self.max_ray)
                if hit is None:
                    continue
                label, dist = hit
                hits.append({"row": i, "col": j, "label": label, "distance": dist, "pitch": ry})
        return hits

    def _wide_rays(self) -> list[dict]:
        hits = []
        for deg in range(0, 360, 15):
            forward, _right = _yaw_basis(self.yaw + deg)
            hit = self._march(forward[0], forward[1], self.wide_range)
            if hit is None:
                continue
            label, dist = hit
            hits.append({"yaw_offset": deg, "label": label, "distance": dist})
        return hits

    def _march(self, dx, dz, limit):
        norm = math.hypot(dx, dz)
        if norm < 1e-6:
            return None
        dx, dz = dx / norm, dz / norm
        best = None
        for ent in self.entities:
            ex = float(ent["x"]) - self.pos[0]
            ez = float(ent["z"]) - self.pos[2]
            t = ex * dx + ez * dz
            if t <= 0.2 or t > limit:
                continue
            perp = abs(ex * dz - ez * dx)
            if perp <= float(ent.get("radius", 0.4)):
                if best is None or t < best[1]:
                    best = (ent["name"], t)
        steps = int(limit) + 1
        for s in range(1, steps + 1):
            cell = self._cell(self.pos[0] + dx * s, self.pos[2] + dz * s)
            name = self.blocks.get(cell)
            if cell in self.water:
                name = name or "water"
            if cell in self.lava:
                name = name or "lava"
            if name:
                dist = float(s)
                if best is None or dist < best[1]:
                    best = (name, dist)
                break
        return best

    def distance_to(self, name: str):
        import math as _math

        from lfnav.gt.scene_state import resolve_label

        labels = self.cfg.get("labels") or {}
        objects = list(labels.get("objects") or [])
        aliases = labels.get("aliases") or {}
        best = None
        for ent in self.entities:
            resolved = resolve_label(ent.get("name"), aliases, objects)
            if ent.get("name") == name or resolved == name:
                dist = _math.hypot(float(ent["x"]) - self.pos[0], float(ent["z"]) - self.pos[2])
                best = dist if best is None else min(best, dist)
        for (x, z), raw in self.blocks.items():
            resolved = resolve_label(raw, aliases, objects)
            if raw == name or resolved == name:
                dist = _math.hypot((x + 0.5) - self.pos[0], (z + 0.5) - self.pos[2])
                best = dist if best is None else min(best, dist)
        return best
