"""MineDojo adapter. Imported only on a machine that has minedojo.

The make() arguments are the same fields MP5 already passes in run_agent.make_minedojo_env.
Aligned lidar rays are added only when cfg['use_aligned_lidar'] is true, after the camera
fit has been written into fov_deg / fov_axis. Reset is reused across episodes.
"""

from __future__ import annotations

import math

from lfnav.geom import patch_grid, ray_directions
from lfnav.gt.fov import ray_angles_deg


def aligned_lidar_rays(cfg: dict) -> list[tuple[float, float, float]]:
    height, width = cfg.get("image_hw", [224, 364])
    n_h, n_w = patch_grid(int(height), int(width), int(cfg.get("patch", 14)))
    rays = ray_directions(n_h, n_w, float(cfg.get("fov_deg", 70)), str(cfg.get("fov_axis", "vertical")))
    reach = float(cfg.get("aligned_ray_range", 16))
    out = []
    for row in rays:
        for direction in row:
            yaw, pitch = ray_angles_deg(direction)
            out.append((math.radians(pitch), math.radians(yaw), reach))
    return out


def wide_lidar_rays(cfg: dict) -> list[tuple[float, float, float]]:
    step = float(cfg.get("wide_yaw_step", 15))
    reach = float(cfg.get("wide_ray_range", 32))
    yaw = -180.0
    out = []
    while yaw < 180.0 - 1e-6:
        out.append((0.0, math.radians(yaw), reach))
        yaw += step
    return out


def lidar_plan(cfg: dict) -> dict:
    """One lidar_rays list: image-aligned rays, then the wide ring when wide_lidar is on."""
    height, width = cfg.get("image_hw", [224, 364])
    n_h, n_w = patch_grid(int(height), int(width), int(cfg.get("patch", 14)))
    aligned = aligned_lidar_rays(cfg)
    cells = [(row, col) for row in range(n_h) for col in range(n_w)]
    if len(cells) != len(aligned):
        raise RuntimeError(f"aligned rays {len(aligned)} do not match the {n_h}x{n_w} patch grid")
    wide = wide_lidar_rays(cfg) if cfg.get("wide_lidar", True) else []
    return {
        "lidar_rays": aligned + wide,
        "aligned": (0, len(aligned)),
        "wide": (len(aligned), len(aligned) + len(wide)),
        "rows_cols": cells,
        "wide_angles_deg": [(math.degrees(pitch), math.degrees(yaw)) for pitch, yaw, _reach in wide],
    }


def _ray_at(values, index):
    if values is None:
        return None
    try:
        value = values[index]
    except Exception:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _is_miss(name, miss_names: set[str]) -> bool:
    if name is None:
        return True
    return str(name).strip().lower() in miss_names


def _distance(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_hit(block_name, block_distance, entity_name, entity_distance, miss_names: set[str]):
    """Entity wins only when it is a real hit closer than the block. Otherwise the block."""
    block_miss = _is_miss(block_name, miss_names)
    entity_miss = _is_miss(entity_name, miss_names)
    block_dist = None if block_miss else _distance(block_distance)
    entity_dist = None if entity_miss else _distance(entity_distance)
    if entity_dist is not None and (block_dist is None or entity_dist < block_dist):
        return str(entity_name), entity_dist
    if block_dist is not None:
        return str(block_name), block_dist
    return None, None


def convert_rays(rays: dict, plan: dict, cfg: dict) -> dict:
    """Turn MineDojo's parallel ray arrays into aligned and wide lfnav rays."""
    from lfnav.gt.scene_state import resolve_label

    fields = ((cfg.get("obs_keys") or {}).get("ray_fields") or {})
    miss_names = {str(name).strip().lower() for name in (fields.get("miss_names") or ["", "air", "null", "none"])}
    labels = cfg.get("labels") or {}
    objects = list(labels.get("objects") or [])
    aliases = labels.get("aliases") or {}
    block_names = rays.get(fields.get("block_name", "block_name"))
    block_dists = rays.get(fields.get("block_distance", "block_distance"))
    entity_names = rays.get(fields.get("entity_name", "entity_name"))
    entity_dists = rays.get(fields.get("entity_distance", "entity_distance"))

    def one(index):
        raw, dist = first_hit(
            _ray_at(block_names, index),
            _ray_at(block_dists, index),
            _ray_at(entity_names, index),
            _ray_at(entity_dists, index),
            miss_names,
        )
        label = None if raw is None else resolve_label(raw, aliases, objects)
        return label, dist

    aligned = []
    start, stop = plan["aligned"]
    for offset, (row, col) in enumerate(plan["rows_cols"]):
        label, dist = one(start + offset)
        aligned.append({"row": row, "col": col, "label": label, "distance": dist})
    wide = []
    w0, _w1 = plan["wide"]
    for offset, (pitch, yaw) in enumerate(plan["wide_angles_deg"]):
        label, dist = one(w0 + offset)
        wide.append({
            "pitch": pitch,
            "yaw": yaw,
            "yaw_offset": yaw,
            "label": label,
            "distance": dist,
        })
    if stop != start + len(plan["rows_cols"]):
        raise RuntimeError("aligned ray span does not match the patch cells")
    return {"aligned": aligned, "wide": wide}


def _episode_of(task) -> dict:
    if task is None:
        return {}
    episode = task.episode if hasattr(task, "episode") else task.get("episode")
    return dict(episode or {})


def _put(out: dict, names: dict, logical: str, value) -> None:
    key = names.get(logical)
    if not key or value is None:
        return
    out[key] = value


def _difficulty(cfg: dict) -> dict:
    name = cfg.get("difficulty") or "easy"
    table = (cfg.get("difficulty_table") or {}).get(name)
    if not isinstance(table, dict):
        raise KeyError(f"difficulty {name!r} is missing from the lfnav difficulty table")
    return table


def _mob_names(episode: dict) -> list[str]:
    names = []
    for item in episode.get("spawn") or []:
        if item.get("kind") == "mob" and item.get("name"):
            names.append(str(item["name"]))
    return names


def _distractor_names(required: list[str], cfg: dict, table: dict, seed: int) -> list[str]:
    count = int(table.get("distractors") or 0)
    if count <= 0:
        return []
    pool = [str(name) for name in ((cfg.get("minedojo") or {}).get("distractor_mobs") or []) if str(name) not in required]
    if not pool:
        return []
    import random
    rng = random.Random(seed)
    return [rng.choice(pool) for _ in range(count)]


def episode_kwargs(task, cfg: dict, seed: int) -> dict:
    """Map one task's scene onto minedojo.make arguments.

    Argument names come from minedojo.param_names. Block spawns (tree, grass,
    water) are not placed; the biome is expected to grow them.
    """
    env_cfg = cfg.get("minedojo") or {}
    names = env_cfg.get("param_names") or {}
    episode = _episode_of(task)
    table = _difficulty(cfg)
    out: dict = {}
    _put(out, names, "biome", episode.get("specified_biome"))
    time_name = episode.get("start_time")
    time_key = names.get("time")
    if time_key and time_name is not None:
        values = env_cfg.get("time_values") or {}
        if time_name not in values:
            raise KeyError(f"minedojo.time_values has no entry for {time_name!r}")
        out[time_key] = values[time_name]
    weather = episode.get("initial_weather")
    weather_values = env_cfg.get("weather_values") or {}
    if weather in weather_values:
        weather = weather_values[weather]
    _put(out, names, "weather", weather)
    required = _mob_names(episode)
    mobs = required + _distractor_names(required, cfg, table, seed)
    _put(out, names, "mobs", mobs)
    if "spawn_distance_min" in table:
        low = int(table["spawn_distance_min"])
        _put(out, names, "mob_range_low", (-low, 1, -low))
    if "spawn_distance_max" in table:
        high = int(table["spawn_distance_max"])
        _put(out, names, "mob_range_high", (high, 3, high))
    return out


def _default_scene_kwargs(cfg: dict) -> dict:
    """Scene used when no task is attached. Names still come from param_names."""
    env_cfg = cfg.get("minedojo") or {}
    names = env_cfg.get("param_names") or {}
    out: dict = {}
    _put(out, names, "biome", env_cfg.get("specified_biome"))
    time_key = names.get("time")
    if time_key:
        clock = "night" if env_cfg.get("start_at_night") else "day"
        values = env_cfg.get("time_values") or {}
        if clock in values:
            out[time_key] = values[clock]
    return out


def _kwargs(cfg: dict, seed: int, task=None) -> dict:
    env_cfg = cfg.get("minedojo") or {}
    if not env_cfg.get("task_id"):
        raise RuntimeError("Set minedojo.task_id in the lfnav config before creating a real environment.")
    vradius = int(env_cfg.get("voxel_radius", 5))
    kwargs = dict(
        task_id=env_cfg["task_id"],
        image_size=tuple(env_cfg.get("image_size", [512, 820])),
        world_seed=int(seed),
        seed=int(env_cfg.get("seed", 0)),
        use_voxel=bool(env_cfg.get("use_voxel", True)),
        voxel_size=dict(xmin=-vradius, ymin=-vradius, zmin=-vradius, xmax=vradius, ymax=vradius, zmax=vradius),
        use_lidar=bool(env_cfg.get("use_lidar", True)),
    )
    for key in ("target_names", "target_quantities", "spawn_rate", "break_speed_multiplier"):
        if env_cfg.get(key) is not None:
            kwargs[key] = env_cfg[key]
    if env_cfg.get("spawn_range_low") is not None:
        kwargs["spawn_range_low"] = tuple(env_cfg["spawn_range_low"])
        kwargs["spawn_range_high"] = tuple(env_cfg["spawn_range_high"])
    if kwargs["use_lidar"]:
        kwargs["lidar_rays"] = lidar_plan(cfg)["lidar_rays"]
    if task is None:
        kwargs.update(_default_scene_kwargs(cfg))
    else:
        kwargs.update(episode_kwargs(task, cfg, seed))
    return kwargs


class MineDojoEnv:
    def __init__(self, cfg: dict, task=None, seed: int = 0):
        import minedojo
        self.cfg = cfg
        self.task = task
        self.ray_plan = lidar_plan(cfg) if (cfg.get("minedojo") or {}).get("use_lidar", True) else None
        self._env = minedojo.make(**_kwargs(cfg, seed, task))
        self._obs = None

    def _adopt(self, obs):
        if isinstance(obs, dict) and self.ray_plan is not None and isinstance(obs.get("rays"), dict):
            converted = convert_rays(obs["rays"], self.ray_plan, self.cfg)
            keys = self.cfg.get("obs_keys") or {}
            obs[keys.get("aligned_rays") or "lfnav_rays"] = converted["aligned"]
            obs[keys.get("wide_rays") or "lfnav_wide_rays"] = converted["wide"]
        self._obs = obs
        return obs

    def reset(self, **_kwargs):
        return self._adopt(self._env.reset())

    def step(self, action):
        if isinstance(action, str):
            vectors = (self.cfg.get("actions") or {}).get("vectors") or {}
            if not (self.cfg.get("actions") or {}).get("confirmed", False):
                raise RuntimeError("Action vectors are not confirmed. Run probe_obs.py and set actions.confirmed.")
            action = vectors[action]
        obs, reward, done, info = self._env.step(action)
        return self._adopt(obs), reward, done, info

    def observation(self):
        return self._obs

    def close(self):
        self._env.close()
