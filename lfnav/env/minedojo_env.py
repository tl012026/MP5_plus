"""MineDojo adapter. Imported only on a machine that has minedojo.

The make() arguments are the same fields MP5 already passes in run_agent.make_minedojo_env.
Aligned lidar rays are added only when cfg['use_aligned_lidar'] is true, after the camera
fit has been written into fov_deg / fov_axis. Reset is reused across episodes.
"""

from __future__ import annotations

import math

from lfnav.geom import aligned_subrays_n, patch_grid, rays_from_cfg, subray_grid_from_cfg
from lfnav.gt.fov import ray_angles_deg
from lfnav.obsutil import scalar


def to_minedojo_lidar(direction, reach: float) -> tuple[float, float, float]:
    """lfnav camera direction -> MineDojo lidar_rays (pitch, yaw, range).

    lfnav: +x right, +y down, +z forward; yaw+ is right; pitch+ is down
    (same as lfnav.gt.fov). MineDojo lidar_rays: yaw+ is left, pitch+ is up.
    So this function negates both angles. Evidence: verify_alignment overlay
    2026-11-08 on 512x820 — pig pixels at x≈0–180, y≈90–310 while pig-labeled
    patches sat at x≈630–820, y≈190–415 (left-right and up-down mirrored).
    Intrinsics calibration is unaffected. Call this once; do not negate again
    in convert_rays, the expert, or scene_state.
    """
    yaw_deg, pitch_deg = ray_angles_deg(direction)
    return (math.radians(-pitch_deg), math.radians(-yaw_deg), float(reach))


def from_minedojo_lidar(pitch_rad, yaw_rad) -> tuple[float, float, float]:
    """Inverse of to_minedojo_lidar: MineDojo (pitch, yaw) -> lfnav unit ray."""
    yaw = -float(yaw_rad)
    pitch = -float(pitch_rad)
    cp = math.cos(pitch)
    x = math.sin(yaw) * cp
    y = math.sin(pitch)
    z = math.cos(yaw) * cp
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    return (x / n, y / n, z / n)


def aligned_lidar_rays(cfg: dict) -> list[tuple[float, float, float]]:
    height, width = cfg.get("image_hw", [224, 364])
    n_h, n_w = patch_grid(int(height), int(width), int(cfg.get("patch", 14)))
    reach = float(cfg.get("aligned_ray_range", 16))
    n_sub = aligned_subrays_n(cfg)
    out = []
    if n_sub <= 1:
        rays = rays_from_cfg(cfg)
        for row in rays:
            for direction in row:
                out.append(to_minedojo_lidar(direction, reach))
        return out
    grid = subray_grid_from_cfg(cfg)
    for row in grid:
        for dirs in row:
            for direction in dirs:
                out.append(to_minedojo_lidar(direction, reach))
    return out


def wide_lidar_rays(cfg: dict) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]]]:
    """MineDojo tuples plus lfnav (pitch_deg, yaw_deg) for the expert. yaw+ is right.

    Extra pitch rings look down. Config pitches use MineDojo's sign (pitch+ is up),
    so -3/-8 aim under the eye-height horizontal and hit short mobs (pig ~0.9).
    """
    step = float(cfg.get("wide_yaw_step", 15))
    reach = float(cfg.get("wide_ray_range", 32))
    pitches = list(cfg.get("wide_pitches_deg") or [0.0, -3.0, -8.0])
    out = []
    lfnav_deg = []
    for pitch_deg in pitches:
        p = math.radians(float(pitch_deg))
        cp = math.cos(p)
        # Config pitch+ is up; lfnav +y is down, so y = -sin(config_pitch).
        ly = -math.sin(p)
        yaw = -180.0
        while yaw < 180.0 - 1e-6:
            direction = (math.sin(math.radians(yaw)) * cp, ly, math.cos(math.radians(yaw)) * cp)
            out.append(to_minedojo_lidar(direction, reach))
            lfnav_deg.append((float(pitch_deg), yaw))
            yaw += step
    return out, lfnav_deg


def lidar_plan(cfg: dict) -> dict:
    """One lidar_rays list: image-aligned rays, then the wide ring when wide_lidar is on."""
    height, width = cfg.get("image_hw", [224, 364])
    n_h, n_w = patch_grid(int(height), int(width), int(cfg.get("patch", 14)))
    aligned = aligned_lidar_rays(cfg)
    cells = [(row, col) for row in range(n_h) for col in range(n_w)]
    n_sub = aligned_subrays_n(cfg)
    expected = len(cells) if n_sub <= 1 else len(cells) * n_sub * n_sub
    if len(aligned) != expected:
        raise RuntimeError(
            f"aligned rays {len(aligned)} do not match the {n_h}x{n_w} patch grid with subrays={n_sub}"
        )
    wide, wide_lfnav = wide_lidar_rays(cfg) if cfg.get("wide_lidar", True) else ([], [])
    return {
        "lidar_rays": aligned + wide,
        "aligned": (0, len(aligned)),
        "wide": (len(aligned), len(aligned) + len(wide)),
        "rows_cols": cells,
        "wide_angles_deg": wide_lfnav,
        "subrays": n_sub,
    }


def nearest_patch(direction, cfg: dict) -> tuple[int, int]:
    rays = rays_from_cfg(cfg)
    best_dot, best = -2.0, (0, 0)
    tx, ty, tz = direction
    for i, row in enumerate(rays):
        for j, (x, y, z) in enumerate(row):
            dot = x * tx + y * ty + z * tz
            if dot > best_dot:
                best_dot, best = dot, (i, j)
    return best


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
    cells = plan["rows_cols"]
    n_sub = max(1, int(plan.get("subrays") or 1))
    if n_sub <= 1:
        for offset, (row, col) in enumerate(cells):
            label, dist = one(start + offset)
            aligned.append({"row": row, "col": col, "label": label, "distance": dist})
        if stop != start + len(cells):
            raise RuntimeError("aligned ray span does not match the patch cells")
    else:
        per = n_sub * n_sub
        if stop != start + len(cells) * per:
            raise RuntimeError("aligned ray span does not match the patch cells times subrays")
        extras = []
        for cell_i, (row, col) in enumerate(cells):
            by_obj = {}
            closest = (None, None)
            for s in range(per):
                label, dist = one(start + cell_i * per + s)
                if label is None or dist is None:
                    continue
                if label not in by_obj or dist < by_obj[label]:
                    by_obj[label] = dist
                if closest[1] is None or dist < closest[1]:
                    closest = (label, dist)
            if not by_obj:
                aligned.append({"row": row, "col": col, "label": None, "distance": None})
                continue
            primary, pdist = closest
            aligned.append({"row": row, "col": col, "label": primary, "distance": pdist})
            for name, dist in by_obj.items():
                if name != primary:
                    extras.append({"row": row, "col": col, "label": name, "distance": dist})
        aligned.extend(extras)
    wide = []
    w0, _w1 = plan["wide"]
    for offset, (pitch, yaw) in enumerate(plan["wide_angles_deg"]):
        label, dist = one(w0 + offset)
        wide.append({
            "pitch": pitch,
            "yaw": yaw,
            "yaw_offset": yaw,  # lfnav degrees: +yaw is right, already un-flipped in lidar_plan
            "label": label,
            "distance": dist,
        })
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


def mob_ring_box(min_d, max_d, seed: int, behind: bool = False, y_low: int = 1, y_high: int = 3):
    """MineDojo only takes an axis-aligned box, so pick one cardinal ring sector.

    hard behind=True uses the agent's back (-z when yaw 0 faces +z).
    Box corners can sit slightly outside [min_d, max_d]; the true spawn distance
    is recorded after reset as initial_target_distance.
    """
    import random

    min_d = float(min_d)
    max_d = float(max_d)
    if max_d < min_d:
        min_d, max_d = max_d, min_d
    w = max(abs(min_d) / 2.0, 0.5)
    dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))
    if behind:
        dx, dz = (0, -1)
    else:
        dx, dz = random.Random(int(seed)).choice(dirs)
    if dz == 1:
        x0, x1, z0, z1 = -w, w, min_d, max_d
    elif dz == -1:
        x0, x1, z0, z1 = -w, w, -max_d, -min_d
    elif dx == 1:
        x0, x1, z0, z1 = min_d, max_d, -w, w
    else:
        x0, x1, z0, z1 = -max_d, -min_d, -w, w
    low = (min(x0, x1), float(y_low), min(z0, z1))
    high = (max(x0, x1), float(y_high), max(z0, z1))
    return low, high


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
    if not table.get("leave_biome"):
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
    # Old path: HarvestMeta reset-hook /summon. It does not actually spawn (2026-11).
    # Keep the code; turn it on with minedojo.spawn_via_reset_hook.
    if env_cfg.get("spawn_via_reset_hook", False):
        _put(out, names, "mobs", mobs)
        if "spawn_distance_min" in table and "spawn_distance_max" in table:
            behind = str(cfg.get("difficulty") or "") == "hard"
            low, high = mob_ring_box(
                table["spawn_distance_min"], table["spawn_distance_max"], seed, behind=behind,
            )
            _put(out, names, "mob_range_low", low)
            _put(out, names, "mob_range_high", high)
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


def sample_mob_offset(min_d, max_d, yaw_deg, seed: int, index: int, behind: bool = False, y: int = 1):
    """World offset for /summon. yaw=0 faces +z; forward is (-sin yaw, cos yaw)."""
    import random

    rng = random.Random(int(seed) * 7919 + int(index))
    radius = rng.uniform(float(min_d), float(max_d))
    if behind:
        phi = rng.uniform(180.0 - 45.0, 180.0 + 45.0)
    else:
        phi = rng.uniform(-180.0, 180.0)
    bearing = math.radians(float(yaw_deg) + phi)
    dx = int(round(-radius * math.sin(bearing)))
    dz = int(round(radius * math.cos(bearing)))
    dy = int(y)
    return {"dx": dx, "dy": dy, "dz": dz, "radius": radius, "bearing_deg": phi, "y": dy}


def _noop_tick(env) -> dict:
    inner = env._env
    nop = inner.action_space.no_op()
    obs, _reward, _done, _info = inner.step(nop)
    adopt = getattr(env, "_adopt", None)
    return adopt(obs) if callable(adopt) else obs


def _sim(env):
    inner = env._env
    return getattr(inner, "env", inner)


def _execute_cmd(env, cmd: str) -> None:
    _sim(env).execute_cmd(cmd)


def _execute_summon(env, name: str, dx: int, dy: int, dz: int) -> None:
    _execute_cmd(env, f"/summon {name} ~{dx} ~{dy} ~{dz}")


MC_FILL_LIMIT = 32768


def task_mentions_grass(task) -> bool:
    target = getattr(task, "target", None)
    if target is None and isinstance(task, dict):
        target = task.get("target")
    if str(target or "").strip().lower() == "grass":
        return True
    cond = getattr(task, "conditions", None)
    if cond is None and isinstance(task, dict):
        cond = task.get("conditions")
    near = (cond or {}).get("near") or []
    if isinstance(near, str):
        near = [near]
    return any(str(name).strip().lower() == "grass" for name in near)


def vegetation_fill_commands(radius_xz: int = 32, radius_y: int = 4, blocks=("tallgrass", "double_plant")) -> list[str]:
    """Chunked /fill replace so each command stays under the 32768-block cap."""
    y0, y1 = -int(radius_y), int(radius_y)
    z0, z1 = -int(radius_xz), int(radius_xz)
    ny = y1 - y0 + 1
    nz = z1 - z0 + 1
    max_nx = max(1, MC_FILL_LIMIT // max(ny * nz, 1))
    cmds = []
    x = -int(radius_xz)
    x_end = int(radius_xz)
    while x <= x_end:
        x1 = min(x + max_nx - 1, x_end)
        vol = (x1 - x + 1) * ny * nz
        if vol > MC_FILL_LIMIT:
            raise RuntimeError(f"fill volume {vol} exceeds {MC_FILL_LIMIT}")
        for block in blocks:
            cmds.append(f"/fill ~{x} ~{y0} ~{z0} ~{x1} ~{y1} ~{z1} air 0 replace {block}")
        x = x1 + 1
    return cmds


def _verify_and_retry_target(env, obs, cfg, target, names, seed, min_d, max_d, yaw, hard, retries, verify, summon_one, meta):
    """Old ray-check retry. Kept behind minedojo.spawn_verify_retry (default false)."""
    from lfnav.gt.initial import privileged_ray_distance

    attempts = 1
    failed = privileged_ray_distance(obs, cfg, target) is None
    target_index = next(i for i, name in enumerate(names) if name == target)
    while failed and attempts < max(retries, 1):
        attempts += 1
        obs = summon_one(target, target_index + 1000 * (attempts - 1), behind=hard, y=3)
        for _ in range(max(verify, 0)):
            obs = _noop_tick(env)
        failed = privileged_ray_distance(obs, cfg, target) is None
    meta["spawn_attempts"] = attempts
    return obs


def spawn_task_mobs(env, task, cfg: dict, seed: int) -> dict:
    """Summon task mobs after reset. Invalid chat commands do not advance the tick."""
    from lfnav.obsutil import yaw_of

    env_cfg = cfg.get("minedojo") or {}
    settle = int(env_cfg.get("settle_ticks", 5))
    verify = int(env_cfg.get("verify_ticks", 10))
    retries = int(env_cfg.get("spawn_retries", 3))
    drop_y = int(env_cfg.get("spawn_drop_height", 2))
    settle_after = int(env_cfg.get("settle_after_spawn", 20))
    table = _difficulty(cfg)
    episode = _episode_of(task)
    required = _mob_names(episode)
    distractors = _distractor_names(required, cfg, table, seed)
    names = required + distractors
    obs = getattr(env, "_obs", None)
    meta = {
        "spawn_failed": False,
        "spawn_attempts": 0,
        "spawn_offsets": [],
        "target_spawn_offset": None,
        "n_targets_spawned": 0,
        "natural_mobs_cleared": False,
        "vegetation_cleared": False,
    }
    env.spawn_meta = meta
    if not names:
        return obs
    try:
        for _ in range(max(settle, 0)):
            obs = _noop_tick(env)
        if env_cfg.get("clear_entities", True):
            _execute_cmd(env, "/kill @e[type=!player]")
            meta["natural_mobs_cleared"] = True
            for _ in range(2):
                obs = _noop_tick(env)
        skip_plants = task_mentions_grass(task)
        if env_cfg.get("clear_vegetation", True) and not skip_plants:
            for cmd in vegetation_fill_commands():
                _execute_cmd(env, cmd)
                obs = _noop_tick(env)
            meta["vegetation_cleared"] = True
        yaw = yaw_of(obs, cfg)
        min_d = float(table.get("spawn_distance_min", 4))
        max_d = float(table.get("spawn_distance_max", 8))
        hard = str(cfg.get("difficulty") or "") == "hard"
        target = getattr(task, "target", None) or (task.get("target") if isinstance(task, dict) else None)

        def summon_one(name, index, behind, y):
            offset = sample_mob_offset(min_d, max_d, yaw, seed, index, behind=behind, y=y)
            offset = dict(offset)
            offset["name"] = name
            _execute_summon(env, name, offset["dx"], offset["dy"], offset["dz"])
            tick = _noop_tick(env)
            if env_cfg.get("spawn_resistance", True):
                _execute_cmd(env, f"/effect @e[type={name}] resistance 10 4")
                tick = _noop_tick(env)
            meta["spawn_offsets"].append(offset)
            if name == target:
                meta["n_targets_spawned"] = int(meta["n_targets_spawned"]) + 1
                if meta["target_spawn_offset"] is None:
                    meta["target_spawn_offset"] = {"dx": offset["dx"], "dz": offset["dz"]}
            return tick

        for i, name in enumerate(names):
            obs = summon_one(name, i, behind=(hard and name == target), y=drop_y)
        if env_cfg.get("freeze_mobs", True):
            seen = []
            for name in names:
                if name in seen:
                    continue
                seen.append(name)
                _execute_cmd(env, f"/effect @e[type={name}] slowness 1000000 10 true")
                obs = _noop_tick(env)
        if env_cfg.get("spawn_verify_retry", False) and target and any(n == target for n in names):
            for _ in range(max(verify, 0)):
                obs = _noop_tick(env)
            obs = _verify_and_retry_target(
                env, obs, cfg, target, names, seed, min_d, max_d, yaw, hard, retries, verify, summon_one, meta,
            )
        else:
            for _ in range(max(settle_after, 0)):
                obs = _noop_tick(env)
            meta["spawn_attempts"] = 1 if meta["n_targets_spawned"] else 0
        env.spawn_meta = meta
        return obs
    except Exception:
        meta["spawn_failed"] = True
        env.spawn_meta = meta
        return obs


def python_location_stats(loc: dict) -> dict:
    """MineDojo location_stats: size-1 arrays become Python scalars; pos is 3 floats.

    Real dumps (probe 2026-11-08) store biome_id as integer (1,), is_raining as
    bool (1,), other scalars as float (1,), and pos as length 3. rgb/voxels/rays
    are not touched here.
    """
    out = {}
    for key, val in loc.items():
        if key == "pos":
            out[key] = None if val is None else [float(v) for v in val]
            continue
        out[key] = scalar(val)
    return out


class MineDojoEnv:
    def __init__(self, cfg: dict, task=None, seed: int = 0):
        import minedojo
        self.cfg = cfg
        self.task = task
        self.seed = int(seed)
        self.spawn_meta = {}
        self.ray_plan = lidar_plan(cfg) if (cfg.get("minedojo") or {}).get("use_lidar", True) else None
        from lfnav.env.gradle_home import env_create_guard
        with env_create_guard(cfg):
            self._env = minedojo.make(**_kwargs(cfg, seed, task))
        self._obs = None
        from lfnav.env.frames import FrameRecorder
        self.frames = FrameRecorder(cfg)

    def _adopt(self, obs):
        if isinstance(obs, dict):
            obs = dict(obs)
            loc = obs.get("location_stats")
            if isinstance(loc, dict):
                obs["location_stats"] = python_location_stats(loc)
            if self.ray_plan is not None and isinstance(obs.get("rays"), dict):
                converted = convert_rays(obs["rays"], self.ray_plan, self.cfg)
                keys = self.cfg.get("obs_keys") or {}
                obs[keys.get("aligned_rays") or "lfnav_rays"] = converted["aligned"]
                obs[keys.get("wide_rays") or "lfnav_wide_rays"] = converted["wide"]
        self._obs = obs
        return obs

    def reset(self, **kwargs):
        obs = self._adopt(self._env.reset())
        task = kwargs.get("task", self.task)
        seed = int(kwargs.get("seed", self.seed))
        self.task = task
        self.seed = seed
        if task is not None and not (self.cfg.get("minedojo") or {}).get("spawn_via_reset_hook", False):
            obs = spawn_task_mobs(self, task, self.cfg, seed)
        return obs

    def step(self, action):
        if isinstance(action, str):
            vectors = (self.cfg.get("actions") or {}).get("vectors") or {}
            if not (self.cfg.get("actions") or {}).get("confirmed", False):
                raise RuntimeError("Action vectors are not confirmed. Run probe_obs.py and set actions.confirmed.")
            action = vectors[action]
        obs, reward, done, info = self._env.step(action)
        obs = self._adopt(obs)
        self.frames.after_step(obs)
        return obs, reward, done, info

    def observation(self):
        return self._obs

    def close(self):
        self._env.close()
