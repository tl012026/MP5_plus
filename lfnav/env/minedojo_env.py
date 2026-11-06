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


def _kwargs(cfg: dict, seed: int) -> dict:
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
    for key in ("target_names", "target_quantities", "spawn_rate", "break_speed_multiplier", "specified_biome", "start_at_night"):
        if env_cfg.get(key) is not None:
            kwargs[key] = env_cfg[key]
    if env_cfg.get("spawn_range_low") is not None:
        kwargs["spawn_range_low"] = tuple(env_cfg["spawn_range_low"])
        kwargs["spawn_range_high"] = tuple(env_cfg["spawn_range_high"])
    if kwargs["use_lidar"]:
        if cfg.get("use_aligned_lidar"):
            kwargs["lidar_rays"] = aligned_lidar_rays(cfg)
        elif env_cfg.get("lidar_rays") is not None:
            kwargs["lidar_rays"] = [tuple(ray) for ray in env_cfg["lidar_rays"]]
    return kwargs


class MineDojoEnv:
    def __init__(self, cfg: dict, task=None, seed: int = 0):
        import minedojo
        self.cfg = cfg
        self.task = task
        self._env = minedojo.make(**_kwargs(cfg, seed))
        self._obs = None

    def reset(self, **_kwargs):
        self._obs = self._env.reset()
        return self._obs

    def step(self, action):
        if isinstance(action, str):
            vectors = (self.cfg.get("actions") or {}).get("vectors") or {}
            if not (self.cfg.get("actions") or {}).get("confirmed", False):
                raise RuntimeError("Action vectors are not confirmed. Run probe_obs.py and set actions.confirmed.")
            action = vectors[action]
        self._obs, reward, done, info = self._env.step(action)
        return self._obs, reward, done, info

    def observation(self):
        return self._obs

    def close(self):
        self._env.close()
