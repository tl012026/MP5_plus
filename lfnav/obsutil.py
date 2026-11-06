"""Read nested observation fields from a configured path. No MineDojo import."""

from __future__ import annotations


def lookup(obs: dict, path, default=None):
    if path is None or obs is None:
        return default
    if isinstance(path, str):
        return obs.get(path, default)
    cur = obs
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def position_of(obs: dict, cfg: dict):
    keys = cfg.get("obs_keys") or {}
    pos = lookup(obs, keys.get("position"))
    if pos is None:
        return None
    return [float(v) for v in pos]


def yaw_of(obs: dict, cfg: dict) -> float:
    keys = cfg.get("obs_keys") or {}
    yaw = lookup(obs, keys.get("yaw"), 0.0)
    return float(yaw or 0.0)
