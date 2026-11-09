"""Read nested observation fields from a configured path. No MineDojo import."""

from __future__ import annotations


def scalar(x):
    """Numpy 0-d / size-1 array -> Python scalar. Anything else is returned as-is."""
    if x is None:
        return None
    item = getattr(x, "item", None)
    size = getattr(x, "size", None)
    if callable(item) and size == 1:
        try:
            return item()
        except Exception:
            return x
    return x


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
    yaw = scalar(lookup(obs, keys.get("yaw"), 0.0))
    return float(0.0 if yaw is None else yaw)


def pitch_of(obs: dict, cfg: dict) -> float:
    keys = cfg.get("obs_keys") or {}
    pitch = scalar(lookup(obs, keys.get("pitch"), 0.0))
    return float(0.0 if pitch is None else pitch)
