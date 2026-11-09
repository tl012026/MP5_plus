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


def is_empty(x) -> bool:
    """True if x is missing or has size 0. Does not truth-test numpy arrays."""
    if x is None:
        return True
    size = getattr(x, "size", None)
    if size is not None:
        try:
            return int(size) == 0
        except (TypeError, ValueError):
            return False
    if isinstance(x, (list, tuple)):
        return len(x) == 0
    return False


def voxels_of(obs: dict, cfg: dict):
    """Aligned voxel crop from obs_keys.voxels, or None if missing/empty."""
    keys = cfg.get("obs_keys") or {}
    voxels = lookup(obs, keys.get("voxels"))
    if is_empty(voxels):
        return None
    return voxels


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
