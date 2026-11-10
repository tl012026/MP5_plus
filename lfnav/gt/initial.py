"""Privileged initial-scene metrics. Never pass these to a policy."""

from __future__ import annotations

import math

from lfnav.gt.judge import aligned_target_distance
from lfnav.gt.scene_state import resolve_label
from lfnav.obsutil import lookup
from lfnav.types import TaskSpec


def world_seed_attempt(seed: int, attempt: int) -> int:
    seed = int(seed)
    attempt = int(attempt)
    if attempt <= 0:
        return seed
    return seed * 1000 + attempt


def spawn_has_blocks(task: TaskSpec) -> bool:
    return any(item.get("kind") == "block" for item in (task.episode.get("spawn") or []))


def privileged_ray_distance(obs: dict, cfg: dict, target: str) -> float | None:
    keys = cfg.get("obs_keys") or {}
    labels = cfg.get("labels") or {}
    objects = list(labels.get("objects") or [])
    aliases = labels.get("aliases") or {}
    best = None
    for key in (keys.get("aligned_rays"), keys.get("wide_rays")):
        for ray in lookup(obs, key, []) or []:
            raw = ray.get("label")
            if raw != target and resolve_label(raw, aliases, objects) != target:
                continue
            try:
                dist = float(ray["distance"])
            except (KeyError, TypeError, ValueError):
                continue
            if best is None or dist < best:
                best = dist
    return best


def initial_scene_metrics(obs: dict, env, cfg: dict, task: TaskSpec) -> dict:
    visible = aligned_target_distance(obs, cfg, task.target) is not None
    euclidean = None
    getter = getattr(env, "distance_to", None)
    if callable(getter):
        try:
            euclidean = getter(task.target)
        except Exception:
            euclidean = None
    ray_d = privileged_ray_distance(obs, cfg, task.target)
    reach = float(cfg.get("wide_ray_range", 32))
    if euclidean is not None:
        numeric = float(euclidean)
        stored = numeric
        beyond = False
    elif ray_d is not None:
        numeric = float(ray_d)
        stored = numeric
        beyond = False
    else:
        numeric = None
        stored = f"> {reach:g}"
        beyond = True
    return {
        "initial_target_visible": bool(visible),
        "initial_target_distance": stored,
        "initial_target_distance_rays": (float(ray_d) if ray_d is not None else f"> {reach:g}"),
        "initial_target_distance_numeric": numeric,
        "initial_distance_beyond_range": beyond,
    }


def apply_spawn_offset_distance(metrics: dict, spawn_meta: dict | None, cfg: dict | None = None) -> dict:
    """Set initial_target_distance used by eval and demo collection (and by SPL).

    When minedojo.clear_entities is true (static robustness), use hypot(dx, dz)
    from /summon. When it is false (main eval / dynamic demos), use the nearest
    aligned or wide ray on the same species. spawn_target_distance is always the offset.
    """
    metrics = dict(metrics)
    if "initial_target_distance_rays" not in metrics:
        metrics["initial_target_distance_rays"] = metrics.get("initial_target_distance")
    metrics.setdefault("spawn_target_distance", None)
    metrics.setdefault("natural_target_closer", False)
    off = (spawn_meta or {}).get("target_spawn_offset") or {}
    if off.get("dx") is None or off.get("dz") is None:
        return metrics
    dist = math.hypot(float(off["dx"]), float(off["dz"]))
    metrics["spawn_target_distance"] = dist
    clear = True if cfg is None else bool((cfg.get("minedojo") or {}).get("clear_entities", True))
    if not clear:
        ray = metrics.get("initial_target_distance_rays")
        ray_n = float(ray) if isinstance(ray, (int, float)) else None
        if ray_n is not None:
            metrics["initial_target_distance"] = ray_n
            metrics["initial_target_distance_numeric"] = ray_n
            metrics["initial_distance_beyond_range"] = False
        metrics["natural_target_closer"] = bool(ray_n is not None and ray_n + 1.0 < dist)
        return metrics
    metrics["initial_target_distance"] = dist
    metrics["initial_target_distance_numeric"] = dist
    metrics["initial_distance_beyond_range"] = False
    metrics["natural_target_closer"] = False
    return metrics


def distance_meets_min(metrics: dict, min_distance: float) -> bool:
    if min_distance <= 0:
        return True
    if metrics.get("initial_distance_beyond_range"):
        return True
    numeric = metrics.get("initial_target_distance_numeric")
    if numeric is None:
        return True
    return float(numeric) >= float(min_distance)
