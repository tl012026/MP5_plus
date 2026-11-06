"""Automatic success checks. near-X means the target and X are both in the same view."""

from __future__ import annotations

import math

import logging

from lfnav.gt.scene_state import resolve_label
from lfnav.obsutil import lookup
from lfnav.types import SceneState, TaskSpec

log = logging.getLogger("lfnav.gt.judge")


def aligned_target_distance(obs: dict, cfg: dict, target: str) -> float | None:
    """Nearest aligned-ray hit on the target. None if the target is not in the camera frame.

    SR-b uses this instead of env.distance_to, which the real MineDojo env does not implement.
    Wide lidar is ignored: the target has to be in view.
    """
    keys = cfg.get("obs_keys") or {}
    labels = cfg.get("labels") or {}
    objects = list(labels.get("objects") or [])
    aliases = labels.get("aliases") or {}
    rays = lookup(obs, keys.get("aligned_rays"), []) or []
    best = None
    for ray in rays:
        if resolve_label(ray.get("label"), aliases, objects) != target:
            continue
        try:
            dist = float(ray["distance"])
        except (KeyError, TypeError, ValueError):
            continue
        if best is None or dist < best:
            best = dist
    return best


def _holds(scene: SceneState, task: TaskSpec, tau: float = 0.5) -> tuple[bool, list[str]]:
    missing = []
    if scene.object(task.target).present < tau:
        missing.append("target")
    cond = task.conditions or {}
    for name in list(cond.get("near") or []):
        if scene.object(name).present < tau:
            missing.append(f"near:{name}")
    unknown = set(scene.unknown or [])
    for field in ("biome", "time", "weather", "brightness"):
        required = cond.get(field)
        if not required:
            continue
        if field in unknown:
            log.warning("%s condition %s is unknown, not a silent miss", field, required)
            missing.append(f"{field}:{required}:unknown")
            continue
        if float(getattr(scene, field).get(required, 0.0)) < tau:
            missing.append(f"{field}:{required}")
    return not missing, missing


def straight_distance(start, stop) -> float:
    if start is None or stop is None:
        return 0.0
    acc = 0.0
    for a, b in zip(start, stop):
        acc += (float(a) - float(b)) ** 2
    return math.sqrt(acc)


def spl_approx(sr_b: bool, straight: float, path_length: float) -> float:
    """SR-b * d / max(L, d). If both lengths are 0, a success scores 1."""
    if not sr_b:
        return 0.0
    if straight <= 0.0 and path_length <= 0.0:
        return 1.0
    return straight / max(path_length, straight)


def judge_stop(
    task: TaskSpec,
    scene: SceneState,
    distance_blocks: float | None,
    radius: float = 3.0,
    tau: float = 0.5,
) -> dict:
    """Call this when the agent has stopped on its own."""
    sr_a, missing = _holds(scene, task, tau=tau)
    in_range = distance_blocks is not None and float(distance_blocks) <= float(radius)
    sr_b = bool(sr_a and in_range)
    if sr_b:
        reason = "target_in_view_and_within_radius_and_conditions"
    elif sr_a:
        reason = "conditions_hold_but_too_far"
    else:
        reason = "conditions_missing:" + ",".join(missing)
    return {
        "sr_a": sr_a,
        "sr_b": sr_b,
        "false_stop": not sr_a,
        "timeout": False,
        "missing": missing,
        "reason": reason,
        "distance_blocks": distance_blocks,
    }


def judge_episode(
    task: TaskSpec,
    stopped: bool,
    steps: int,
    scene: SceneState | None,
    distance_blocks: float | None,
    path_length: float,
    start,
    stop,
    max_steps: int = 3000,
    radius: float = 3.0,
) -> dict:
    if not stopped:
        timeout = steps >= max_steps
        return {
            "sr_a": False,
            "sr_b": False,
            "false_stop": False,
            "timeout": timeout,
            "missing": ["not_stopped"],
            "reason": "timeout" if timeout else "still_running",
            "distance_blocks": distance_blocks,
            "path_length": path_length,
            "straight": straight_distance(start, stop),
            "spl": 0.0,
            "outcome": "timeout" if timeout else "incomplete",
            "success": False,
            "steps": steps,
            "start": start,
            "stop": stop,
        }
    verdict = judge_stop(task, scene or SceneState(), distance_blocks, radius=radius)
    straight = straight_distance(start, stop)
    verdict["path_length"] = path_length
    verdict["straight"] = straight
    verdict["spl"] = spl_approx(verdict["sr_b"], straight, path_length)
    verdict["steps"] = steps
    verdict["start"] = list(start) if start is not None else None
    verdict["stop"] = list(stop) if stop is not None else None
    if verdict["sr_b"]:
        verdict["outcome"] = "success"
        verdict["success"] = True
    elif verdict["sr_a"]:
        verdict["outcome"] = "fail_too_far"
        verdict["success"] = False
    else:
        verdict["outcome"] = "false_stop"
        verdict["success"] = False
    return verdict
