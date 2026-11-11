"""Shared creative16 scene builder.

MineDojo HarvestMeta's reset-hook ``/summon`` does not spawn. ``MineDojoEnv.reset``
calls ``spawn_task_mobs`` after reset: settle, ``/kill @e[type=!player]``,
vegetation ``/fill``, ring ``/summon``, resistance, freeze (slowness, gravity kept),
then settle ticks.

Call ``make_scene_env`` from both lfnav eval and MP5 so the scene is one source.
"""

from __future__ import annotations

import json
from pathlib import Path

from lfnav.env.minedojo_env import peaceful_command_sent, spawn_task_mobs
from lfnav.gt.initial import (
    apply_spawn_offset_distance,
    distance_meets_min,
    initial_scene_metrics,
    spawn_has_blocks,
    world_seed_attempt,
)
from lfnav.obsutil import can_see_sky_of
from lfnav.types import TaskSpec


def as_task_spec(task_spec) -> TaskSpec:
    if isinstance(task_spec, TaskSpec):
        return task_spec
    if isinstance(task_spec, (str, Path)):
        return TaskSpec.from_dict(json.loads(Path(task_spec).read_text()))
    return TaskSpec.from_dict(dict(task_spec))


def _initial_metrics(env, obs, cfg: dict, task: TaskSpec, used: int, reroll_count: int, unmet: bool) -> dict:
    metrics = apply_spawn_offset_distance(
        initial_scene_metrics(obs, env, cfg, task),
        getattr(env, "spawn_meta", None),
        cfg,
    )
    spawn_meta = dict(getattr(env, "spawn_meta", None) or {})
    return {
        "world_seed": used,
        "reroll_count": reroll_count,
        "difficulty_unmet": unmet,
        "initial_target_visible": metrics.get("initial_target_visible"),
        "initial_target_distance": metrics.get("initial_target_distance"),
        "initial_target_distance_rays": metrics.get("initial_target_distance_rays"),
        "initial_distance_beyond_range": metrics.get("initial_distance_beyond_range"),
        "_numeric_initial_distance": metrics.get("initial_target_distance_numeric"),
        "spawn_failed": bool(spawn_meta.get("spawn_failed", False)),
        "spawn_attempts": spawn_meta.get("spawn_attempts", 0),
        "spawn_offsets": spawn_meta.get("spawn_offsets") or [],
        "target_spawn_offset": spawn_meta.get("target_spawn_offset"),
        "n_targets_spawned": spawn_meta.get("n_targets_spawned", 0),
        "natural_mobs_cleared": bool(spawn_meta.get("natural_mobs_cleared", False)),
        "vegetation_cleared": bool(spawn_meta.get("vegetation_cleared", False)),
        "spawn_target_distance": metrics.get("spawn_target_distance"),
        "natural_target_closer": bool(metrics.get("natural_target_closer", False)),
        "peaceful_sent": peaceful_command_sent(spawn_meta),
        "peaceful_ok": peaceful_command_sent(spawn_meta),
        "spawn_underground": bool(spawn_meta.get("spawn_underground", False)),
        "underground_rerolls": int(spawn_meta.get("underground_rerolls", 0)),
        "can_see_sky": can_see_sky_of(obs),
    }


def make_scene_env(task_spec, difficulty, seed, cfg, session=None):
    """Build the shared creative16 scene and return ``(env, obs, initial_metrics)``.

    ``env`` has already been reset: spawn, clear-field, freeze, vegetation clear.
    Block-target episodes reroll the world seed until ``min_initial_distance`` holds
    (same loop as the previous ``open_episode``). Pass an ``EnvReuse`` session from
    lfnav eval; MP5 can omit it and this function will create the env.
    """
    from lfnav.eval.run import EnvReuse

    task = as_task_spec(task_spec)
    work = dict(cfg)
    if difficulty is not None:
        work["difficulty"] = difficulty
    own = session is None
    if own:
        session = EnvReuse(work)
        cfg = work
    table = (cfg.get("difficulty_table") or {}).get(cfg.get("difficulty", "easy"), {}) or {}
    min_d = float(table.get("min_initial_distance") or 0)
    max_reroll = int(table.get("max_reroll") or 0)
    distance_reroll = spawn_has_blocks(task) and min_d > 0
    attempts = max_reroll + 1
    env = None
    obs = None
    used = int(seed)
    unmet = False
    reroll_count = 0
    underground_rerolls = 0
    spawn_underground = False
    try:
        for attempt in range(attempts):
            used = world_seed_attempt(seed, attempt)
            if attempt > 0:
                session.release()
            env = session.acquire(task, used)
            obs = env.reset(seed=used, task=task, spawn=False)
            if not can_see_sky_of(obs):
                underground_rerolls += 1
                spawn_underground = True
                continue
            spawn_underground = False
            metrics = apply_spawn_offset_distance(
                initial_scene_metrics(obs, env, cfg, task),
                getattr(env, "spawn_meta", None),
                cfg,
            )
            if not distance_reroll or distance_meets_min(metrics, min_d):
                reroll_count = attempt
                unmet = False
                break
            reroll_count = attempt + 1
        else:
            if spawn_underground:
                underground_rerolls = max_reroll
                reroll_count = max_reroll
                unmet = False
            else:
                unmet = bool(distance_reroll)
                reroll_count = max_reroll
        if not spawn_underground and getattr(env, "spawn_after_reset", False):
            obs = spawn_task_mobs(env, task, cfg, used)
        spawn_meta = dict(getattr(env, "spawn_meta", None) or {})
        spawn_meta["spawn_underground"] = spawn_underground
        spawn_meta["underground_rerolls"] = underground_rerolls
        env.spawn_meta = spawn_meta
        public = _initial_metrics(env, obs, cfg, task, used, reroll_count, unmet)
        public["spawn_underground"] = spawn_underground
        public["underground_rerolls"] = underground_rerolls
        if own:
            session.env = None
            session.key = None
        return env, obs, public
    except Exception:
        if own:
            session.close()
        raise
