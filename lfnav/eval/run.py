"""Batch episodes for one perception backend and one policy.

Outputs match the MP5_plus shell: one JSON per episode, an events JSONL, run_meta.json,
and summary.csv. --skip-done leaves finished episodes untouched. The original MP5 agent
is not launched; pass --backend mp5 only to record that it belongs in eval/run_batch.py.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from lfnav.config import load_config
from lfnav.control.controller import Controller
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.data.demo_format import config_name
from lfnav.env.minedojo_env import count_hostile_hits, mob_ring_box, peaceful_command_sent
from lfnav.env.scene import make_scene_env
from lfnav.gt.initial import (
    apply_spawn_offset_distance,
    initial_scene_metrics,
    privileged_ray_distance,
)
from lfnav.gt.judge import aligned_target_distance, judge_episode
from lfnav.gt.scene_state import scene_state_from_obs
from lfnav.io import atomic_write_json, json_default, read_seeds, read_tasks
from lfnav.matcher import Matcher
from lfnav.memory import Memory
from lfnav.obsutil import position_of, yaw_of
from lfnav.perception.gt import GTPerception
from lfnav.policy.c0 import RandomWalk
from lfnav.policy.c1 import RuleBased
from lfnav.policy.joint import belief_aligned_distance
from lfnav.types import PolicyInput, TaskSpec


def git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except Exception:
        return None


def _joint_agent(cfg: dict):
    if cfg.get("_joint_agent") is None:
        from lfnav.policy.joint import JointAgent

        cfg["_joint_agent"] = JointAgent(cfg)
    return cfg["_joint_agent"]


def make_perception(cfg: dict):
    if cfg.get("policy_name") == "joint" or cfg.get("perception_backend") == "joint":
        cfg["perception_backend"] = "joint"
        return _joint_agent(cfg)
    name = cfg.get("perception_backend", "gt")
    if name == "gt":
        return GTPerception(cfg)
    if name == "minellm":
        from lfnav.perception.minellm import MineLLMPerception
        return MineLLMPerception(cfg)
    if name in ("dinov2", "learned", "mineclip"):
        from lfnav.perception.learned import LearnedPerception
        return LearnedPerception(cfg)
    raise KeyError(name)


def make_policy(cfg: dict):
    name = cfg.get("policy_name", "c1")
    if name == "joint":
        cfg["perception_backend"] = "joint"
        return _joint_agent(cfg)
    if name in ("c0", "random"):
        return RandomWalk(cfg)
    if name in ("c1", "rule"):
        return RuleBased(cfg)
    if name in ("c2", "bc"):
        from lfnav.policy.c2 import BehaviorCloning
        return BehaviorCloning(cfg)
    if name == "c3":
        from lfnav.policy.c3 import DiffusionPolicy
        return DiffusionPolicy(cfg)
    if name == "mp5":
        raise RuntimeError(
            "Original MP5 is not run from lfnav. Use eval/run_batch.py with the same task specs, seeds, and lfnav judge once an API key is available."
        )
    raise KeyError(name)


def match_distance(perception, obs, cfg: dict, task, scene):
    if getattr(perception, "name", None) == "joint":
        return belief_aligned_distance(scene, task.target, cfg)
    return aligned_target_distance(obs, cfg, task.target)


def make_env(cfg: dict, task: TaskSpec, seed: int):
    backend = cfg.get("backend", "fake")
    if backend == "fake":
        return FakeMineDojoEnv(cfg, layout_for(task, cfg, seed))
    if backend == "minedojo":
        from lfnav.env.minedojo_env import MineDojoEnv
        return MineDojoEnv(cfg, task, seed)
    raise KeyError(backend)


def layout_for(task: TaskSpec, cfg: dict, seed: int) -> dict:
    import random
    rng = random.Random(seed)
    table = (cfg.get("difficulty_table") or {}).get(cfg.get("difficulty", "easy"), {})
    lo = float(table.get("spawn_distance_min", 4))
    hi = float(table.get("spawn_distance_max", 8))
    behind = str(cfg.get("difficulty") or "") == "hard"
    low, high = mob_ring_box(lo, hi, seed, behind=behind)
    cx = 0.5 * (low[0] + high[0])
    cz = 0.5 * (low[2] + high[2])
    entities = []
    blocks = {}
    for item in task.episode.get("spawn") or []:
        if item.get("kind") == "mob":
            entities.append({"name": item["name"], "x": cx, "z": cz})
        else:
            blocks[(int(round(cx)), int(round(cz)))] = item["name"]
    for i in range(int(table.get("distractors", 0))):
        entities.append({"name": "sheep", "x": rng.uniform(-6, 6), "z": rng.uniform(2, hi)})
    return {
        "pos": [0.0, 64.0, 0.0],
        "yaw": 0.0,
        "entities": entities,
        "blocks": blocks,
        "biome": task.episode.get("specified_biome") or "plains",
        "time": task.episode.get("start_time") or "day",
        "weather": task.episode.get("initial_weather") or "clear",
        "brightness": "sufficient",
        "can_see_sky": True,
    }


def _hide_wide(obs: dict, cfg: dict) -> dict:
    if cfg.get("policy_sees_wide_lidar", False):
        return obs
    keys = cfg.get("obs_keys") or {}
    wide_key = keys.get("wide_rays")
    if isinstance(obs, dict) and wide_key:
        obs = dict(obs)
        obs[wide_key] = []
    return obs


def open_episode(session, task: TaskSpec, cfg: dict, seed: int):
    """Make/reset until min_initial_distance holds, then return obs before the first decision."""
    return make_scene_env(task, cfg.get("difficulty"), seed, cfg, session=session)


def _recorder(env):
    return getattr(env, "frames", None)


def run_episode(env, task: TaskSpec, cfg: dict, perception=None, policy=None, seed: int = 0, obs=None, episode_meta=None, on_decision=None) -> tuple[dict, list]:
    perception = perception or make_perception(cfg)
    policy = policy or make_policy(cfg)
    if hasattr(perception, "reset"):
        perception.reset()
    policy.reset()
    controller = Controller(env, cfg)
    matcher = Matcher(cfg)
    pol = cfg.get("policy") or {}
    if getattr(perception, "name", None) == "joint":
        memory = Memory(int(perception.memory_size), int(perception.spec.recent_k))
    else:
        memory = Memory(pol.get("memory_size", 15), pol.get("recent_k", 8))
    judge_cfg = cfg.get("judge") or {}
    max_steps = int(judge_cfg.get("max_steps", 3000))
    radius = float(judge_cfg.get("radius", 3))
    wall0 = time.perf_counter()
    if obs is None:
        obs = env.reset(seed=seed, task=task)
    rec = _recorder(env)
    if rec is not None:
        rec.capture(obs, tag="start", keyframe=True)
    meta = dict(episode_meta or {})
    if meta.get("spawn_underground"):
        pos = position_of(obs, cfg) if obs is not None else None
        verdict = {
            "sr_a": False,
            "sr_b": False,
            "false_stop": False,
            "timeout": False,
            "missing": ["spawn_underground"],
            "reason": "spawn_underground",
            "distance_blocks": None,
            "path_length": 0.0,
            "straight": 0.0,
            "spl": None,
            "outcome": "spawn_underground",
            "success": False,
            "steps": 0,
            "start": pos,
            "stop": pos,
            "task_name": task.id,
            "task_kind": "lfnav",
            "seed": seed,
            "elapsed_s": 0.0,
            "frames_s": 0.0,
            "frames_overhead": 0.0,
            "failure_category": "spawn_underground",
            "self_reported_steps": 0,
            "perception_backend": cfg.get("perception_backend", "gt"),
            "policy_name": cfg.get("policy_name", "c1"),
            "difficulty": cfg.get("difficulty"),
            "stop_mode": (cfg.get("match") or {}).get("stop_mode", "sr_b"),
            "gt_at_stop": None,
            "perception_at_stop": None,
            "world_seed": meta.get("world_seed", seed),
            "reroll_count": meta.get("reroll_count", 0),
            "difficulty_unmet": False,
            "initial_target_distance": meta.get("initial_target_distance"),
            "initial_target_visible": meta.get("initial_target_visible"),
            "task": task.id,
            "policy": cfg.get("policy_name", "c1"),
            "target_never_seen": True,
            "config_name": config_name(cfg),
            "spawn_underground": True,
            "underground_rerolls": int(meta.get("underground_rerolls") or 0),
            "can_see_sky": False,
            "hostile_seen": 0,
            "peaceful_sent": peaceful_command_sent(meta),
            "peaceful_ok": peaceful_command_sent(meta),
        }
        return verdict, []
    if "initial_target_distance" not in meta:
        metrics = apply_spawn_offset_distance(
            initial_scene_metrics(obs, env, cfg, task),
            getattr(env, "spawn_meta", None),
            cfg,
        )
        meta.update({
            "world_seed": seed,
            "reroll_count": 0,
            "difficulty_unmet": False,
            "initial_target_visible": metrics.get("initial_target_visible"),
            "initial_target_distance": metrics.get("initial_target_distance"),
            "initial_target_distance_rays": metrics.get("initial_target_distance_rays"),
            "initial_distance_beyond_range": metrics.get("initial_distance_beyond_range"),
            "_numeric_initial_distance": metrics.get("initial_target_distance_numeric"),
            "spawn_target_distance": metrics.get("spawn_target_distance"),
            "natural_target_closer": bool(metrics.get("natural_target_closer", False)),
        })
    seen_gt = privileged_ray_distance(obs, cfg, task.target) is not None
    hostile_seen = count_hostile_hits(obs, cfg)
    obs = _hide_wide(obs, cfg)
    start = position_of(obs, cfg)
    prev = start
    path = 0.0
    env_steps = 0
    events = []
    stopped = False
    scene = None
    last_stuck = False
    while env_steps < max_steps:
        t0 = time.perf_counter()
        scene = perception.perceive(obs)
        t1 = time.perf_counter()
        match = matcher.match(task, scene, aligned_distance=match_distance(perception, obs, cfg, task, scene))
        t2 = time.perf_counter()
        pos = position_of(obs, cfg)
        yaw = yaw_of(obs, cfg)
        mem = memory.to_dict(pos, yaw)
        mem["stuck"] = last_stuck
        mem["blocked"] = last_stuck
        mem["env_steps"] = env_steps
        gt_frame = scene_state_from_obs(obs, cfg)
        if rec is not None:
            rec.set_info(match.to_dict())
        if match.stop:
            if rec is not None:
                rec.set_info(match.to_dict(), action="stop")
                rec.capture(obs, tag="stop", keyframe=True)
            row = {
                "decision": len(events),
                "scene": scene.to_dict(),
                "perception": scene.to_dict(),
                "gt": gt_frame.to_dict(),
                "match": match.to_dict(),
                "policy": None,
                "control": None,
                "seconds": {"perceive": t1 - t0, "match": t2 - t1, "policy": 0.0},
            }
            events.append(row)
            if on_decision is not None:
                on_decision(row, obs, mem)
            stopped = True
            break
        pin = PolicyInput(perception.features(obs), task.to_dict(), match.to_dict(), scene.to_dict(), mem, list(memory.recent_actions))
        decision_obs = obs
        output = policy.act(pin)
        t3 = time.perf_counter()
        control_rows = []
        for action in output.action_chunk:
            remain = max_steps - env_steps
            if remain <= 0:
                break
            if rec is not None:
                rec.set_info(match.to_dict(), action=action)
            result = controller.execute(action, budget=remain)
            if privileged_ray_distance(result.obs, cfg, task.target) is not None:
                seen_gt = True
            hostile_seen += count_hostile_hits(result.obs, cfg)
            result.obs = _hide_wide(result.obs, cfg)
            env_steps += result.env_steps
            new_pos = position_of(result.obs, cfg)
            if prev is not None and new_pos is not None:
                path += math.hypot(new_pos[0] - prev[0], new_pos[-1] - prev[-1])
            prev = new_pos or prev
            skip_stuck = bool((cfg.get("policy") or {}).get("skip_visits_when_stuck", True))
            memory.observe(
                new_pos,
                yaw_of(result.obs, cfg),
                action,
                match.to_dict(),
                increment_visits=not (skip_stuck and result.stuck),
            )
            last_stuck = result.stuck
            obs = result.obs
            control_rows.append(result.to_dict())
            if env_steps >= max_steps:
                break
        row = {
            "decision": len(events),
            "scene": scene.to_dict(),
            "perception": scene.to_dict(),
            "gt": gt_frame.to_dict(),
            "match": match.to_dict(),
            "policy": output.to_dict(),
            "control": control_rows,
            "seconds": {"perceive": t1 - t0, "match": t2 - t1, "policy": t3 - t2},
        }
        events.append(row)
        if on_decision is not None:
            on_decision(row, decision_obs, mem)
    if rec is not None:
        rec.capture(obs, tag="end", keyframe=True)
    live = env.observation() if hasattr(env, "observation") else obs
    if privileged_ray_distance(live, cfg, task.target) is not None:
        seen_gt = True
    stop_pos = position_of(obs, cfg)
    distance = aligned_target_distance(obs, cfg, task.target)
    gt_scene = scene_state_from_obs(obs, cfg)
    verdict = judge_episode(
        task, stopped, env_steps, gt_scene, distance, path, start, stop_pos,
        max_steps=max_steps, radius=radius,
        initial_target_distance=meta.get("initial_target_distance"),
    )
    verdict.update({
        "task_name": task.id,
        "task_kind": "lfnav",
        "seed": seed,
        "elapsed_s": float(time.perf_counter() - wall0),
        "frames_s": 0.0 if rec is None else rec.elapsed_s,
        "frames_overhead": (
            0.0 if rec is None or (time.perf_counter() - wall0) <= 0
            else rec.elapsed_s / max(time.perf_counter() - wall0, 1e-9)
        ),
        "failure_category": None if verdict.get("success") else verdict.get("outcome"),
        "self_reported_steps": env_steps,
        "perception_backend": cfg.get("perception_backend", "gt"),
        "policy_name": cfg.get("policy_name", "c1"),
        "difficulty": cfg.get("difficulty"),
        "stop_mode": (cfg.get("match") or {}).get("stop_mode", "sr_b"),
        "gt_at_stop": gt_scene.to_dict(),
        "perception_at_stop": None if scene is None else scene.to_dict(),
        "world_seed": meta.get("world_seed", seed),
        "reroll_count": meta.get("reroll_count", 0),
        "difficulty_unmet": bool(meta.get("difficulty_unmet", False)),
        "initial_target_visible": meta.get("initial_target_visible"),
        "initial_target_distance": meta.get("initial_target_distance"),
        "initial_distance_beyond_range": bool(meta.get("initial_distance_beyond_range", False)),
        "spawn_failed": bool(meta.get("spawn_failed", False)),
        "spawn_attempts": meta.get("spawn_attempts", 0),
        "spawn_offsets": meta.get("spawn_offsets") or [],
        "target_spawn_offset": meta.get("target_spawn_offset"),
        "n_targets_spawned": meta.get("n_targets_spawned", 0),
        "natural_mobs_cleared": bool(meta.get("natural_mobs_cleared", False)),
        "task": task.id,
        "policy": cfg.get("policy_name", "c1"),
        "target_never_seen": not seen_gt,
        "initial_target_distance_rays": meta.get("initial_target_distance_rays"),
        "vegetation_cleared": bool(meta.get("vegetation_cleared", False)),
        "n_stuck": int(getattr(controller, "n_stuck", 0)),
        "n_escape": int(getattr(controller, "n_escape", 0)),
        "stuck_steps_total": int(getattr(controller, "stuck_steps_total", 0)),
        "spawn_target_distance": meta.get("spawn_target_distance"),
        "natural_target_closer": bool(meta.get("natural_target_closer", False)),
        "config_name": config_name(cfg),
        "peaceful_sent": peaceful_command_sent(meta),
        "peaceful_ok": peaceful_command_sent(meta),
        "hostile_seen": int(hostile_seen),
        "spawn_underground": bool(meta.get("spawn_underground", False)),
        "underground_rerolls": int(meta.get("underground_rerolls") or 0),
        "can_see_sky": bool(meta.get("can_see_sky", True)),
    })
    return verdict, events


def write_summary(out: Path) -> None:
    import csv
    rows = []
    fields = [
        "task_name", "seed", "task_kind", "outcome", "success", "steps", "elapsed_s",
        "failure_category", "self_reported_steps", "sr_a", "sr_b", "false_stop", "timeout", "spl",
        "difficulty", "world_seed", "reroll_count", "difficulty_unmet",
        "initial_target_visible", "initial_target_distance", "initial_target_distance_rays",
        "spawn_failed", "target_never_seen", "natural_mobs_cleared", "n_targets_spawned",
        "vegetation_cleared", "n_stuck", "stuck_steps_total",
    ]
    for path in sorted(out.glob("*/*.json")):
        if path.name == "run_meta.json":
            continue
        rec = json.loads(path.read_text())
        if "seed" not in rec:
            continue
        rows.append({k: rec.get(k) for k in fields})
    with open(out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def close_env(env) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        close()


class EnvReuse:
    """One environment per episode, unless reuse_env keeps a task and difficulty together.

    With reuse off, make() receives that episode's seed as the world seed.
    With reuse on, make() runs for the first episode of a (task, difficulty) streak
    and later episodes only reset(), so the world stays the first seed's world.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reuse = bool(cfg.get("reuse_env", False))
        self.env = None
        self.key = None

    def acquire(self, task: TaskSpec, seed: int):
        key = (task.id, self.cfg.get("difficulty"))
        if self.reuse and self.env is not None and self.key == key:
            return self.env
        self.release()
        self.env = make_env(self.cfg, task, seed)
        self.key = key
        return self.env

    def release(self, keep: bool = False) -> None:
        if keep and self.reuse and self.env is not None:
            return
        close_env(self.env)
        self.env = None
        self.key = None

    def close(self) -> None:
        self.release()


def run_batch(cfg: dict, tasks: list[str], seeds: list[int], out: Path, skip_done: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "run_meta.json", {
        "config": {k: v for k, v in cfg.items() if not str(k).endswith("_table") and k != "labels"},
        "git_commit": git_hash(),
        "argv": sys.argv,
        "tasks": tasks,
        "seeds": seeds,
    })
    session = EnvReuse(cfg)
    try:
        for task_path in tasks:
            task = TaskSpec.from_dict(json.loads(Path(task_path).read_text()))
            for seed in seeds:
                dest = out / task.id / f"{seed}.json"
                event_path = out / task.id / str(seed) / "events.jsonl"
                if skip_done and dest.is_file():
                    continue
                env = None
                try:
                    env, obs, meta = open_episode(session, task, cfg, seed)
                    frames_dir = out / task.id / str(seed) / "frames"
                    env_rec = _recorder(env)
                    if env_rec is not None:
                        env_rec.begin_episode(frames_dir, task.id, seed)
                    rec, events = run_episode(env, task, cfg, seed=seed, obs=obs, episode_meta=meta)
                    atomic_write_json(dest, rec)
                    event_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = event_path.with_suffix(".jsonl.tmp")
                    with tmp.open("w") as handle:
                        for event in events:
                            handle.write(json.dumps(event, default=json_default) + "\n")
                    tmp.replace(event_path)
                except Exception:
                    try:
                        env_rec = _recorder(env) if env is not None else None
                        if env_rec is not None:
                            env_rec.capture(getattr(env, "observation", lambda: {})(), tag="end", keyframe=True)
                    except Exception:
                        pass
                    session.release()
                    raise
                else:
                    session.release(keep=True)
    finally:
        session.close()
    write_summary(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--perception", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--frames-every", type=int, default=None)
    parser.add_argument("--no-frames", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.perception:
        cfg["perception_backend"] = args.perception
    if args.policy:
        cfg["policy_name"] = args.policy
    if args.backend:
        cfg["backend"] = args.backend
    if args.difficulty:
        cfg["difficulty"] = args.difficulty
    cfg["frames"] = dict(cfg.get("frames") or {})
    if args.no_frames:
        cfg["frames"]["enabled"] = False
    if args.frames_every is not None:
        cfg["frames"]["enabled"] = True
        cfg["frames"]["every"] = int(args.frames_every)
    run_batch(cfg, read_tasks(args.tasks), read_seeds(args.seeds), Path(args.out), args.skip_done)


if __name__ == "__main__":
    main()
