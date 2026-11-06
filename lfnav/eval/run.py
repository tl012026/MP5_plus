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
from lfnav.gt.judge import aligned_target_distance, judge_episode
from lfnav.gt.scene_state import scene_state_from_obs
from lfnav.io import atomic_write_json, read_seeds, read_tasks
from lfnav.matcher import Matcher
from lfnav.memory import Memory
from lfnav.obsutil import position_of, yaw_of
from lfnav.perception.gt import GTPerception
from lfnav.policy.c0 import RandomWalk
from lfnav.policy.c1 import RuleBased
from lfnav.types import PolicyInput, TaskSpec


def git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except Exception:
        return None


def make_perception(cfg: dict):
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
    entities = []
    blocks = {}
    for item in task.episode.get("spawn") or []:
        dist = rng.uniform(lo, hi)
        if item.get("kind") == "mob":
            entities.append({"name": item["name"], "x": 0.0, "z": dist})
        else:
            blocks[(0, int(round(dist)))] = item["name"]
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
    }


def run_episode(env, task: TaskSpec, cfg: dict, perception=None, policy=None, seed: int = 0) -> tuple[dict, list]:
    perception = perception or make_perception(cfg)
    policy = policy or make_policy(cfg)
    if hasattr(perception, "reset"):
        perception.reset()
    policy.reset()
    controller = Controller(env, cfg)
    matcher = Matcher(cfg)
    pol = cfg.get("policy") or {}
    memory = Memory(pol.get("memory_size", 15), pol.get("recent_k", 8))
    judge_cfg = cfg.get("judge") or {}
    max_steps = int(judge_cfg.get("max_steps", 3000))
    radius = float(judge_cfg.get("radius", 3))
    obs = env.reset(seed=seed, task=task)
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
        match = matcher.match(task, scene)
        t2 = time.perf_counter()
        pos = position_of(obs, cfg)
        yaw = yaw_of(obs, cfg)
        mem = memory.to_dict(pos, yaw)
        mem["stuck"] = last_stuck
        mem["blocked"] = last_stuck
        gt_frame = scene_state_from_obs(obs, cfg)
        if match.stop:
            events.append({
                "decision": len(events),
                "scene": scene.to_dict(),
                "perception": scene.to_dict(),
                "gt": gt_frame.to_dict(),
                "match": match.to_dict(),
                "policy": None,
                "control": None,
                "seconds": {"perceive": t1 - t0, "match": t2 - t1, "policy": 0.0},
            })
            stopped = True
            break
        pin = PolicyInput(perception.features(obs), task.to_dict(), match.to_dict(), scene.to_dict(), mem, list(memory.recent_actions))
        output = policy.act(pin)
        t3 = time.perf_counter()
        control_rows = []
        for action in output.action_chunk:
            result = controller.execute(action)
            env_steps += result.env_steps
            new_pos = position_of(result.obs, cfg)
            if prev is not None and new_pos is not None:
                path += math.hypot(new_pos[0] - prev[0], new_pos[-1] - prev[-1])
            prev = new_pos or prev
            memory.observe(new_pos, yaw_of(result.obs, cfg), action, match.to_dict())
            last_stuck = result.stuck
            obs = result.obs
            control_rows.append(result.to_dict())
            if env_steps >= max_steps:
                break
        events.append({
            "decision": len(events),
            "scene": scene.to_dict(),
            "perception": scene.to_dict(),
            "gt": gt_frame.to_dict(),
            "match": match.to_dict(),
            "policy": output.to_dict(),
            "control": control_rows,
            "seconds": {"perceive": t1 - t0, "match": t2 - t1, "policy": t3 - t2},
        })
    stop_pos = position_of(obs, cfg)
    distance = aligned_target_distance(obs, cfg, task.target)
    gt_scene = scene_state_from_obs(obs, cfg)
    verdict = judge_episode(
        task, stopped, env_steps, gt_scene, distance, path, start, stop_pos,
        max_steps=max_steps, radius=radius,
    )
    verdict.update({
        "task_name": task.id,
        "task_kind": "lfnav",
        "seed": seed,
        "elapsed_s": sum(sum(ev["seconds"].values()) for ev in events),
        "failure_category": None if verdict.get("success") else verdict.get("outcome"),
        "self_reported_steps": env_steps,
        "perception_backend": cfg.get("perception_backend", "gt"),
        "policy_name": cfg.get("policy_name", "c1"),
        "difficulty": cfg.get("difficulty"),
        "stop_mode": (cfg.get("match") or {}).get("stop_mode", "sr_b"),
        "gt_at_stop": gt_scene.to_dict(),
        "perception_at_stop": None if scene is None else scene.to_dict(),
    })
    return verdict, events


def write_summary(out: Path) -> None:
    import csv
    rows = []
    fields = [
        "task_name", "seed", "task_kind", "outcome", "success", "steps", "elapsed_s",
        "failure_category", "self_reported_steps", "sr_a", "sr_b", "false_stop", "timeout", "spl",
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
                try:
                    env = session.acquire(task, seed)
                    rec, events = run_episode(env, task, cfg, seed=seed)
                    atomic_write_json(dest, rec)
                    event_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = event_path.with_suffix(".jsonl.tmp")
                    with tmp.open("w") as handle:
                        for event in events:
                            handle.write(json.dumps(event) + "\n")
                    tmp.replace(event_path)
                except Exception:
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
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.perception:
        cfg["perception_backend"] = args.perception
    if args.policy:
        cfg["policy_name"] = args.policy
    if args.backend:
        cfg["backend"] = args.backend
    run_batch(cfg, read_tasks(args.tasks), read_seeds(args.seeds), Path(args.out), args.skip_done)


if __name__ == "__main__":
    main()
