"""Record expert trajectories. Successful episodes are the training set.

Failed episodes are stored beside them for analysis. World seeds ending in 9 are
refused here so they can be kept for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lfnav.config import load_config
from lfnav.data.demo_format import (
    config_name,
    find_episode,
    save_decision_frame,
    step_row,
    write_episode,
)
from lfnav.eval.run import EnvReuse, open_episode, run_episode
from lfnav.io import atomic_write_json
from lfnav.perception.gt import GTPerception
from lfnav.policy.expert import Expert, uses_wide_hint, wide_target
from lfnav.types import TaskSpec

BLOCK_DEMO_TARGETS = ("tree", "grass")


def assert_train_seed(seed: int) -> None:
    if int(seed) % 10 == 9:
        raise ValueError(f"seed {seed} is reserved for evaluation")


def skip_demo_cell(task: TaskSpec, difficulty: str) -> bool:
    """Block targets (tree/grass) are not collected on medium/hard."""
    if str(difficulty or "").lower() not in ("medium", "hard"):
        return False
    return str(task.target) in BLOCK_DEMO_TARGETS


def used_wide_for_decision(wide: dict, match: dict | None) -> bool:
    """Same gate as Expert.act: wide hint only when aligned rays miss the target."""
    return uses_wide_hint(wide, match)


class _ExpertLoop:
    """Wrap Expert so run_episode's policy sees the wide-lidar hint."""

    def __init__(self, cfg, task, env):
        self.expert = Expert(cfg)
        self.cfg = cfg
        self.task = task
        self.env = env
        self.used_wide_log: list[bool] = []

    def reset(self):
        self.expert.reset()
        self.used_wide_log.clear()

    def act(self, policy_input):
        obs = self.env.observation()
        wide = wide_target(obs, self.task.target, self.cfg)
        policy_input.memory["wide_target"] = wide
        used = used_wide_for_decision(wide, policy_input.match)
        self.used_wide_log.append(used)
        return self.expert.act(policy_input)


def _write_legacy(out: Path, task: TaskSpec, seed: int, rec: dict, events: list, bucket: str) -> None:
    """Old flat JSON layout. Kept behind cfg demo_legacy_json."""
    atomic_write_json(out / bucket / task.id / f"{seed}.json", {"record": rec, "events": events})
    atomic_write_json(out / task.id / f"{seed}.json", {"bucket": bucket})


def collect(cfg, tasks, seeds, out: Path, backend: str = "fake") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    saved = {"success": 0, "failure": 0, "skipped": 0}
    perception = GTPerception(cfg)
    cfg = dict(cfg)
    cfg["backend"] = backend
    difficulty = str(cfg.get("difficulty") or "easy")
    save_frames = bool(cfg.get("save_decision_frames", False))
    session = EnvReuse(cfg)
    try:
        for task_path in tasks:
            task = TaskSpec.from_dict(json.loads(Path(task_path).read_text()))
            if skip_demo_cell(task, difficulty):
                saved["skipped"] += 1
                continue
            for seed in seeds:
                assert_train_seed(seed)
                if find_episode(out, task.id, difficulty, seed) is not None:
                    saved["skipped"] += 1
                    continue
                try:
                    env, obs, meta = open_episode(session, task, cfg, seed)
                    policy = _ExpertLoop(cfg, task, env)
                    pending = []

                    def on_decision(event, decision_obs, memory):
                        pending.append((event, decision_obs, dict(memory or {})))

                    rec, events = run_episode(
                        env, task, cfg, perception=perception, policy=policy, seed=seed,
                        obs=obs, episode_meta=meta, on_decision=on_decision,
                    )
                    bucket = "success" if rec.get("success") else "failure"
                    rec = dict(rec)
                    rec["config_name"] = config_name(cfg)
                    rec["difficulty"] = difficulty
                    rows = []
                    wide_i = 0
                    for event, decision_obs, memory in pending:
                        used = False
                        if event.get("policy") is not None:
                            if wide_i < len(policy.used_wide_log):
                                used = policy.used_wide_log[wide_i]
                            wide_i += 1
                        rows.append(step_row(event, int(memory.get("env_steps") or 0), memory, used))
                    dest = write_episode(out, bucket, task.id, difficulty, seed, rec, rows)
                    if save_frames and bucket == "success":
                        for event, decision_obs, _memory in pending:
                            save_decision_frame(dest, event.get("decision") or 0, decision_obs, cfg)
                    if cfg.get("demo_legacy_json"):
                        _write_legacy(out, task, seed, rec, events, bucket)
                    saved[bucket] += 1
                except Exception:
                    session.release()
                    raise
                else:
                    session.release(keep=True)
    finally:
        session.close()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=("fake", "minedojo"), default="fake")
    args = parser.parse_args()
    from lfnav.io import read_seeds, read_tasks
    cfg = load_config(args.config) if args.config else load_config()
    cfg["perception_backend"] = "gt"
    if args.difficulty:
        cfg["difficulty"] = args.difficulty
    cfg["judge"] = dict(cfg.get("judge") or {})
    print(collect(cfg, read_tasks(args.tasks), read_seeds(args.seeds), Path(args.out), args.backend))


if __name__ == "__main__":
    main()
