"""Record expert trajectories. Successful episodes are the training set.

Failed episodes are stored beside them for analysis. World seeds ending in 9 are
refused here so they can be kept for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lfnav.config import load_config
from lfnav.eval.run import EnvReuse, run_episode
from lfnav.io import atomic_write_json
from lfnav.perception.gt import GTPerception
from lfnav.policy.expert import Expert, wide_target
from lfnav.types import TaskSpec


def assert_train_seed(seed: int) -> None:
    if int(seed) % 10 == 9:
        raise ValueError(f"seed {seed} is reserved for evaluation")


class _ExpertLoop:
    """Wrap Expert so run_episode's policy sees the wide-lidar hint."""

    def __init__(self, cfg, task, env):
        self.expert = Expert(cfg)
        self.cfg = cfg
        self.task = task
        self.env = env

    def reset(self):
        self.expert.reset()

    def act(self, policy_input):
        obs = self.env.observation()
        policy_input.memory["wide_target"] = wide_target(obs, self.task.target, self.cfg)
        return self.expert.act(policy_input)


def collect(cfg, tasks, seeds, out: Path, backend: str = "fake") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    saved = {"success": 0, "failure": 0}
    perception = GTPerception(cfg)
    cfg = dict(cfg)
    cfg["backend"] = backend
    session = EnvReuse(cfg)
    try:
        for task_path in tasks:
            task = TaskSpec.from_dict(json.loads(Path(task_path).read_text()))
            for seed in seeds:
                assert_train_seed(seed)
                marker = out / task.id / f"{seed}.json"
                if marker.is_file():
                    continue
                try:
                    env = session.acquire(task, seed)
                    policy = _ExpertLoop(cfg, task, env)
                    rec, events = run_episode(env, task, cfg, perception=perception, policy=policy, seed=seed)
                    bucket = "success" if rec.get("success") else "failure"
                    saved[bucket] += 1
                    atomic_write_json(out / bucket / task.id / f"{seed}.json", {"record": rec, "events": events})
                    atomic_write_json(marker, {"bucket": bucket})
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
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=("fake", "minedojo"), default="fake")
    args = parser.parse_args()
    from lfnav.io import read_seeds, read_tasks
    cfg = load_config(args.config) if args.config else load_config()
    cfg["perception_backend"] = "gt"
    cfg["judge"] = dict(cfg.get("judge") or {})
    cfg["judge"]["max_steps"] = int(cfg["judge"].get("max_steps", 40))
    print(collect(cfg, read_tasks(args.tasks), read_seeds(args.seeds), Path(args.out), args.backend))


if __name__ == "__main__":
    main()
