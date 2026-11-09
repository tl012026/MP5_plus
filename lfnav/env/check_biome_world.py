"""Walk 100 blocks from spawn in four directions and record biome_id.

Run inside MineDojo (sbatch). Prints whether specified_biome worlds look like
a single biome. Do not import this on the login node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lfnav.config import load_config
from lfnav.env.minedojo_env import MineDojoEnv, _kwargs
from lfnav.obsutil import lookup, scalar
from lfnav.types import TaskSpec


def _biome_id(obs, cfg) -> int | None:
    keys = cfg.get("obs_keys") or {}
    loc = lookup(obs, keys.get("location"), {}) or {}
    value = loc.get("biome_id") if isinstance(loc, dict) else None
    if value is None:
        return None
    try:
        return int(scalar(value))
    except (TypeError, ValueError):
        return None


def walk(env, cfg, steps: int, action: str) -> list[int | None]:
    ids = []
    for _ in range(steps):
        obs, _r, _d, _i = env.step(action)
        ids.append(_biome_id(obs, cfg))
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--biome", default="forest")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["backend"] = "minedojo"
    task = TaskSpec(
        id="biome-probe",
        source="check_biome_world",
        description="biome walk",
        target="tree",
        episode={"specified_biome": args.biome, "start_time": "day", "initial_weather": "clear", "spawn": []},
    )
    env = MineDojoEnv(cfg, task, args.seed)
    try:
        obs = env.reset()
        origin = _biome_id(obs, cfg)
        report = {
            "kwargs_biome": _kwargs(cfg, args.seed, task).get("specified_biome"),
            "origin_biome_id": origin,
            "steps": args.steps,
            "walks": {},
        }
        changed = False
        headings = ["north", "east", "south", "west"]
        for heading in headings:
            env.close()
            env = MineDojoEnv(cfg, task, args.seed)
            env.reset()
            yaw_needed = {"north": 0, "east": 1, "south": 2, "west": 3}[heading]
            for _ in range(yaw_needed * 2):
                env.step("turn_right_45")
            ids = walk(env, cfg, args.steps, "forward")
            unique = sorted({i for i in ids if i is not None})
            report["walks"][heading] = {"unique_biome_ids": unique, "n": len(ids)}
            if origin is not None and any(i != origin for i in ids if i is not None):
                changed = True
        report["single_biome"] = not changed
        print(json.dumps(report, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        if report["single_biome"]:
            print("specified_biome looks like a single-biome world; leave_biome cannot be implemented this way.")
        else:
            print("biome_id changed with distance; leave_biome is possible.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
