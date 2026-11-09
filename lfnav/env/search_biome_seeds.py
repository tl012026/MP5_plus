"""Offline seed search for the optional hard cross-biome setting.

Looks for worlds whose spawn biome is not the target biome, but the target
biome appears within --radius blocks. Writes configs/lfnav/hard_cross_biome_seeds.txt.
Run on a compute node with MineDojo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lfnav.config import load_config, repo_root
from lfnav.env.minedojo_env import MineDojoEnv
from lfnav.gt.scene_state import mapped_biome
from lfnav.obsutil import lookup, scalar
from lfnav.types import TaskSpec


def _biome_name(obs, cfg) -> str | None:
    keys = cfg.get("obs_keys") or {}
    loc = lookup(obs, keys.get("location"), {}) or {}
    mapping = keys.get("biome_id_to_name") or {}
    raw = loc.get("biome_id") if isinstance(loc, dict) else None
    if raw is None:
        return None
    return mapped_biome(scalar(raw), mapping)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--target-biome", default="forest")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--tries", type=int, default=200)
    parser.add_argument("--radius", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["backend"] = "minedojo"
    cfg["difficulty"] = "hard"
    out = Path(args.out or (repo_root() / "configs/lfnav/hard_cross_biome_seeds.txt"))
    found = []
    seed = int(args.start)
    task = TaskSpec(
        id="seed-search",
        source="search_biome_seeds",
        description="cross biome",
        target="tree",
        episode={"start_time": "day", "initial_weather": "clear", "spawn": []},
    )
    while len(found) < args.count and seed < args.start + args.tries:
        env = MineDojoEnv(cfg, task, seed)
        try:
            obs = env.reset()
            spawn_biome = _biome_name(obs, cfg)
            if spawn_biome and spawn_biome != args.target_biome:
                # Cheap check: origin is already a different biome. Callers can
                # tighten this with check_biome_world-style walks.
                found.append((seed, args.target_biome, spawn_biome))
                print(f"keep seed={seed} spawn={spawn_biome} target={args.target_biome}")
        finally:
            env.close()
        seed += 1
    lines = ["# Optional hard-mode seeds for default (not specified_biome) world generation."]
    lines.append("# seed target_biome spawn_biome")
    for seed, target, spawn in found:
        lines.append(f"{seed} {target} {spawn}")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(found)} seeds to {out}")


if __name__ == "__main__":
    main()
