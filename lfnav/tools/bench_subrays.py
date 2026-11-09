"""Compare aligned_subrays n=1,2,3: step rate and pig hits while turning in place.

Uses one world seed (9). Default backend is fake so it runs on the login node.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lfnav.config import load_config
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.gt.judge import aligned_target_distance


def _cfg(n: int, reach: float) -> dict:
    cfg = load_config()
    cfg["aligned_subrays"] = int(n)
    cfg["aligned_ray_range"] = float(reach)
    cfg["backend"] = "fake"
    cfg["wide_lidar"] = True
    return cfg


def _env(cfg: dict, dist: float | None, seed: int) -> FakeMineDojoEnv:
    layout = {"pos": [0.0, 64.0, 0.0], "yaw": 0.0, "biome": "plains"}
    if dist is not None:
        layout["entities"] = [{"name": "pig", "x": 0.0, "z": float(dist), "radius": 0.4}]
    env = FakeMineDojoEnv(cfg, layout)
    env.reset(seed=seed)
    return env


def bench_steps(cfg: dict, seed: int, n_steps: int) -> float:
    env = _env(cfg, None, seed)
    vectors = (cfg.get("actions") or {}).get("vectors") or {}
    action = vectors.get("turn_right_15", "turn_right_15")
    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step(action)
    elapsed = max(time.perf_counter() - t0, 1e-9)
    env.close()
    return n_steps / elapsed


def scan_hits(cfg: dict, dist: float, seed: int) -> tuple[int, int]:
    env = _env(cfg, dist, seed)
    vectors = (cfg.get("actions") or {}).get("vectors") or {}
    action = vectors.get("turn_right_15", "turn_right_15")
    hits = 0
    headings = 24
    for _ in range(headings):
        obs = env.observation()
        if aligned_target_distance(obs, cfg, "pig") is not None:
            hits += 1
        env.step(action)
    env.close()
    return hits, headings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--dists", type=float, nargs="+", default=[14.0, 25.0])
    args = parser.parse_args()
    reach = max(args.dists) + 8.0
    print(f"seed={args.seed} steps={args.steps}")
    for n in args.ns:
        cfg = _cfg(n, reach)
        rate = bench_steps(cfg, args.seed, args.steps)
        print(f"n={n}  steps/s={rate:.2f}")
    print("in-place scan (24 × 15°), frozen pig")
    for dist in args.dists:
        parts = [f"dist={dist:g}"]
        for n in args.ns:
            cfg = _cfg(n, reach)
            hits, total = scan_hits(cfg, dist, args.seed)
            parts.append(f"n={n}:{hits}/{total}")
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    main()
