"""Probe /difficulty peaceful, /summon, and resistance NBT inside the MineDojo image.

Exits non-zero if any check fails. Host Python is not enough; run via probe_commands.sbatch.
"""

from __future__ import annotations

import argparse
import sys
import time

from lfnav.config import load_config
from lfnav.env.minedojo_env import (
    RESISTANCE_NBT,
    MineDojoEnv,
    _send_chat,
    count_hostile_hits,
)
from lfnav.gt.initial import privileged_ray_distance
from lfnav.types import TaskSpec


def _tick(env, n: int):
    obs = env.observation()
    for _ in range(int(n)):
        obs, _r, _d, _i = env.step("noop")
    return obs


def _fail(name: str, detail: str) -> str:
    print(f"FAIL {name}: {detail}")
    return name


def run_probes(cfg: dict) -> list[str]:
    failed = []
    task = TaskSpec(id="probe", source="", description="probe", target="pig", conditions={}, episode={
        "specified_biome": "plains",
        "start_time": "day",
        "initial_weather": "clear",
        "spawn": [],
    })
    env = MineDojoEnv(cfg, task, seed=3)
    try:
        env.reset(task=task, seed=3)
        t0 = time.perf_counter()
        try:
            _send_chat(env, "/difficulty peaceful")
            _send_chat(env, "/summon zombie ~ ~ ~4")
            obs = _tick(env, 20)
            hits = count_hostile_hits(obs, cfg)
            if hits > 0:
                failed.append(_fail("peaceful", f"zombie still on rays after 20 ticks ({hits})"))
            else:
                print("PASS peaceful")
        except Exception as exc:
            failed.append(_fail("peaceful", str(exc)))

        try:
            _send_chat(env, "/summon pig ~ ~1 ~4")
            obs = _tick(env, 20)
            dist = privileged_ray_distance(obs, cfg, "pig")
            if dist is None:
                failed.append(_fail("summon", "pig not on rays after 20 ticks"))
            else:
                print(f"PASS summon pig_distance={dist}")
        except Exception as exc:
            failed.append(_fail("summon", str(exc)))

        try:
            _send_chat(env, f"/summon pig ~ ~1 ~6 {RESISTANCE_NBT}")
            print("PASS resistance_nbt")
        except Exception as exc:
            failed.append(_fail("resistance_nbt", str(exc)))
        print(f"probe_s={time.perf_counter() - t0:.1f}")
    finally:
        env.close()
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lfnav/main.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["backend"] = "minedojo"
    failed = run_probes(cfg)
    if failed:
        print("FAILED:", ",".join(failed))
        sys.exit(1)
    print("ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
