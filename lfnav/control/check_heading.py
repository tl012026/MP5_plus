"""Check voxel (dx, dz) against real MineDojo motion. Run in the image.

    xvfb-run -a python lfnav/control/check_heading.py --out $SCR/lfnav/calib/heading.json

Turns to yaw 0, 90, 180, 270, walks forward, and compares the position delta
to lfnav.control.controller._front_delta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lfnav.config import load_config
from lfnav.control.controller import _front_delta
from lfnav.eval.run import close_env
from lfnav.io import atomic_write_json
from lfnav.obsutil import position_of, yaw_of


def _wrap(err: float) -> float:
    return (float(err) + 180.0) % 360.0 - 180.0


def face_yaw(env, cfg: dict, target: float, limit: int = 48):
    obs = env.observation()
    vectors = (cfg.get("actions") or {}).get("vectors") or {}
    for _ in range(limit):
        yaw = yaw_of(obs, cfg)
        err = _wrap(target - yaw)
        if abs(err) <= 8.0:
            return obs, yaw
        name = "turn_right_15" if err > 0 else "turn_left_15"
        obs, _r, _d, _i = env.step(vectors[name])
    return obs, yaw_of(obs, cfg)


def walk(env, cfg: dict, n: int):
    vectors = (cfg.get("actions") or {}).get("vectors") or {}
    obs = env.observation()
    start = position_of(obs, cfg)
    for _ in range(n):
        obs, _r, _d, _i = env.step(vectors["forward"])
    stop = position_of(obs, cfg)
    return obs, start, stop


def signs_match(delta, expected) -> bool:
    for d, e in zip(delta, expected):
        if e == 0:
            if abs(d) >= 0.4:
                return False
        elif d * e <= 0:
            return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    from lfnav.env.minedojo_env import MineDojoEnv

    env = MineDojoEnv(cfg, task=None, seed=0)
    rows = []
    try:
        env.reset()
        for target in (0.0, 90.0, 180.0, 270.0):
            obs, yaw = face_yaw(env, cfg, target)
            obs, start, stop = walk(env, cfg, args.steps)
            dx = float(stop[0] - start[0])
            dz = float(stop[2] - start[2])
            expected = _front_delta(yaw)
            ok = signs_match((dx, dz), expected)
            rows.append({
                "target_yaw": target,
                "measured_yaw": yaw,
                "start": start,
                "stop": stop,
                "delta_xz": [dx, dz],
                "front_delta": list(expected),
                "match": ok,
            })
            print(
                f"yaw~{yaw:.1f} (want {target}) dpos=({dx:.2f},{dz:.2f}) "
                f"front_delta={expected} match={ok}",
                flush=True,
            )
    finally:
        close_env(env)
    payload = {
        "rows": rows,
        "all_match": all(r["match"] for r in rows) if rows else False,
        "note": "lfnav yaw 0 looks +z; +yaw is right. See _front_delta.",
    }
    atomic_write_json(Path(args.out), payload)
    print(json.dumps({"out": args.out, "all_match": payload["all_match"]}, indent=2))
    return 0 if payload["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
