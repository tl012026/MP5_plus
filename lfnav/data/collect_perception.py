"""Collect RGB frames and ground-truth scene states. Shards are atomic and resumable.

Pilot size is configs/lfnav/collect_pilot.yaml (about 1000 frames). The full run is
the same script with a larger --frames. Use one environment and reset between scenes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lfnav.config import load_config
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.eval.run import close_env
from lfnav.gt.scene_state import scene_state_from_obs
from lfnav.io import atomic_write_json
from lfnav.obsutil import lookup


def _shard_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with np.load(path) as data:
        return int(data["rgb"].shape[0])


def frames_done(out: Path) -> int:
    return sum(_shard_count(path) for path in sorted(out.glob("shard_*.npz")))


def write_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.stack([np.asarray(row["rgb"], dtype=np.uint8) for row in rows])
    payload = {k: np.asarray([row[k] for row in rows]) for k in ("yaw", "pitch", "x", "z", "seed")}
    scenes = np.asarray([json.dumps(row["scene"]) for row in rows])
    tmp = path.parent / (path.stem + ".partial.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, rgb=rgb, scene=scenes, **payload)
    tmp.replace(path)


def _rgb(obs, cfg):
    raw = lookup(obs, (cfg.get("obs_keys") or {}).get("rgb", "rgb"))
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def capture(env, cfg, n_frames: int, out: Path, seed: int, yaw_step: int = 30) -> int:
    done = frames_done(out)
    if done >= n_frames:
        print(f"already have {done} frames", flush=True)
        return done
    obs = env.reset(seed=seed)
    shard_size = int(cfg.get("shard_size", 50))
    rows = []
    shard_index = len(list(out.glob("shard_*.npz")))
    started = time.perf_counter()
    produced = done
    while produced < n_frames:
        for _ in range(max(int(360 / yaw_step), 1)):
            scene = scene_state_from_obs(obs, cfg)
            pos = lookup(obs, (cfg.get("obs_keys") or {}).get("position"), [0, 0, 0])
            rows.append({
                "rgb": _rgb(obs, cfg),
                "scene": scene.to_dict(),
                "yaw": float(lookup(obs, (cfg.get("obs_keys") or {}).get("yaw"), 0.0) or 0.0),
                "pitch": float(lookup(obs, (cfg.get("obs_keys") or {}).get("pitch"), 0.0) or 0.0),
                "x": float(pos[0]),
                "z": float(pos[-1]),
                "seed": seed,
            })
            produced += 1
            if len(rows) >= shard_size or produced >= n_frames:
                write_shard(out / f"shard_{shard_index:05d}.npz", rows)
                rows = []
                shard_index += 1
            if produced >= n_frames:
                break
            obs, _r, _d, _i = env.step("turn_right_15")
            if yaw_step >= 30:
                obs, _r, _d, _i = env.step("turn_right_15")
        obs, _r, _d, _i = env.step("forward")
    elapsed = max(time.perf_counter() - started, 1e-6)
    atomic_write_json(out / "collect_meta.json", {"frames": produced, "seconds": elapsed, "frames_per_sec": (produced - done) / elapsed, "seed": seed})
    print(f"frames={produced} speed={(produced - done) / elapsed:.2f} fps", flush=True)
    return produced


def make_fake(cfg, seed: int):
    cfg = dict(cfg)
    cfg["image_hw"] = list(cfg.get("image_hw", [28, 42]))
    return FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": 4.0}], "blocks": {(2, 3): "grass"}, "biome": "plains", "time": "day", "weather": "clear"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("fake", "minedojo"), default="fake")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    cfg["image_hw"] = list(cfg.get("collect_image_hw", cfg.get("image_hw")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        if args.backend == "fake":
            env = make_fake(cfg, args.seed)
        else:
            from lfnav.env.minedojo_env import MineDojoEnv
            env = MineDojoEnv(cfg, task=None, seed=args.seed)
        capture(env, cfg, args.frames, out, args.seed)
    finally:
        close_env(env)


if __name__ == "__main__":
    main()
