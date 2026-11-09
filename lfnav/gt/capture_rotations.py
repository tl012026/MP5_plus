"""Capture RGB while turning in place. Run inside the MineDojo image.

Does not walk, sprint, or jump. After each camera action it waits two no-ops,
then writes a PNG and one frames.json row. Angle steps are the measured
location_stats yaw/pitch, not the commanded 15 degrees.

    python lfnav/gt/capture_rotations.py --out /scratch/$USER/lfnav/calib/rot
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from lfnav.config import load_config
from lfnav.obsutil import lookup, scalar as numpy_scalar


NOOP = [0, 0, 0, 12, 12, 0, 0, 0]


def as_hwc(rgb) -> np.ndarray:
    """MineDojo rgb is CHW in the probe dump. Accept HWC too."""
    arr = np.asarray(rgb)
    if arr.ndim != 3:
        raise ValueError(f"rgb must be 3-D, got {arr.shape}")
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return arr[..., :3].astype(np.uint8)


def wrap_delta_deg(now: float, prev: float) -> float:
    return (float(now) - float(prev) + 180.0) % 360.0 - 180.0


def delta_minus_15k(delta_deg: float) -> float:
    """Measured step minus the nearest multiple of 15 degrees."""
    step = 15.0 * round(float(delta_deg) / 15.0)
    return float(delta_deg) - step


def scalar(value, default=None):
    if value is None:
        return default
    out = numpy_scalar(value)
    if out is value and isinstance(value, (list, tuple)) and value:
        return scalar(value[0], default)
    return default if out is None else out


def pose_of(obs: dict, cfg: dict) -> dict:
    keys = cfg.get("obs_keys") or {}
    loc = obs.get("location_stats") if isinstance(obs, dict) else {}
    loc = loc if isinstance(loc, dict) else {}
    pos = lookup(obs, keys.get("position"), loc.get("pos"))
    if pos is not None:
        pos = [float(v) for v in np.asarray(pos).reshape(-1)[:3]]
    yaw = scalar(lookup(obs, keys.get("yaw"), loc.get("yaw")), 0.0)
    pitch = scalar(lookup(obs, keys.get("pitch"), loc.get("pitch")), 0.0)
    sky = scalar(lookup(obs, keys.get("can_see_sky"), loc.get("can_see_sky")), True)
    sea = scalar(loc.get("sea_level"), 63.0)
    return {
        "pos": pos,
        "yaw": float(yaw or 0.0),
        "pitch": float(pitch or 0.0),
        "can_see_sky": bool(sky) if sky is not None else True,
        "sea_level": float(sea if sea is not None else 63.0),
    }


def skip_seed(pose: dict) -> str | None:
    pos = pose.get("pos") or [0.0, 70.0, 0.0]
    if len(pos) < 2:
        return "no position"
    if float(pos[1]) < float(pose.get("sea_level", 63.0)):
        return "below sea_level"
    if not pose.get("can_see_sky", True):
        return "can_see_sky is false"
    return None


def yaw_sweep_names() -> list[str]:
    """+15 x3, back to start, -15 x3."""
    right = ["turn_right_15"] * 3
    left = ["turn_left_15"] * 3
    return right + left + left


def pitch_hold_names() -> list[str]:
    """0 -> down 15 -> down 30 -> 0 -> up 15. Positive pitch looks down."""
    return ["look_down_15", "look_down_15", "look_up_15", "look_up_15", "look_up_15"]


def rotation_schedule() -> list[str]:
    """Camera-only steps. Pitch 0 yaw sweep, look-down yaw sweep, then pitch while yaw is held."""
    down = ["look_down_15"]
    reset_pitch = ["look_up_15"]
    return yaw_sweep_names() + down + yaw_sweep_names() + reset_pitch + pitch_hold_names()


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
        return
    except ImportError:
        pass
    h, w, _c = rgb.shape
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    path.write_bytes(header + rgb.tobytes())


def capture_kwargs(cfg: dict, seed: int, biome: str, image_size) -> dict:
    from lfnav.env.minedojo_env import _kwargs

    copy = dict(cfg)
    env_cfg = dict(copy.get("minedojo") or {})
    env_cfg["use_lidar"] = False
    env_cfg["use_voxel"] = False
    env_cfg["specified_biome"] = biome
    env_cfg["start_at_night"] = False
    env_cfg["image_size"] = [int(image_size[0]), int(image_size[1])]
    env_cfg.pop("spawn_rate", None)
    copy["minedojo"] = env_cfg
    kwargs = _kwargs(copy, seed)
    kwargs["image_size"] = (int(image_size[0]), int(image_size[1]))
    kwargs["use_lidar"] = False
    kwargs.pop("lidar_rays", None)
    kwargs["start_at_night"] = False
    kwargs["initial_weather"] = "clear"
    kwargs["allow_time_passage"] = False
    kwargs["specified_biome"] = biome
    kwargs.pop("initial_mobs", None)
    kwargs.pop("spawn_rate", None)
    kwargs.pop("initial_mob_spawn_range_low", None)
    kwargs.pop("initial_mob_spawn_range_high", None)
    return kwargs


def _vector(cfg: dict, name: str) -> list:
    if name == "noop":
        return list(NOOP)
    vectors = (cfg.get("actions") or {}).get("vectors") or {}
    if name not in vectors:
        raise KeyError(name)
    return list(vectors[name])


def _step(env, cfg, name: str):
    obs, _r, _d, _i = env.step(_vector(cfg, name))
    return obs


def _settle(env, cfg, obs, n: int = 2):
    for _ in range(n):
        obs = _step(env, cfg, "noop")
    return obs


def _level_pitch(env, cfg, obs, limit: int = 8):
    """Drive pitch toward 0 with look_up_15 / look_down_15. Positive pitch looks down."""
    pose = pose_of(obs, cfg)
    for _ in range(limit):
        pitch = pose["pitch"]
        if abs(pitch) < 7.5:
            return obs, pose
        name = "look_up_15" if pitch > 0 else "look_down_15"
        obs = _settle(env, cfg, _step(env, cfg, name))
        pose = pose_of(obs, cfg)
    return obs, pose


def run_seed(env, cfg, out: Path, seed: int, image_size, frames: list) -> int:
    obs = env.reset()
    obs = _settle(env, cfg, obs, n=4)
    obs, pose = _level_pitch(env, cfg, obs)
    reason = skip_seed(pose)
    if reason:
        print(f"skip seed {seed}: {reason}", flush=True)
        return 0
    n = 0
    prev = pose

    def snap(obs):
        nonlocal n, prev
        pose = pose_of(obs, cfg)
        rgb = as_hwc(obs["rgb"])
        name = f"s{seed:03d}_{n:03d}.png"
        write_png(out / name, rgb)
        dyaw = wrap_delta_deg(pose["yaw"], prev["yaw"]) if n else 0.0
        dpitch = wrap_delta_deg(pose["pitch"], prev["pitch"]) if n else 0.0
        frames.append({
            "file": name,
            "yaw": pose["yaw"],
            "pitch": pose["pitch"],
            "image_size": [int(image_size[0]), int(image_size[1])],
            "seed": int(seed),
            "pos": pose["pos"],
            "can_see_sky": pose["can_see_sky"],
            "yaw_delta_minus_15k": delta_minus_15k(dyaw) if n else 0.0,
            "pitch_delta_minus_15k": delta_minus_15k(dpitch) if n else 0.0,
        })
        prev = pose
        n += 1
        return pose

    snap(obs)
    for action in rotation_schedule():
        obs = _settle(env, cfg, _step(env, cfg, action))
        pose = snap(obs)
        if abs(pose["pitch"]) > 45.0 + 1e-3:
            print(f"seed {seed}: pitch {pose['pitch']} exceeded 45, stop", flush=True)
            break
    return n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Turn in place and save RGB frames.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("H", "W"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--biome", default="plains")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    image_size = args.image_size or (cfg.get("minedojo") or {}).get("image_size") or [512, 820]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames: list[dict] = []
    import minedojo
    from lfnav.eval.run import close_env

    for seed in args.seeds:
        kwargs = capture_kwargs(cfg, seed, args.biome, image_size)
        env = minedojo.make(**kwargs)
        try:
            n = run_seed(env, cfg, out, seed, image_size, frames)
            print(f"seed {seed}: {n} frames", flush=True)
        finally:
            close_env(env)
    (out / "frames.json").write_text(json.dumps({
        "image_size": [int(image_size[0]), int(image_size[1])],
        "biome": args.biome,
        "frames": frames,
    }, indent=2))
    print(json.dumps({"out": str(out), "n": len(frames)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
