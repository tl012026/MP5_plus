"""Check aligned lidar patches against pink pig pixels. Run in the MineDojo image.

    python lfnav/gt/verify_alignment.py --intrinsics $SCR/lfnav/calib/intrinsics.json --out $SCR/lfnav/calib/align
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lfnav.config import load_config
from lfnav.geom import patch_grid
from lfnav.gt.capture_rotations import NOOP, as_hwc, pose_of
from lfnav.io import atomic_write_json
from lfnav.obsutil import lookup


def pink_mask(rgb: np.ndarray, low, high) -> np.ndarray:
    low = np.asarray(low, np.float32)
    high = np.asarray(high, np.float32)
    pix = rgb.astype(np.float32)
    return np.all((pix >= low) & (pix <= high), axis=-1)


def patch_sets(mask: np.ndarray, n_h: int, n_w: int, frac: float) -> set[tuple[int, int]]:
    h, w = mask.shape
    ph, pw = h / n_h, w / n_w
    hit = set()
    for i in range(n_h):
        for j in range(n_w):
            y0, y1 = int(i * ph), int((i + 1) * ph)
            x0, x1 = int(j * pw), int((j + 1) * pw)
            cell = mask[y0:y1, x0:x1]
            if cell.size and float(cell.mean()) >= frac:
                hit.add((i, j))
    return hit


def ray_pig_patches(obs: dict, cfg: dict) -> set[tuple[int, int]]:
    keys = cfg.get("obs_keys") or {}
    rays = lookup(obs, keys.get("aligned_rays"), []) or []
    hit = set()
    for ray in rays:
        if str(ray.get("label") or "").lower() != "pig":
            continue
        hit.add((int(ray["row"]), int(ray["col"])))
    return hit


def iou(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mean_center_distance(ray_cells: set, pink: np.ndarray, n_h: int, n_w: int) -> float | None:
    h, w = pink.shape
    ys, xs = np.where(pink)
    if len(xs) == 0 or not ray_cells:
        return None
    pink_pts = np.stack([xs, ys], 1).astype(np.float64)
    ph, pw = h / n_h, w / n_w
    dists = []
    for i, j in ray_cells:
        cx = (j + 0.5) * pw
        cy = (i + 0.5) * ph
        d = np.hypot(pink_pts[:, 0] - cx, pink_pts[:, 1] - cy).min()
        dists.append(float(d))
    return float(np.mean(dists))


def overlay(rgb, ray_cells, pink_cells, n_h, n_w) -> np.ndarray:
    out = rgb.copy()
    h, w, _ = out.shape
    ph, pw = h / n_h, w / n_w

    def box(i, j, color):
        y0, y1 = int(i * ph), max(int((i + 1) * ph) - 1, int(i * ph) + 1)
        x0, x1 = int(j * pw), max(int((j + 1) * pw) - 1, int(j * pw) + 1)
        out[y0:y0 + 2, x0:x1] = color
        out[y1 - 1:y1 + 1, x0:x1] = color
        out[y0:y1, x0:x0 + 2] = color
        out[y0:y1, x1 - 1:x1 + 1] = color

    for i, j in pink_cells:
        box(i, j, (40, 180, 255))
    for i, j in ray_cells:
        box(i, j, (255, 40, 40))
    return out


def apply_intrinsics(cfg: dict, path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    cfg = dict(cfg)
    cfg["intrinsics"] = {
        "fx": data["fx"],
        "fy": data["fy"],
        "cx": data["cx"],
        "cy": data["cy"],
        "width": data["width"],
        "height": data["height"],
    }
    env_cfg = dict(cfg.get("minedojo") or {})
    env_cfg["use_lidar"] = True
    env_cfg["start_at_night"] = False
    env_cfg["specified_biome"] = "plains"
    cfg["minedojo"] = env_cfg
    return cfg


def capture_kwargs(cfg: dict, seed: int):
    from lfnav.env.minedojo_env import _kwargs
    kwargs = _kwargs(cfg, seed)
    kwargs["initial_weather"] = "clear"
    kwargs["start_at_night"] = False
    kwargs["allow_time_passage"] = False
    kwargs["specified_biome"] = "plains"
    kwargs["initial_mobs"] = ["pig", "pig", "pig"]
    kwargs["initial_mob_spawn_range_low"] = (-6, 1, 3)
    kwargs["initial_mob_spawn_range_high"] = (6, 3, 6)
    return kwargs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args(argv)
    cfg = apply_intrinsics(load_config(args.config), Path(args.intrinsics))
    calib = cfg.get("camera_calib") or {}
    low = calib.get("pig_rgb_low", [180, 80, 80])
    high = calib.get("pig_rgb_high", [255, 200, 180])
    frac = float(calib.get("pig_frac", 0.08))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import minedojo
    from lfnav.env.minedojo_env import convert_rays, lidar_plan
    from lfnav.eval.run import close_env

    kwargs = capture_kwargs(cfg, args.seed)
    env = minedojo.make(**kwargs)
    plan = lidar_plan(cfg)
    key = (cfg.get("obs_keys") or {}).get("aligned_rays") or "lfnav_rays"
    n_h, n_w = patch_grid(int(cfg["image_hw"][0]), int(cfg["image_hw"][1]), int(cfg.get("patch", 14)))
    rows = []
    try:
        raw = env.reset()
        for t in range(args.frames):
            if t:
                raw, _r, _d, _i = env.step(NOOP if t % 2 == 0 else [0, 0, 0, 12, 13, 0, 0, 0])
            if isinstance(raw.get("rays"), dict):
                converted = convert_rays(raw["rays"], plan, cfg)
                raw = dict(raw)
                raw[key] = converted["aligned"]
            rgb = as_hwc(raw["rgb"])
            pink = pink_mask(rgb, low, high)
            pink_cells = patch_sets(pink, n_h, n_w, frac)
            ray_cells = ray_pig_patches(raw, cfg)
            score = iou(ray_cells, pink_cells)
            dist = mean_center_distance(ray_cells, pink, n_h, n_w)
            vis = overlay(rgb, ray_cells, pink_cells, n_h, n_w)
            name = f"align_{t:02d}.png"
            try:
                from PIL import Image
                Image.fromarray(vis).save(out / name)
            except ImportError:
                (out / name).write_bytes(b"")
            rows.append({
                "file": name,
                "iou": score,
                "mean_px": dist,
                "n_ray": len(ray_cells),
                "n_pink": len(pink_cells),
                "pose": pose_of(raw, cfg),
            })
    finally:
        close_env(env)
    ious = [r["iou"] for r in rows]
    dists = [r["mean_px"] for r in rows if r["mean_px"] is not None]
    summary = {
        "frames": rows,
        "iou_median": float(np.median(ious)) if ious else None,
        "mean_px": float(np.mean(dists)) if dists else None,
        "iou_good": float(calib.get("iou_good", 0.5)),
        "note": "median IoU above iou_good is treated as aligned enough",
    }
    atomic_write_json(out / "verify.json", summary)
    print(json.dumps({k: summary[k] for k in ("iou_median", "mean_px", "iou_good")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
