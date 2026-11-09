"""Fit a pinhole K from pure-rotation RGB frames. OpenCV is optional.

Reads frames.json written by capture_rotations.py. Does not edit default.yaml;
prints a suggestion for fov_deg or a full intrinsics block.

    python lfnav/gt/fit_intrinsics.py --in $SCR/lfnav/calib/rot --out $SCR/lfnav/calib/intrinsics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lfnav.config import load_config
from lfnav.gt.camera import (
    fov_from_K,
    initial_K,
    match_pair,
    optimize_K,
    ransac_inliers,
    relative_rotation,
    residuals,
    suggest_config,
)
from lfnav.io import atomic_write_json


def load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))
    except ImportError:
        pass
    raw = path.read_bytes()
    if raw[:2] != b"P6":
        raise RuntimeError("need Pillow to read this PNG, or save PPM")
    header, body = raw.split(b"\n", 3)[0:4] if False else (None, None)
    parts = raw.split(b"\n")
    # P6\nW H\n255\n
    idx = 0
    lines = []
    rest = raw
    while not rest.startswith(b"P6"):
        raise RuntimeError("not a P6 ppm")
    rest = rest[3:]
    if rest[:1] == b"\n":
        rest = rest[1:]
    # skip comments
    while rest[:1] == b"#":
        rest = rest.split(b"\n", 1)[1]
    dims, rest = rest.split(b"\n", 1)
    w, h = [int(v) for v in dims.split()]
    maxv, rest = rest.split(b"\n", 1)
    return np.frombuffer(rest, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)


def pair_ok(a: dict, b: dict) -> bool:
    dyaw = abs(((a["yaw"] - b["yaw"] + 180) % 360) - 180)
    dpitch = abs(a["pitch"] - b["pitch"])
    return 8.0 <= max(dyaw, dpitch) <= 40.0


def fit_directory(folder: Path, cfg: dict, use_opencv: bool = True) -> dict:
    meta = json.loads((folder / "frames.json").read_text())
    frames = meta["frames"]
    height, width = (meta.get("image_size") or frames[0]["image_size"])[:2]
    # capture stored [H, W]
    if height > width * 1.2:
        # already H,W
        pass
    calib = cfg.get("camera_calib") or {}
    k0 = initial_K(height, width, float(cfg.get("fov_deg", 70)))
    pairs = []
    pair_rms = []
    for i, a in enumerate(frames):
        for b in frames[i + 1:i + 4]:
            if a.get("seed") != b.get("seed"):
                continue
            if not pair_ok(a, b):
                continue
            rgb1 = load_rgb(folder / a["file"])
            rgb2 = load_rgb(folder / b["file"])
            if rgb1.shape[0] != height:
                height, width = rgb1.shape[0], rgb1.shape[1]
                k0 = initial_K(height, width, float(cfg.get("fov_deg", 70)))
            R = relative_rotation(a["yaw"], a["pitch"], b["yaw"], b["pitch"])
            src, dst = match_pair(rgb1, rgb2, R, k0, calib, use_opencv=use_opencv)
            if len(src) < 8:
                continue
            keep = ransac_inliers(src, dst, k0, R, thresh=float(calib.get("ransac_px", 4.0)))
            if keep.sum() < 8:
                continue
            pairs.append((src[keep], dst[keep], R))
            pred_err = residuals(np.array([k0[0, 0], k0[1, 1], k0[0, 2], k0[1, 2]]), [(src[keep], dst[keep], R)], width, height)
            pair_rms.append(float(np.sqrt(np.mean(pred_err ** 2))))
    if not pairs:
        raise RuntimeError(f"no correspondence pairs in {folder}")
    K, rms = optimize_K(pairs, k0, width, height)
    report = fov_from_K(K, height, width)
    report["rms"] = rms
    arr = np.array(pair_rms, np.float64) if pair_rms else np.array([rms])
    report["pair_rms_median"] = float(np.median(arr))
    report["pair_rms_iqr"] = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    report["n_pairs"] = len(pairs)
    report["delta_fov_v_vs_70"] = report["fov_v_deg"] - 70.0
    report["suggest"] = suggest_config(report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-opencv", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    report = fit_directory(Path(args.src), cfg, use_opencv=not args.no_opencv)
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
