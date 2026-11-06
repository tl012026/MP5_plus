"""Draw patch labels on a frame so a person can check the ray alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lfnav.aggregate import column_direction


def tint(rgb: np.ndarray, present, names: list[str]) -> np.ndarray:
    image = np.array(rgb, copy=True)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    n_h = len(present[names[0]])
    n_w = len(present[names[0]][0])
    height, width = image.shape[:2]
    ph, pw = height // n_h, width // n_w
    for i in range(n_h):
        for j in range(n_w):
            label = None
            for name in names:
                if present[name][i][j] >= 0.5:
                    label = name
                    break
            if label is None:
                continue
            color = {"pig": (220, 80, 80), "grass": (40, 160, 40), "tree": (20, 100, 20), "cow": (80, 80, 200)}.get(label, (200, 200, 40))
            y0, x0 = i * ph, j * pw
            image[y0:y0 + max(ph // 5, 1), x0:x0 + pw] = color
            image[y0:y0 + ph, x0:x0 + max(pw // 8, 1)] = color
    return image


def write_ppm(path: Path, rgb: np.ndarray) -> None:
    height, width = rgb.shape[:2]
    header = f"P6\n{width} {height}\n255\n".encode()
    path.write_bytes(header + np.asarray(rgb, dtype=np.uint8).tobytes())


def direction_strip(n_w: int) -> list[str]:
    return [column_direction(j, n_w) for j in range(n_w)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--names", nargs="+", default=["pig", "grass", "tree", "cow", "water"])
    args = parser.parse_args()
    with np.load(args.shard, allow_pickle=False) as data:
        rgb = data["rgb"][args.index]
        scene = json.loads(str(data["scene"][args.index]))
    # The stored scene is aggregated. Overlay is informative only when patch maps
    # were saved. This draws the direction of the target if present.
    image = np.asarray(rgb)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ppm(out, image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=-1))
    print(json.dumps({"out": str(out), "target_present": {n: scene.get("objects", {}).get(n, {}).get("present", 0) for n in args.names}}))


if __name__ == "__main__":
    main()
