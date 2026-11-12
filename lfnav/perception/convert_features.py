"""Pack per-episode feature .npy files into memmap arrays. No torch; one episode at a time."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lfnav.perception.encoder import pool_patch_tokens

PACKED = {"tokens.npy", "pooled.npy"}
DEFAULT_POOL = 2
GRID_16x26 = (16, 26)
GRID_8x13 = (8, 13)


def load_payload(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.item()
    if not isinstance(raw, dict):
        raise TypeError(f"{path} is not an episode feature dict")
    return raw


def default_feat_dir(version: str = "joint_v1") -> Path:
    scr = os.environ.get("SCR") or f"/scratch/{os.environ.get('USER', 'user')}"
    return Path(scr) / "lfnav" / "features" / str(version)


def episode_folder(stem: str) -> str:
    parts = str(stem).split("__")
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}/{parts[2]}"
    return str(stem)


def episode_files(feat_dir: Path) -> list[Path]:
    rows = []
    for path in sorted(Path(feat_dir).glob("*.npy")):
        if path.name in PACKED:
            continue
        rows.append(path)
    return rows


def maybe_pool_grid(grid, pool: int = DEFAULT_POOL):
    arr = np.asarray(grid)
    if arr.ndim != 3:
        raise ValueError(f"expected (H, W, D), got {arr.shape}")
    if arr.shape[:2] == GRID_16x26:
        k = int(pool) if int(pool) > 1 else DEFAULT_POOL
        return pool_patch_tokens(arr.astype(np.float32), k)
    return arr


def prepare_episode(payload: dict, pool: int = DEFAULT_POOL) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    tokens = np.asarray(payload["tokens"])
    if tokens.ndim != 4:
        raise ValueError(f"expected (T, H, W, D) tokens, got {tokens.shape}")
    grids = []
    for i in range(tokens.shape[0]):
        grids.append(maybe_pool_grid(tokens[i], pool).astype(np.float16, copy=False))
    packed = np.stack(grids, axis=0)
    pooled = packed.astype(np.float32).mean(axis=(1, 2)).astype(np.float16)
    decisions = np.asarray(payload.get("decisions"), dtype=np.int32)
    if decisions.shape[0] != packed.shape[0]:
        decisions = np.arange(packed.shape[0], dtype=np.int32)
    episode = str(payload.get("episode") or "")
    return packed, pooled, decisions, episode


def count_and_shape(files: list[Path], pool: int) -> tuple[int, tuple[int, int, int]]:
    n = 0
    shape = None
    for path in files:
        payload = load_payload(path)
        tokens = np.asarray(payload["tokens"])
        if tokens.ndim != 4 or tokens.shape[0] == 0:
            continue
        grid = tuple(int(x) for x in maybe_pool_grid(tokens[0], pool).shape)
        if shape is None:
            shape = grid
        elif grid != shape:
            raise RuntimeError(f"{path} grid {grid} != {shape}")
        n += int(tokens.shape[0])
        del payload, tokens
    if not n or shape is None:
        raise RuntimeError(f"no feature rows in {files[0].parent if files else '?'}")
    return n, shape


def open_write_memmap(path: Path, shape: tuple, dtype) -> np.ndarray:
    path = Path(path)
    if path.exists():
        path.unlink()
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)


def convert_dir(feat_dir, pool: int = DEFAULT_POOL) -> dict:
    feat_dir = Path(feat_dir)
    files = episode_files(feat_dir)
    if not files:
        raise RuntimeError(f"no per-episode .npy files in {feat_dir}")
    n, (height, width, dim) = count_and_shape(files, pool)
    tokens_path = feat_dir / "tokens.npy"
    pooled_path = feat_dir / "pooled.npy"
    tokens_mm = open_write_memmap(tokens_path, (n, height, width, dim), np.float16)
    pooled_mm = open_write_memmap(pooled_path, (n, dim), np.float16)
    rows = []
    offset = 0
    for path in files:
        packed, pooled, decisions, episode = prepare_episode(load_payload(path), pool)
        t = int(packed.shape[0])
        if not episode:
            episode = path.stem
        tokens_mm[offset : offset + t] = packed
        pooled_mm[offset : offset + t] = pooled
        tokens_mm.flush()
        pooled_mm.flush()
        for k in range(t):
            row = offset + k
            rows.append({
                "row": row,
                "episode": episode,
                "folder": episode_folder(episode),
                "file": path.name,
                "decision": int(decisions[k]),
                "aux_row": row,
                "label_row": row,
            })
        offset += t
        print(f"packed {path.name} rows={t} total={offset}/{n}", flush=True)
        del packed, pooled
    del tokens_mm, pooled_mm
    index = {
        "n": n,
        "grid": [height, width],
        "dim": dim,
        "token_pool": int(pool),
        "tokens": "tokens.npy",
        "pooled": "pooled.npy",
        "aux": "steps.jsonl",
        "labels": "steps.jsonl",
        "rows": rows,
    }
    (feat_dir / "index.json").write_text(json.dumps(index) + "\n")
    print(f"wrote {tokens_path} shape={(n, height, width, dim)}", flush=True)
    return index


def load_memmaps(feat_dir) -> tuple[np.ndarray, np.ndarray, dict]:
    feat_dir = Path(feat_dir)
    tokens = np.load(feat_dir / "tokens.npy", mmap_mode="r")
    pooled = np.load(feat_dir / "pooled.npy", mmap_mode="r")
    index = json.loads((feat_dir / "index.json").read_text())
    return tokens, pooled, index


def rows_float32(store: np.ndarray, indices) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    return np.asarray(store[idx], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="joint_v1")
    parser.add_argument("--features", default=None)
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL)
    args = parser.parse_args()
    feat = Path(args.features) if args.features else default_feat_dir(args.version)
    convert_dir(feat, pool=args.pool)


if __name__ == "__main__":
    main()
