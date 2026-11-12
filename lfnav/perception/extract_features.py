"""Cache frozen encoder features as fp16 shards or per-episode demo files.

The fake encoder needs no weights. Demo extraction only keeps episodes that pass
the C2 training filter, and only decisions that have both an action chunk and a frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lfnav.config import load_config
from lfnav.data.demo_format import read_episode
from lfnav.perception.encoder import FakeEncoder, load_encoder, token_pool_size
from lfnav.perception.joint_features import encode_joint_tokens
from lfnav.policy.c2_data import demo_roots, iter_success_episodes, select_training_episodes


def extract_shard(src: Path, dst: Path, encoder) -> None:
    with np.load(src) as data:
        rgb = data["rgb"]
    cls_rows = []
    token_rows = []
    for frame in rgb:
        feats = encoder(frame)
        cls_rows.append(np.asarray(feats["cls"], dtype=np.float16))
        token_rows.append(np.asarray(feats["tokens"], dtype=np.float16))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.stem + ".partial.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, cls=np.stack(cls_rows), tokens=np.stack(token_rows))
    tmp.replace(dst)


def extract_tree(src_dir: Path, dst_dir: Path, encoder) -> int:
    written = 0
    for src in sorted(src_dir.glob("shard_*.npz")):
        dst = dst_dir / src.name
        if dst.is_file():
            written += 1
            continue
        extract_shard(src, dst, encoder)
        written += 1
    return written


def episode_feature_name(record: dict, folder: Path) -> str:
    task_id = str(record.get("task_name") or record.get("task") or folder.parts[-3])
    difficulty = str(record.get("difficulty") or folder.parts[-2])
    seed = int(record.get("seed") if record.get("seed") is not None else folder.name)
    return f"{task_id}__{difficulty}__{seed}.npy"


def feature_path_for(out: Path, record: dict, folder: Path) -> Path:
    return Path(out) / episode_feature_name(record, folder)


def _load_frame(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def decision_frames(folder: Path, steps: list[dict]) -> list[tuple[int, Path]]:
    rows = []
    for i, step in enumerate(steps):
        if not step.get("action_chunk"):
            continue
        decision = int(step["decision"] if step.get("decision") is not None else i)
        frame = Path(folder) / "frames" / f"{decision:05d}.jpg"
        if frame.is_file():
            rows.append((decision, frame))
    return rows


def extract_episode(folder: Path, dst: Path, encoder, pool: int = 1) -> int:
    packed = read_episode(folder)
    pairs = decision_frames(folder, packed["steps"])
    if not pairs:
        return 0
    if dst.is_file():
        return len(pairs)
    pooled_rows = []
    token_rows = []
    decisions = []
    for decision, frame in pairs:
        # On-disk JPEG already went through preprocess_frame at collect; do not re-JPEG.
        feats = encode_joint_tokens(_load_frame(frame), encoder, pool)
        pooled_rows.append(np.asarray(feats["pooled"], dtype=np.float16))
        token_rows.append(np.asarray(feats["tokens"], dtype=np.float16))
        decisions.append(decision)
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pooled": np.stack(pooled_rows),
        "tokens": np.stack(token_rows),
        "decisions": np.asarray(decisions, dtype=np.int32),
        "episode": episode_feature_name(packed["record"], Path(folder)).removesuffix(".npy"),
    }
    tmp = dst.with_suffix(".partial.npy")
    if tmp.exists():
        tmp.unlink()
    np.save(tmp, payload, allow_pickle=True)
    tmp.replace(dst)
    return len(pairs)


def extract_demos(demos, out: Path, encoder, cfg: dict | None = None) -> dict:
    folders = []
    for root in demo_roots(demos, cfg):
        folders.extend(iter_success_episodes(root))
    kept, stats = select_training_episodes(folders, cfg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_decisions = 0
    skipped_no_frame = 0
    pool = token_pool_size(cfg)
    for folder in kept:
        packed = read_episode(folder)
        dst = feature_path_for(out, packed["record"], folder)
        n = extract_episode(folder, dst, encoder, pool=pool)
        if n == 0:
            skipped_no_frame += 1
            continue
        n_files += 1
        n_decisions += n
    manifest = {
        "n_episodes": n_files,
        "n_decisions": n_decisions,
        "skipped_no_frame": skipped_no_frame,
        "filter_stats": stats,
        "encoder": getattr(encoder, "name", None),
        "token_pool": pool,
        "out": str(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"features episodes={n_files} decisions={n_decisions} skipped_no_frame={skipped_no_frame}")
    return manifest


def _maybe_cuda(encoder):
    try:
        import torch
    except ImportError:
        return encoder
    if not torch.cuda.is_available():
        return encoder
    if hasattr(encoder, "to"):
        encoder.to("cuda")
        print("encoder device=cuda")
    return encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=None, help="legacy perception shard directory")
    parser.add_argument("--demos", nargs="*", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder", default="dinov2_vits14")
    parser.add_argument("--config", default=None)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    encoder = FakeEncoder() if args.encoder in ("fake", "dry-run") else load_encoder(args.encoder)
    encoder = _maybe_cuda(encoder)
    out = Path(args.out)
    if args.src and args.demos is None:
        count = extract_tree(Path(args.src), out, encoder)
        print(f"shards={count}")
        return
    cfg = load_config(args.config) if args.config else load_config()
    version = args.version or ((cfg.get("joint") or {}).get("data_version") or "joint_v1")
    if out.name != version:
        out = out / version
    extract_demos(args.demos, out, encoder, cfg)


if __name__ == "__main__":
    main()
