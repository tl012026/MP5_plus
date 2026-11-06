"""Cache frozen encoder features as fp16 shards. The fake encoder needs no weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lfnav.perception.encoder import FakeEncoder, load_encoder


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder", default="fake")
    args = parser.parse_args()
    encoder = FakeEncoder() if args.encoder == "fake" else load_encoder(args.encoder)
    count = extract_tree(Path(args.src), Path(args.out), encoder)
    print(f"shards={count}")


if __name__ == "__main__":
    main()
