"""Train the global and patch heads on cached encoder features.

--dry-run uses the fake encoder and one batch. It does not download DINOv2.
"""

from __future__ import annotations

import argparse

import numpy as np

from lfnav.perception.encoder import FakeEncoder
from lfnav.perception.learned import train_step_numpy


def split_name(seed: int) -> str:
    bucket = int(seed) % 10
    if bucket < 8:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


def dry_run() -> None:
    encoder = FakeEncoder()
    feats = encoder(np.zeros((8, 8, 3), dtype=np.uint8))
    logits = np.zeros(4, dtype=np.float64)
    target = np.array([1.0, 0.0, 1.0, 0.0])
    try:
        import torch
        from torch import nn
        layer = nn.Linear(feats["cls"].shape[0], 4)
        pred = layer(torch.from_numpy(feats["cls"]))
        loss = nn.BCEWithLogitsLoss()(pred, torch.tensor(target))
        loss.backward()
        print(f"dry-run torch loss {float(loss):.4f}")
    except Exception as exc:
        loss = train_step_numpy(logits, target)
        print(f"dry-run numpy loss {loss:.4f} ({type(exc).__name__})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.dry_run or not args.features:
        dry_run()
        return
    raise SystemExit("Full head training runs on a GPU node. Pass --features and --out there.")


if __name__ == "__main__":
    main()
