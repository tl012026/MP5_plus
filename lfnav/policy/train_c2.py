"""Train C2+GT from expert demos. --dry-run does not read demonstrations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lfnav.config import load_config
from lfnav.io import atomic_write_json
from lfnav.policy.c2 import ChunkMLP
from lfnav.policy.c2_data import C2Split, Sample, class_weights, load_c2_split
from lfnav.policy.c2_linear import ACTIONS, BehaviorCloning as LinearBehaviorCloning


def git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return None


def dry_run_linear() -> float:
    policy = LinearBehaviorCloning({"seed": 0, "policy": {"chunk": 4, "dagger": False}})
    vector = np.zeros(policy.dim)
    labels = [ACTIONS.index("forward")] * policy.horizon
    before = policy.train_step(vector, labels)
    after = policy.loss(vector, labels)
    if after > before + 1e-6:
        raise RuntimeError(f"loss increased from {before} to {after}")
    print(f"dry-run c2 loss {before:.4f} -> {after:.4f}")
    return after


def _stack(rows: list[Sample]):
    x = np.stack([row.x for row in rows], axis=0)
    y = np.stack([row.y for row in rows], axis=0)
    w = np.asarray([row.weight for row in rows], dtype=np.float32)
    return x, y, w


def chunk_loss(model, x, y, w, class_w):
    import torch
    from torch.nn import functional as F

    logits = model(x)
    total = x.new_zeros(x.shape[0])
    for k, head in enumerate(logits):
        total = total + F.cross_entropy(head, y[:, k], weight=class_w, reduction="none")
    return (total * w).mean()


def step_accuracy(model, x, y) -> list[float]:
    import torch

    with torch.no_grad():
        logits = model(x)
        acc = []
        for k, head in enumerate(logits):
            pred = head.argmax(dim=-1)
            acc.append(float((pred == y[:, k]).float().mean().item()))
    return acc


def dry_run_torch() -> float:
    import torch

    from lfnav.policy.c2_data import FeatureSpec, encode_features, pad_chunk

    cfg = load_config()
    spec = FeatureSpec.from_cfg(cfg)
    scene = {"objects": {"pig": {"present": 1.0, "direction": {"center": 1.0}, "distance": {"near": 1.0}}}}
    task = {"target": "pig", "conditions": {}}
    match = {"status": "found", "target_direction": "center", "target_distance": "near"}
    x = encode_features(scene, task, match, [0] * 8, ["forward"], None, 0, spec)
    y = np.asarray(pad_chunk(["forward", "forward", "forward", "forward"], spec), dtype=np.int64)
    xs = np.stack([x] * 32)
    ys = np.stack([y] * 32)
    ws = np.ones(32, dtype=np.float32)
    model = ChunkMLP(spec.dim, len(spec.actions), spec.horizon, 32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    xt = torch.from_numpy(xs)
    yt = torch.from_numpy(ys)
    wt = torch.from_numpy(ws)
    cw = torch.ones(len(spec.actions))
    before = float(chunk_loss(model, xt, yt, wt, cw).item())
    for _ in range(8):
        opt.zero_grad()
        loss = chunk_loss(model, xt, yt, wt, cw)
        loss.backward()
        opt.step()
    after = float(chunk_loss(model, xt, yt, wt, cw).item())
    print(f"dry-run c2 torch {before:.4f} -> {after:.4f}")
    if after > before + 1e-3:
        raise RuntimeError(f"torch loss increased from {before} to {after}")
    return after


def dry_run() -> float:
    after = dry_run_linear()
    try:
        dry_run_torch()
    except ImportError:
        pass
    return after


def train_loop(split: C2Split, out: Path, cfg: dict, epochs: int, batch: int, lr: float, seed: int) -> Path:
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    out.mkdir(parents=True, exist_ok=True)
    model = ChunkMLP(split.spec.dim, len(split.spec.actions), split.spec.horizon, int((cfg.get("c2") or {}).get("hidden") or 256))
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    cw = torch.from_numpy(class_weights(split.train, len(split.spec.actions)))
    best = None
    best_path = out / "c2_gt_v1.pt"
    if not split.train:
        raise RuntimeError("no training samples")
    x_tr, y_tr, w_tr = _stack(split.train)
    x_va = y_va = w_va = None
    if split.val:
        x_va, y_va, w_va = _stack(split.val)

    def batches(n):
        idx = np.arange(n)
        np.random.shuffle(idx)
        for start in range(0, n, int(batch)):
            yield idx[start:start + int(batch)]

    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for sl in batches(len(split.train)):
            xt = torch.from_numpy(x_tr[sl])
            yt = torch.from_numpy(y_tr[sl])
            wt = torch.from_numpy(w_tr[sl])
            opt.zero_grad()
            loss = chunk_loss(model, xt, yt, wt, cw)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        val_loss = train_loss
        acc = [0.0] * split.spec.horizon
        if x_va is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(chunk_loss(
                    model, torch.from_numpy(x_va), torch.from_numpy(y_va), torch.from_numpy(w_va), cw,
                ).item())
                acc = step_accuracy(model, torch.from_numpy(x_va), torch.from_numpy(y_va))
        acc_txt = " ".join(f"k{k}={a:.3f}" for k, a in enumerate(acc))
        print(f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f} acc {acc_txt}")
        if best is None or val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "spec": {
                "dim": split.spec.dim,
                "actions": split.spec.actions,
                "horizon": split.spec.horizon,
            }}, best_path)
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dagger", action="store_true")
    parser.add_argument("--demos", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wide-weight", type=float, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.dry_run or (not args.demos and not args.out):
        if args.dagger:
            print("DAgger merge is enabled. Collect a second round with the learner before this step.")
        dry_run()
        return
    cfg = load_config(args.config) if args.config else load_config()
    cfg["c2"] = dict(cfg.get("c2") or {})
    if args.wide_weight is not None:
        cfg["c2"]["wide_weight"] = float(args.wide_weight)
    split = load_c2_split(args.demos, cfg, rng_seed=args.seed)
    out = Path(args.out)
    ckpt = train_loop(split, out, cfg, args.epochs, args.batch, args.lr, args.seed)
    atomic_write_json(out / "meta.json", {
        "input_dim": split.spec.dim,
        "actions": split.spec.actions,
        "horizon": split.spec.horizon,
        "config": {k: cfg.get(k) for k in ("c2", "policy", "seed") if k in cfg},
        "demos": str(Path(args.demos).resolve()),
        "git_commit": git_hash(),
        "n_train": len(split.train),
        "n_val": len(split.val),
        "used_wide_frac": split.used_wide_frac,
        "val_seeds": split.val_seeds,
        "checkpoint": str(ckpt),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "wide_weight": cfg["c2"].get("wide_weight", 0.5),
    })
    print(f"wrote {ckpt}")


if __name__ == "__main__":
    main()
