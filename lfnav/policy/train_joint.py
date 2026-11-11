"""Train A and C2 together on precomputed DINOv2 features. --dry-run uses fake tensors."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lfnav.config import load_config
from lfnav.data.demo_format import read_episode
from lfnav.io import atomic_write_json
from lfnav.perception.extract_features import episode_feature_name
from lfnav.policy.c2_data import (
    class_weights,
    demo_roots,
    encode_aux,
    iter_success_episodes,
    pad_chunk,
    select_training_episodes,
    split_by_seed,
    _visible,
)
from lfnav.policy.joint import IGNORE, JointNet, joint_loss, joint_spec, joint_val_acc, scene_targets


def git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return None


@dataclass
class JointSample:
    tokens: np.ndarray
    aux: np.ndarray
    y: np.ndarray
    present: np.ndarray
    direction: np.ndarray
    distance: np.ndarray
    biome: np.int64
    time: np.int64
    weather: np.int64
    brightness: np.int64
    seed: int
    episode_id: str


@dataclass
class JointSplit:
    train: list[JointSample]
    val: list[JointSample]
    spec: FeatureSpec
    n_episodes: int
    val_seeds: list[int]
    filter_stats: dict
    dim: int
    grid: tuple[int, int]
    n_decisions: int


def load_feature_file(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.item()
    return raw


def samples_from_episode(folder: Path, feat_dir: Path, spec: FeatureSpec) -> list[JointSample]:
    packed = read_episode(folder)
    record = packed["record"]
    feat_path = Path(feat_dir) / episode_feature_name(record, Path(folder))
    if not feat_path.is_file():
        return []
    feat = load_feature_file(feat_path)
    tokens = np.asarray(feat["tokens"])
    decisions = {int(d): i for i, d in enumerate(np.asarray(feat["decisions"]).tolist())}
    from lfnav.policy.c2_data import load_task

    task = load_task(record, Path(folder))
    seed = int(record.get("seed") if record.get("seed") is not None else Path(folder).name)
    task_id = str(record.get("task_name") or task.get("id") or Path(folder).parts[-3])
    difficulty = str(record.get("difficulty") or Path(folder).parts[-2])
    episode_id = f"{task_id}/{difficulty}/{seed}"
    last = None
    rows = []
    for i, step in enumerate(packed["steps"]):
        chunk = step.get("action_chunk")
        if not chunk:
            continue
        decision = int(step["decision"] if step.get("decision") is not None else i)
        if decision not in decisions:
            continue
        match = step.get("match") or {}
        env_step = int(step.get("env_step") or 0)
        if _visible(match):
            last = {
                "step": env_step,
                "direction": match.get("target_direction"),
                "distance": match.get("target_distance"),
            }
        mem = step.get("memory") or {}
        aux = encode_aux(task, mem.get("sector8"), mem.get("recent_actions"), last, env_step, spec)
        labels = scene_targets(step.get("scene") or {}, spec)
        idx = decisions[decision]
        rows.append(JointSample(
            tokens=np.asarray(tokens[idx], dtype=np.float32),
            aux=aux,
            y=np.asarray(pad_chunk(chunk, spec), dtype=np.int64),
            present=labels["present"],
            direction=labels["direction"],
            distance=labels["distance"],
            biome=labels["biome"],
            time=labels["time"],
            weather=labels["weather"],
            brightness=labels["brightness"],
            seed=seed,
            episode_id=episode_id,
        ))
    return rows


def load_joint_split(demos, feat_dir, cfg: dict | None = None, val_frac: float = 0.15, rng_seed: int = 0) -> JointSplit:
    cfg = cfg or {}
    spec = joint_spec(cfg)
    folders = []
    for root in demo_roots(demos, cfg):
        folders.extend(iter_success_episodes(root))
    episodes, filter_stats = select_training_episodes(folders, cfg)
    train_eps, val_eps, val_seeds = split_by_seed(episodes, val_frac=val_frac, rng_seed=rng_seed)
    train, val = [], []
    for folder in train_eps:
        train.extend(samples_from_episode(folder, Path(feat_dir), spec))
    for folder in val_eps:
        val.extend(samples_from_episode(folder, Path(feat_dir), spec))
    all_rows = train + val
    if not all_rows:
        raise RuntimeError("no joint samples (missing features or empty filter)")
    grid = tuple(int(x) for x in all_rows[0].tokens.shape[:2])
    dim = int(all_rows[0].tokens.shape[-1])
    print(f"joint samples train={len(train)} val={len(val)} episodes={len(episodes)} dim={dim} grid={grid}")
    return JointSplit(train, val, spec, len(episodes), val_seeds, filter_stats, dim, grid, len(all_rows))


def _collate(rows: list[JointSample]) -> dict:
    return {
        "tokens": np.stack([r.tokens for r in rows]),
        "aux": np.stack([r.aux for r in rows]),
        "y": np.stack([r.y for r in rows]),
        "present": np.stack([r.present for r in rows]),
        "direction": np.stack([r.direction for r in rows]),
        "distance": np.stack([r.distance for r in rows]),
        "biome": np.asarray([r.biome for r in rows], dtype=np.int64),
        "time": np.asarray([r.time for r in rows], dtype=np.int64),
        "weather": np.asarray([r.weather for r in rows], dtype=np.int64),
        "brightness": np.asarray([r.brightness for r in rows], dtype=np.int64),
    }


def _to_torch(batch, device):
    import torch

    out = {}
    for key, value in batch.items():
        tensor = torch.from_numpy(np.asarray(value))
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        out[key] = tensor.to(device)
    return out


def dry_run_torch() -> float:
    import torch

    cfg = load_config()
    spec = joint_spec(cfg)
    b, h, w, d = 8, 4, 6, 16
    tokens = torch.randn(b, h, w, d)
    aux = torch.randn(b, spec.aux_dim)
    model = JointNet(d, spec, hidden=32, horizon=spec.horizon)
    y = torch.zeros(b, spec.horizon, dtype=torch.long)
    present = torch.zeros(b, len(spec.objects), dtype=torch.long)
    present[:, 0] = 1
    direction = torch.full((b, len(spec.objects)), IGNORE, dtype=torch.long)
    direction[:, 0] = 1
    distance = torch.full((b, len(spec.objects)), IGNORE, dtype=torch.long)
    distance[:, 0] = 0
    batch = {
        "y": y,
        "present": present,
        "direction": direction,
        "distance": distance,
        "biome": torch.zeros(b, dtype=torch.long),
        "time": torch.zeros(b, dtype=torch.long),
        "weather": torch.zeros(b, dtype=torch.long),
        "brightness": torch.zeros(b, dtype=torch.long),
    }
    cw = torch.ones(len(spec.actions))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = float(joint_loss(model(tokens, aux), batch, cw, 1.0)[0].item())
    for _ in range(8):
        opt.zero_grad()
        loss, _ = joint_loss(model(tokens, aux), batch, cw, 1.0)
        loss.backward()
        opt.step()
    after = float(joint_loss(model(tokens, aux), batch, cw, 1.0)[0].item())
    print(f"dry-run joint torch {before:.4f} -> {after:.4f}")
    if after > before + 1e-3:
        raise RuntimeError(f"torch loss increased from {before} to {after}")
    return after


def dry_run() -> float:
    try:
        return dry_run_torch()
    except ImportError:
        print("dry-run joint skipped (no torch)")
        return 0.0


def train_loop(split: JointSplit, out: Path, cfg: dict, epochs: int, batch: int, lr: float, seed: int, lam: float) -> Path:
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    out.mkdir(parents=True, exist_ok=True)
    model = JointNet(split.dim, split.spec, int((cfg.get("joint") or {}).get("hidden") or 256), split.spec.horizon)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    cw = torch.from_numpy(class_weights(split.train, len(split.spec.actions))).to(device)
    best = None
    best_path = out / "joint_v1.pt"
    if not split.train:
        raise RuntimeError("no training samples")
    tr = _collate(split.train)
    va = _collate(split.val) if split.val else None

    def batches(n):
        idx = np.arange(n)
        np.random.shuffle(idx)
        for start in range(0, n, int(batch)):
            yield idx[start : start + int(batch)]

    def slice_np(data, sl):
        return {k: v[sl] for k, v in data.items()}

    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for sl in batches(len(split.train)):
            batch_t = _to_torch(slice_np(tr, sl), device)
            opt.zero_grad()
            out_m = model(batch_t["tokens"], batch_t["aux"])
            loss, _ = joint_loss(out_m, batch_t, cw, lam)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        val_loss = train_loss
        acc = {"present": 0.0, "direction": 0.0, "distance": 0.0, "action": 0.0}
        if va is not None:
            model.eval()
            with torch.no_grad():
                batch_v = _to_torch(va, device)
                out_v = model(batch_v["tokens"], batch_v["aux"])
                val_loss = float(joint_loss(out_v, batch_v, cw, lam)[0].item())
                acc = joint_val_acc(out_v, batch_v)
        print(
            f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f} "
            f"present={acc['present']:.3f} dir={acc['direction']:.3f} "
            f"dist={acc['distance']:.3f} action={acc['action']:.3f}"
        )
        if best is None or val_loss < best:
            best = val_loss
            torch.save({
                "model": model.state_dict(),
                "dim": split.dim,
                "grid": split.grid,
                "actions": split.spec.actions,
                "horizon": split.spec.horizon,
                "recent_k": split.spec.recent_k,
            }, best_path)
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--demos", nargs="*", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-c2", type=float, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    if args.dry_run or (not args.out and not args.features):
        dry_run()
        return
    cfg = load_config(args.config) if args.config else load_config()
    cfg["joint"] = dict(cfg.get("joint") or {})
    lam = float(args.lambda_c2 if args.lambda_c2 is not None else cfg["joint"].get("lambda_c2", 1.0))
    version = args.version or cfg["joint"].get("data_version") or "joint_v1"
    feat = Path(args.features) if args.features else Path(cfg["paths"]["features"]) / version
    split = load_joint_split(args.demos, feat, cfg, rng_seed=args.seed)
    out = Path(args.out)
    ckpt = train_loop(split, out, cfg, args.epochs, args.batch, args.lr, args.seed, lam)
    atomic_write_json(out / "meta.json", {
        "data_version": version,
        "features": str(feat),
        "git_commit": git_hash(),
        "filter_stats": split.filter_stats,
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_episodes": split.n_episodes,
        "n_decisions": split.n_decisions,
        "val_seeds": split.val_seeds,
        "dim": split.dim,
        "grid": list(split.grid),
        "lambda_c2": lam,
        "checkpoint": str(ckpt),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "actions": split.spec.actions,
        "horizon": split.spec.horizon,
    })
    print(f"wrote {ckpt}")


if __name__ == "__main__":
    main()
