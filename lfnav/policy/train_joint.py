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
from lfnav.perception.convert_features import load_memmaps, rows_float32
from lfnav.perception.extract_features import episode_feature_name
from lfnav.policy.c2_data import (
    FeatureSpec,
    class_weights,
    demo_roots,
    encode_aux,
    iter_success_episodes,
    pad_chunk,
    select_training_episodes,
    split_by_seed,
    _visible,
)
from lfnav.policy.joint import (
    IGNORE,
    JointNet,
    joint_loss,
    joint_policy_flags,
    joint_runtime_from_cfg,
    joint_spec,
    joint_val_acc,
    scene_targets,
)


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {value!r}")


def apply_joint_cli(cfg: dict, args) -> dict:
    cfg["joint"] = dict(cfg.get("joint") or {})
    if getattr(args, "policy_uses_pooled", None) is not None:
        cfg["joint"]["policy_uses_pooled"] = bool(args.policy_uses_pooled)
    if getattr(args, "detach_scene", None) is not None:
        cfg["joint"]["detach_scene"] = bool(args.detach_scene)
    return cfg


def checkpoint_record(flags: dict, version: str, **extra) -> dict:
    payload = dict(extra)
    payload["version"] = str(version)
    payload["policy_uses_pooled"] = bool(flags["policy_uses_pooled"])
    payload["detach_scene"] = bool(flags["detach_scene"])
    for key in ("policy_uses_pooled", "detach_scene", "version", "recent_k", "memory_size", "token_pool", "hidden"):
        if key not in payload:
            raise RuntimeError(f"checkpoint must include {key}")
    return payload


def git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return None


@dataclass
class JointSample:
    row: int
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
    feat_dir: Path


def _index_lookup(index: dict) -> dict[tuple[str, int], int]:
    table = {}
    for row in index.get("rows") or []:
        table[(str(row["episode"]), int(row["decision"]))] = int(row["row"])
    return table


def samples_from_episode(folder: Path, lookup: dict, spec: FeatureSpec) -> list[JointSample]:
    packed = read_episode(folder)
    record = packed["record"]
    stem = episode_feature_name(record, Path(folder)).removesuffix(".npy")
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
        feat_row = lookup.get((stem, decision))
        if feat_row is None:
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
        rows.append(JointSample(
            row=int(feat_row),
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
    feat_dir = Path(feat_dir)
    if not (feat_dir / "tokens.npy").is_file() or not (feat_dir / "index.json").is_file():
        raise RuntimeError(f"missing memmap features in {feat_dir}; run lfnav/perception/convert_features.py")
    tokens, _, index = load_memmaps(feat_dir)
    lookup = _index_lookup(index)
    folders = []
    for root in demo_roots(demos, cfg):
        folders.extend(iter_success_episodes(root))
    episodes, filter_stats = select_training_episodes(folders, cfg)
    train_eps, val_eps, val_seeds = split_by_seed(episodes, val_frac=val_frac, rng_seed=rng_seed)
    train, val = [], []
    for folder in train_eps:
        train.extend(samples_from_episode(folder, lookup, spec))
    for folder in val_eps:
        val.extend(samples_from_episode(folder, lookup, spec))
    all_rows = train + val
    if not all_rows:
        raise RuntimeError("no joint samples (missing features or empty filter)")
    grid = tuple(int(x) for x in (index.get("grid") or tokens.shape[1:3]))
    dim = int(index.get("dim") or tokens.shape[-1])
    print(
        f"joint samples train={len(train)} val={len(val)} episodes={len(episodes)} "
        f"dim={dim} grid={grid} mmap={tokens.shape}",
        flush=True,
    )
    return JointSplit(train, val, spec, len(episodes), val_seeds, filter_stats, dim, grid, len(all_rows), feat_dir)


def _collate(rows: list[JointSample]) -> dict:
    return {
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
    flags = joint_policy_flags(cfg)
    model = JointNet(
        d, spec, hidden=32, horizon=spec.horizon,
        policy_uses_pooled=flags["policy_uses_pooled"],
        detach_scene=flags["detach_scene"],
    )
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
    print(f"dry-run joint torch {before:.4f} -> {after:.4f}", flush=True)
    if after > before + 1e-3:
        raise RuntimeError(f"torch loss increased from {before} to {after}")
    return after


def dry_run() -> float:
    try:
        return dry_run_torch()
    except ImportError:
        print("dry-run joint skipped (no torch)", flush=True)
        return 0.0


def train_loop(split: JointSplit, out: Path, cfg: dict, epochs: int, batch: int, lr: float, seed: int, lam: float, version: str = "joint_v2") -> Path:
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    flags = joint_policy_flags(cfg)
    runtime = joint_runtime_from_cfg(cfg)
    hidden = int((cfg.get("joint") or {}).get("hidden") or 256)
    model = JointNet(
        split.dim,
        split.spec,
        hidden,
        split.spec.horizon,
        policy_uses_pooled=flags["policy_uses_pooled"],
        detach_scene=flags["detach_scene"],
    )
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    cw = torch.from_numpy(class_weights(split.train, len(split.spec.actions))).to(device)
    best = None
    best_path = out / f"{version}.pt"
    if not split.train:
        raise RuntimeError("no training samples")
    tokens_mm, _, _ = load_memmaps(split.feat_dir)
    tr = _collate(split.train)
    va = _collate(split.val) if split.val else None
    train_rows = np.asarray([s.row for s in split.train], dtype=np.int64)
    val_rows = np.asarray([s.row for s in split.val], dtype=np.int64) if split.val else None

    def batches(n, shuffle=True):
        idx = np.arange(n)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, n, int(batch)):
            yield idx[start : start + int(batch)]

    def slice_np(data, sl):
        return {k: v[sl] for k, v in data.items()}

    def torch_batch(labels, feat_rows, sl):
        packed = slice_np(labels, sl)
        packed["tokens"] = rows_float32(tokens_mm, feat_rows[sl])
        return _to_torch(packed, device)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for sl in batches(len(split.train), shuffle=True):
            batch_t = torch_batch(tr, train_rows, sl)
            opt.zero_grad()
            out_m = model(batch_t["tokens"], batch_t["aux"])
            loss, _ = joint_loss(out_m, batch_t, cw, lam)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        val_loss = train_loss
        acc = {"present": 0.0, "direction": 0.0, "distance": 0.0, "action": 0.0}
        if va is not None and val_rows is not None:
            model.eval()
            loss_sum = 0.0
            n_seen = 0
            acc_sum = {k: 0.0 for k in acc}
            with torch.no_grad():
                for sl in batches(len(split.val), shuffle=False):
                    batch_v = torch_batch(va, val_rows, sl)
                    out_v = model(batch_v["tokens"], batch_v["aux"])
                    part = float(joint_loss(out_v, batch_v, cw, lam)[0].item())
                    n_b = int(len(sl))
                    loss_sum += part * n_b
                    n_seen += n_b
                    part_acc = joint_val_acc(out_v, batch_v)
                    for key in acc_sum:
                        acc_sum[key] += part_acc[key] * n_b
            val_loss = loss_sum / max(n_seen, 1)
            acc = {k: acc_sum[k] / max(n_seen, 1) for k in acc}
        print(
            f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f} "
            f"present={acc['present']:.3f} dir={acc['direction']:.3f} "
            f"dist={acc['distance']:.3f} action={acc['action']:.3f}",
            flush=True,
        )
        if best is None or val_loss < best:
            best = val_loss
            payload = checkpoint_record(
                flags,
                version,
                model=model.state_dict(),
                dim=split.dim,
                grid=split.grid,
                actions=split.spec.actions,
                horizon=split.spec.horizon,
                chunk=split.spec.horizon,
                hidden=int(hidden),
                recent_k=int(runtime["recent_k"]),
                memory_size=int(runtime["memory_size"]),
                token_pool=int(runtime["token_pool"]),
            )
            torch.save(payload, best_path)
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
    parser.add_argument("--policy-uses-pooled", type=parse_bool, default=None)
    parser.add_argument("--detach-scene", type=parse_bool, default=None)
    args = parser.parse_args()
    if args.dry_run or (not args.out and not args.features):
        dry_run()
        return
    cfg = load_config(args.config) if args.config else load_config()
    apply_joint_cli(cfg, args)
    lam = float(args.lambda_c2 if args.lambda_c2 is not None else cfg["joint"].get("lambda_c2", 1.0))
    version = args.version or cfg["joint"].get("data_version") or "joint_v2"
    feat_ver = cfg["joint"].get("features_version") or "joint_v1"
    feat = Path(args.features) if args.features else Path(cfg["paths"]["features"]) / feat_ver
    split = load_joint_split(args.demos, feat, cfg, rng_seed=args.seed)
    out = Path(args.out)
    flags = joint_policy_flags(cfg)
    ckpt = train_loop(split, out, cfg, args.epochs, args.batch, args.lr, args.seed, lam, version=version)
    runtime = joint_runtime_from_cfg(cfg)
    atomic_write_json(out / "meta.json", checkpoint_record(flags, version, **{
        "data_version": version,
        "features_version": feat_ver,
        "features": str(feat),
        "recent_k": int(runtime["recent_k"]),
        "memory_size": int(runtime["memory_size"]),
        "token_pool": int(runtime["token_pool"]),
        "git_commit": git_hash(),
        "filter_stats": split.filter_stats,
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_episodes": split.n_episodes,
        "n_decisions": split.n_decisions,
        "val_seeds": split.val_seeds,
        "dim": split.dim,
        "grid": list(split.grid),
        "hidden": int((cfg.get("joint") or {}).get("hidden") or 256),
        "chunk": split.spec.horizon,
        "lambda_c2": lam,
        "checkpoint": str(ckpt),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "actions": split.spec.actions,
        "horizon": split.spec.horizon,
    }))
    print(f"wrote {ckpt}", flush=True)


if __name__ == "__main__":
    main()
