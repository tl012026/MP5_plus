"""Train C2Nav. A and C share demo_dirs from the config. --dry-run needs no features."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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
from lfnav.policy.c2_data import demo_roots, iter_success_episodes, load_task, select_training_episodes
from lfnav.policy.c2nav import (
    C2NavNet,
    encode_match,
    encode_s,
    patch_bearings,
    spec_from_cfg,
    structure_record,
)
from lfnav.policy.c2nav_labels import (
    FORWARD_BINS,
    MODE_NAMES,
    N_TURN,
    branch_is_mapped,
    clip_forward_blocks,
    flip_turn_index,
    forward_index,
    labels_from_step,
    mode_from_branch,
)
from lfnav.policy.state_tracker import STEPS_NORM, TrackerNoise, flip_vector, observation_from_step, replay


def git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return None


def _argmax_index(table, names, empty=0) -> int:
    vals = [float((table or {}).get(n) or 0.0) for n in names]
    if not vals or max(vals) <= 0.0:
        return empty
    return int(np.argmax(vals))


def pool2d(arr: np.ndarray, k: int, how: str) -> np.ndarray:
    a = np.asarray(arr)
    k = max(int(k), 1)
    if k <= 1:
        return a
    h, w = a.shape
    nh, nw = h // k, w // k
    if nh == 0 or nw == 0:
        return a
    blocks = a[: nh * k, : nw * k].reshape(nh, k, nw, k)
    if how == "max":
        return blocks.max(axis=(1, 3))
    if how == "min":
        return blocks.min(axis=(1, 3))
    return blocks.mean(axis=(1, 3))


def patch_targets(scene: dict, objects: list[str], pool: int, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gh, gw = int(grid[0]), int(grid[1])
    k = len(objects)
    present = np.zeros((k, gh, gw), np.float32)
    log_d = np.zeros((k, gh, gw), np.float32)
    ignore = np.zeros((gh, gw), np.float32)
    raw_ignore = scene.get("ignore") or []
    if raw_ignore:
        ig = np.asarray(raw_ignore, dtype=np.float32)
        if ig.ndim == 2:
            ignore = pool2d(ig, pool, "max")
            if ignore.shape != (gh, gw):
                ignore = np.zeros((gh, gw), np.float32)
    patches = scene.get("patches") or {}
    for i, name in enumerate(objects):
        grid_p = np.asarray(patches.get(name) or [], dtype=np.float32)
        if grid_p.ndim != 2:
            continue
        pooled = pool2d(grid_p, pool, "max")
        if pooled.shape != (gh, gw):
            continue
        present[i] = pooled
        log_d[i] = np.where(pooled > 0.5, math.log(8.0), 0.0)
    return present, log_d, ignore


def _index_lookup(index: dict) -> dict[tuple[str, int], int]:
    table = {}
    for row in index.get("rows") or []:
        table[(str(row["episode"]), int(row["decision"]))] = int(row["row"])
    return table


@dataclass
class NavSample:
    row: int
    present: np.ndarray
    log_dist: np.ndarray
    ignore: np.ndarray
    biome: int
    time: int
    weather: int
    brightness: int
    s_vec: np.ndarray
    cond: np.ndarray
    geom: np.ndarray
    target_index: int
    mode: int
    turn: int
    forward: int
    stop: int
    used_wide: bool
    has_c: bool
    seed: int
    world_seed: int
    episode_id: str
    branch: str = ""
    forward_clipped: bool = False
    geom_clean: np.ndarray | None = None
    obs_source: str = "gt_noisy"


def _label_int(labels: dict | None, key: str, default: int) -> int:
    if labels is None:
        return int(default)
    value = labels.get(key)
    if value is None:
        return int(default)
    return int(value)


def _forward_class(labels: dict | None) -> tuple[int, bool]:
    if labels is None or labels.get("forward_index") is None:
        if labels is None or labels.get("forward") is None:
            return 0, False
        clipped, was = clip_forward_blocks(labels["forward"])
        return forward_index(clipped), was
    idx = int(labels["forward_index"])
    if 0 <= idx < len(FORWARD_BINS):
        return idx, False
    clipped, was = clip_forward_blocks(idx)
    return forward_index(clipped), was


def _geom_from_obs(obs) -> np.ndarray:
    if not obs.visible:
        return np.asarray([0.0, 0.0, 1.0, math.log(32.0)], np.float32)
    yaw = float(obs.bearing_deg or 0.0)
    rad = math.radians(yaw)
    log_d = float(obs.log_distance if obs.log_distance is not None else math.log(32.0))
    return np.asarray([1.0, math.sin(rad), math.cos(rad), log_d], np.float32)


def samples_from_episode(folder: Path, lookup: dict, spec: dict, cfg: dict, noise: TrackerNoise | None) -> list[NavSample]:
    packed = read_episode(folder)
    record = packed["record"]
    stem = episode_feature_name(record, Path(folder)).removesuffix(".npy")
    task = load_task(record, Path(folder), cfg)
    seed = int(record.get("seed") if record.get("seed") is not None else Path(folder).name)
    task_id = str(record.get("task_name") or task.get("id") or Path(folder).parts[-3])
    difficulty = str(record.get("difficulty") or Path(folder).parts[-2])
    episode_id = f"{task_id}/{difficulty}/{seed}"
    objects = list(spec["objects"])
    target = str(task.get("target") or "pig")
    target_index = objects.index(target) if target in objects else 0
    grid = tuple(spec["grid"])
    pool = int(spec["token_pool"])
    radius = float((cfg.get("judge") or {}).get("radius") or 3.0)
    states = replay(packed["steps"], max_steps=STEPS_NORM, noise=None)
    rng = random.Random(seed)
    rows = []
    for i, step in enumerate(packed["steps"]):
        decision = int(step["decision"] if step.get("decision") is not None else i)
        feat_row = lookup.get((stem, decision))
        if feat_row is None:
            continue
        scene = step.get("scene") or {}
        present, log_d, ignore = patch_targets(scene, objects, pool, grid)
        labels = labels_from_step(step, radius=radius)
        has_c = labels is not None
        used_wide = bool((labels or {}).get("used_wide"))
        fwd_i, fwd_clip = _forward_class(labels)
        if labels is not None:
            fwd_clip = bool(labels.get("forward_clipped")) or fwd_clip
        obs = observation_from_step(step)
        if noise is not None and has_c:
            obs = noise.apply_obs(obs, rng)
        s_vec = encode_s(states[i] if i < len(states) else states[-1])
        geom = _geom_from_obs(obs)
        rows.append(NavSample(
            row=int(feat_row),
            present=present,
            log_dist=log_d,
            ignore=ignore,
            biome=_argmax_index(scene.get("biome"), spec["biomes"]),
            time=_argmax_index(scene.get("time"), spec["times"]),
            weather=_argmax_index(scene.get("weather"), spec["weathers"]),
            brightness=_argmax_index(scene.get("brightness"), spec["brightness"]),
            s_vec=s_vec,
            cond=encode_match(step.get("match"), objects),
            geom=geom,
            target_index=target_index,
            mode=_label_int(labels, "mode", 0),
            turn=_label_int(labels, "turn_index", 0),
            forward=fwd_i,
            stop=_label_int(labels, "stop", 0),
            used_wide=used_wide,
            has_c=has_c,
            seed=seed,
            world_seed=seed,
            episode_id=episode_id,
            branch="" if step.get("expert_branch") is None else str(step.get("expert_branch")),
            forward_clipped=fwd_clip,
        ))
    return rows


VAL_SEEDS_DEFAULT = (1, 13, 23, 26, 37, 45)


def load_split(cfg: dict, feat_dir: Path) -> tuple[list[NavSample], list[NavSample], dict]:
    spec = spec_from_cfg(cfg)
    tokens, _, index = load_memmaps(feat_dir)
    lookup = _index_lookup(index)
    nav = cfg.get("c2nav") or {}
    roots = demo_roots(nav.get("demo_dirs"), {"c2": {"demo_dirs": nav.get("demo_dirs")}}) if nav.get("demo_dirs") else demo_roots(None, cfg)
    folders = []
    for root in roots:
        if Path(root).is_dir():
            folders.extend(iter_success_episodes(root))
    episodes, stats = select_training_episodes(folders, cfg)
    val_seeds = {int(s) for s in (nav.get("val_seeds") or VAL_SEEDS_DEFAULT)}
    noise_cfg = nav.get("noise") or {}
    noise = TrackerNoise(
        drop_visible=float(noise_cfg.get("drop_visible") or 0.0),
        bearing_std=float(noise_cfg.get("bearing_std") or 0.0),
        log_dist_std=float(noise_cfg.get("log_dist_std") or 0.0),
        odom_std=float(noise_cfg.get("odom_std") or 0.0),
    )
    train, val = [], []
    for folder in episodes:
        rows = samples_from_episode(folder, lookup, spec, cfg, noise)
        seed = int(Path(folder).name) if str(Path(folder).name).isdigit() else 0
        rec = read_episode(folder)["record"]
        seed = int(rec.get("seed") if rec.get("seed") is not None else seed)
        dest = val if seed in val_seeds else train
        dest.extend(rows)
    stats["train_rows"] = len(train)
    stats["val_rows"] = len(val)
    stats["grid"] = list(index.get("grid") or tokens.shape[1:3])
    stats["dim"] = int(index.get("dim") or tokens.shape[-1])
    print(f"c2nav samples train={len(train)} val={len(val)} episodes={len(episodes)}", flush=True)
    print_label_stats("train", train, spec)
    print_label_stats("val", val, spec)
    return train, val, stats


def print_label_stats(name: str, rows: list[NavSample], spec: dict) -> dict:
    from collections import Counter

    n = len(rows)
    ignored = sum(1 for r in rows if not r.has_c)
    wide = sum(1 for r in rows if r.has_c and r.used_wide)
    usable = [r for r in rows if r.has_c and not r.used_wide]
    mode_c = Counter(r.mode for r in usable)
    turn_c = Counter(r.turn for r in usable)
    fwd_c = Counter(r.forward for r in usable)
    branch_c = Counter(r.branch or "(empty)" for r in rows)
    mapped = {}
    for raw, cnt in sorted(branch_c.items()):
        idx, wide_b = mode_from_branch(None if raw == "(empty)" else raw, raw == "wide_turn")
        mapped[raw] = {"count": cnt, "mode": MODE_NAMES[idx], "mode_index": idx, "used_wide": wide_b}
    stop_pos = sum(r.stop for r in rows if r.has_c)
    stop_n = sum(1 for r in rows if r.has_c)
    n_clip = sum(1 for r in usable if r.forward_clipped)
    unmapped = Counter(r.branch for r in rows if r.branch and not branch_is_mapped(r.branch))
    pos_cells = 0
    tot_cells = 0
    for r in rows:
        m = r.ignore < 0.5
        tot_cells += int(r.present[:, m].size)
        pos_cells += int((r.present[:, m] > 0.5).sum())
    stats = {
        "n": n,
        "ignored_no_c": ignored,
        "used_wide": wide,
        "mode": {MODE_NAMES[i]: int(mode_c.get(i, 0)) for i in range(len(MODE_NAMES))},
        "turn": {str(i): int(turn_c.get(i, 0)) for i in range(N_TURN)},
        "forward": {str(FORWARD_BINS[i]): int(fwd_c.get(i, 0)) for i in range(len(FORWARD_BINS))},
        "stop_pos_frac": None if stop_n == 0 else stop_pos / stop_n,
        "present_pos_frac": None if tot_cells == 0 else pos_cells / tot_cells,
        "expert_branch": mapped,
        "forward_clipped": n_clip,
        "unmapped_branch": dict(unmapped),
    }
    print(f"c2nav labels {name} {json.dumps(stats, default=str)}", flush=True)
    return stats


def _presence_pos_weight(rows: list[NavSample], n_obj: int) -> np.ndarray:
    pos = np.zeros(n_obj, np.float64)
    neg = np.zeros(n_obj, np.float64)
    for row in rows:
        for k in range(n_obj):
            m = (row.ignore < 0.5)
            p = row.present[k] > 0.5
            pos[k] += float((p & m).sum())
            neg[k] += float((~p & m).sum())
    w = np.ones(n_obj, np.float32)
    for k in range(n_obj):
        if pos[k] > 0:
            w[k] = np.clip(neg[k] / max(pos[k], 1.0), 1.0, 50.0)
    return w


def class_counts(values: list[int], n: int) -> np.ndarray:
    counts = np.zeros(n, np.float64)
    for v in values:
        if 0 <= int(v) < n:
            counts[int(v)] += 1
    return counts


def class_weights(values: list[int], n: int, cap: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Unseen classes get weight 0. Seen: min(10, sqrt(N/(K n_c))), then mean 1."""
    counts = class_counts(values, n)
    w = np.zeros(n, np.float32)
    seen = counts > 0
    k = int(seen.sum())
    total = float(counts.sum())
    if k == 0 or total <= 0:
        return w, counts
    raw = np.sqrt(total / (k * counts[seen]))
    raw = np.minimum(raw, float(cap))
    mean = float(raw.mean())
    if mean > 0:
        raw = raw / mean
    w[seen] = raw.astype(np.float32)
    return w, counts


def stop_pos_weight(rows: list[NavSample]) -> float:
    n_pos = sum(int(r.stop) for r in rows if r.has_c)
    n_neg = sum(1 for r in rows if r.has_c) - n_pos
    if n_pos <= 0:
        raise RuntimeError("no stop positives in the training set")
    return float(min(10.0, n_neg / float(n_pos)))


def flip_sample(sample: NavSample) -> NavSample:
    return NavSample(
        row=sample.row,
        present=np.flip(sample.present, axis=-1).copy(),
        log_dist=np.flip(sample.log_dist, axis=-1).copy(),
        ignore=np.flip(sample.ignore, axis=-1).copy(),
        biome=sample.biome,
        time=sample.time,
        weather=sample.weather,
        brightness=sample.brightness,
        s_vec=np.asarray(flip_vector(list(sample.s_vec)), np.float32),
        cond=sample.cond,
        geom=np.asarray([sample.geom[0], -sample.geom[1], sample.geom[2], sample.geom[3]], np.float32),
        target_index=sample.target_index,
        mode=sample.mode,
        turn=flip_turn_index(sample.turn),
        forward=sample.forward,
        stop=sample.stop,
        used_wide=sample.used_wide,
        has_c=sample.has_c,
        seed=sample.seed,
        world_seed=sample.world_seed,
        episode_id=sample.episode_id,
        branch=sample.branch,
        forward_clipped=sample.forward_clipped,
        geom_clean=None if sample.geom_clean is None else np.asarray(sample.geom_clean, np.float32).copy(),
        obs_source=sample.obs_source,
    )


def collate(rows: list[NavSample], tokens, flip: bool, rng: random.Random) -> dict:
    chosen = []
    flips = []
    for row in rows:
        use = flip_sample(row) if flip and rng.random() < 0.5 else row
        chosen.append(use)
        flips.append(use is not row and flip)
    idx = [r.row for r in chosen]
    tok = rows_float32(tokens, idx)
    for i, do in enumerate(flips):
        if do:
            tok[i] = tok[i, :, ::-1, :].copy()
    return {
        "tokens": tok,
        "present": np.stack([r.present for r in chosen]),
        "log_dist": np.stack([r.log_dist for r in chosen]),
        "ignore": np.stack([r.ignore for r in chosen]),
        "biome": np.asarray([r.biome for r in chosen], np.int64),
        "time": np.asarray([r.time for r in chosen], np.int64),
        "weather": np.asarray([r.weather for r in chosen], np.int64),
        "brightness": np.asarray([r.brightness for r in chosen], np.int64),
        "s_vec": np.stack([r.s_vec for r in chosen]),
        "cond": np.stack([r.cond for r in chosen]),
        "geom": np.stack([r.geom for r in chosen]),
        "target_index": np.asarray([r.target_index for r in chosen], np.int64),
        "mode": np.asarray([r.mode for r in chosen], np.int64),
        "turn": np.asarray([r.turn for r in chosen], np.int64),
        "forward": np.asarray([r.forward for r in chosen], np.int64),
        "stop": np.asarray([r.stop for r in chosen], np.float32),
        "used_wide": np.asarray([r.used_wide for r in chosen], np.float32),
        "has_c": np.asarray([r.has_c for r in chosen], np.float32),
    }


def _mask_unseen_logits(logits, weight):
    import torch

    valid = weight > 0
    if bool(valid.all()):
        return logits
    fill = torch.full_like(logits, -1.0e9)
    return torch.where(valid.unsqueeze(0), logits, fill)


def nav_loss(out, batch, weights, pos_w, mode_w, turn_w, fwd_w, stop_pos, smooth: float):
    import torch
    from torch.nn import functional as F

    present = out["present_logit"]
    b, k, h, w = present.shape
    ignore = batch["ignore"].unsqueeze(1)
    valid = (ignore < 0.5).float()
    y = batch["present"]
    bce = F.binary_cross_entropy_with_logits(present, y, reduction="none")
    pw = pos_w.view(1, k, 1, 1)
    bce = torch.where(y > 0.5, bce * pw, bce)
    loss_p = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    mask_d = (y > 0.5) & (valid > 0.5)
    if mask_d.any():
        loss_d = F.smooth_l1_loss(out["log_dist"][mask_d], batch["log_dist"][mask_d])
    else:
        loss_d = present.new_zeros(())
    loss_biome = F.cross_entropy(out["biome"], batch["biome"])
    loss_time = F.cross_entropy(out["time"], batch["time"])
    loss_weather = F.cross_entropy(out["weather"], batch["weather"])
    loss_bright = F.cross_entropy(out["brightness"], batch["brightness"])
    loss_g = 0.25 * (loss_biome + loss_time + loss_weather + loss_bright)
    loss_a = loss_p + loss_d + loss_g
    c_mask = (batch["has_c"] > 0.5) & (batch["used_wide"] < 0.5)
    if c_mask.any():
        mode_logits = _mask_unseen_logits(out["mode"][c_mask], mode_w)
        turn_logits = _mask_unseen_logits(out["turn"][c_mask], turn_w)
        fwd_logits = _mask_unseen_logits(out["forward"][c_mask], fwd_w)
        loss_mode = F.cross_entropy(mode_logits, batch["mode"][c_mask], weight=mode_w, label_smoothing=smooth)
        loss_turn = F.cross_entropy(turn_logits, batch["turn"][c_mask], weight=turn_w, label_smoothing=smooth)
        loss_fwd = F.cross_entropy(fwd_logits, batch["forward"][c_mask], weight=fwd_w, label_smoothing=smooth)
        loss_wp = 0.5 * (loss_turn + loss_fwd)
    else:
        loss_mode = present.new_zeros(())
        loss_wp = present.new_zeros(())
        loss_turn = present.new_zeros(())
        loss_fwd = present.new_zeros(())
    if (batch["has_c"] > 0.5).any():
        stop_mask = batch["has_c"] > 0.5
        loss_stop = F.binary_cross_entropy_with_logits(
            out["stop"][stop_mask], batch["stop"][stop_mask], pos_weight=stop_pos,
        )
    else:
        loss_stop = present.new_zeros(())
    total = (
        float(weights["a"]) * loss_a
        + float(weights["mode"]) * loss_mode
        + float(weights["wp"]) * loss_wp
        + float(weights["stop"]) * loss_stop
    )
    return total, {
        "a": float(loss_a.detach().item()),
        "present": float(loss_p.detach().item()),
        "log_dist": float(loss_d.detach().item()),
        "global": float(loss_g.detach().item()),
        "biome": float(loss_biome.detach().item()),
        "time": float(loss_time.detach().item()),
        "weather": float(loss_weather.detach().item()),
        "brightness": float(loss_bright.detach().item()),
        "mode": float(loss_mode.detach().item()),
        "wp": float(loss_wp.detach().item()),
        "turn": float(loss_turn.detach().item()),
        "forward": float(loss_fwd.detach().item()),
        "stop": float(loss_stop.detach().item()),
        "total": float(total.detach().item()),
    }


def _to_torch(batch, device, bearings=None):
    import torch

    out = {}
    for key, value in batch.items():
        tensor = torch.from_numpy(np.asarray(value))
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        out[key] = tensor.to(device)
    if bearings is not None:
        out["bearings"] = torch.as_tensor(bearings, device=device, dtype=torch.float32)
    return out


def _binary_auc(y, p) -> float | None:
    y = np.asarray(y, dtype=np.int32)
    p = np.asarray(p, dtype=np.float64)
    if y.size == 0 or p.size == 0 or y.min() == y.max():
        return None
    order = np.argsort(p)
    y_sorted = y[order]
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = np.arange(1, y.size + 1, dtype=np.float64)
    return float((ranks[y_sorted == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def heads_out(model, batch):
    """A from tokens; C from GT geom so policy metrics match the training objective."""
    a = model.perception(batch["tokens"])
    c = model.policy(batch["geom"], batch["s_vec"], batch["cond"])
    out = dict(a)
    out.update(c)
    return out


def offline_metrics(model, rows: list[NavSample], tokens, device, spec: dict, cfg: dict, bearings) -> dict:
    import torch

    if not rows:
        return {
            "mode_acc": None,
            "mode_acc_by_name": {name: None for name in MODE_NAMES},
            "turn_acc": None,
            "stop_f1": None,
            "stop_auc": None,
            "present_auc": None,
            "log_dist_mae": None,
            "n": 0,
        }
    model.eval()
    rng = random.Random(0)
    y_mode, p_mode = [], []
    y_stop, p_stop = [], []
    y_turn, p_turn = [], []
    present_y, present_p = [], []
    dist_err = []
    bs = 16
    with torch.no_grad():
        for start in range(0, len(rows), bs):
            chunk = rows[start : start + bs]
            batch = _to_torch(collate(chunk, tokens, False, rng), device, bearings=bearings)
            out = heads_out(model, batch)
            for i, row in enumerate(chunk):
                if row.has_c and not row.used_wide:
                    y_mode.append(row.mode)
                    p_mode.append(int(out["mode"][i].argmax().item()))
                    y_turn.append(row.turn)
                    p_turn.append(int(out["turn"][i].argmax().item()))
                if row.has_c:
                    y_stop.append(int(row.stop))
                    p_stop.append(float(torch.sigmoid(out["stop"][i]).item()))
                mask = row.ignore < 0.5
                if mask.any():
                    present_y.append(row.present[:, mask].reshape(-1))
                    present_p.append(out["present_p"][i].detach().cpu().numpy()[:, mask].reshape(-1))
                    exist = (row.present > 0.5) & mask
                    if exist.any():
                        pred = out["log_dist"][i].detach().cpu().numpy()[exist]
                        dist_err.append(float(np.mean(np.abs(pred - row.log_dist[exist]))))
    def acc(y, p):
        return None if not y else float(np.mean(np.asarray(y) == np.asarray(p)))
    f1 = None
    if y_stop:
        pred = [1 if v >= 0.5 else 0 for v in p_stop]
        tp = sum(1 for a, b in zip(y_stop, pred) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_stop, pred) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_stop, pred) if a == 1 and b == 0)
        if tp + fp + fn == 0:
            f1 = None
        else:
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    mode_acc = {name: None for name in MODE_NAMES}
    if y_mode:
        y_mode_a = np.asarray(y_mode)
        p_mode_a = np.asarray(p_mode)
        for i, name in enumerate(MODE_NAMES):
            m = y_mode_a == i
            mode_acc[name] = None if not m.any() else float(np.mean(p_mode_a[m] == i))
    py = np.concatenate(present_y) if present_y else np.zeros(0)
    pp = np.concatenate(present_p) if present_p else np.zeros(0)
    return {
        "mode_acc": acc(y_mode, p_mode),
        "mode_acc_by_name": mode_acc,
        "turn_acc": acc(y_turn, p_turn),
        "stop_f1": f1,
        "stop_auc": _binary_auc(y_stop, p_stop),
        "present_auc": _binary_auc((py > 0.5).astype(int), pp) if py.size else None,
        "log_dist_mae": None if not dist_err else float(np.mean(dist_err)),
        "pred_mode": {MODE_NAMES[i]: int(p_mode.count(i)) for i in range(len(MODE_NAMES))} if p_mode else None,
        "n": len(rows),
        "n_mode": len(y_mode),
        "n_stop": len(y_stop),
    }


def save_checkpoint(path: Path, model, spec: dict, epoch: int, metrics: dict, extra: dict | None = None) -> None:
    import torch

    payload = structure_record(spec, model={k: v.detach().cpu() for k, v in model.state_dict().items()})
    payload["meta"] = {"epoch": int(epoch), "metrics": metrics, "git": git_hash()}
    if extra:
        payload["meta"].update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def overfit_check(cfg: dict, feat_dir: Path, seed: int = 0) -> dict:
    import torch

    spec = spec_from_cfg(cfg)
    spec["dropout"] = 0.0
    train, _val, _stats = load_split(cfg, feat_dir)
    if len(train) < 2:
        raise RuntimeError("need training samples for overfit check")
    rng = random.Random(seed)
    subset = [train[i] for i in rng.sample(range(len(train)), k=min(64, len(train)))]
    tokens, _, index = load_memmaps(feat_dir)
    spec["dim"] = int(index.get("dim") or spec["dim"])
    spec["grid"] = tuple(index.get("grid") or spec["grid"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = C2NavNet(spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    k = len(spec["objects"])
    pos_w = torch.ones(k, device=device)
    mode_w = torch.ones(len(MODE_NAMES), device=device)
    turn_w = torch.ones(N_TURN, device=device)
    fwd_w = torch.ones(len(FORWARD_BINS), device=device)
    stop_pos = torch.tensor(1.0, device=device)
    wts = {"a": 1.0, "mode": 1.0, "wp": 1.0, "stop": 1.0}
    bearings = patch_bearings(spec["grid"][0], spec["grid"][1], cfg)
    batch = _to_torch(collate(subset, tokens, False, random.Random(0)), device, bearings=bearings)
    model.train()
    out0 = heads_out(model, batch)
    before, parts0 = nav_loss(out0, batch, wts, pos_w, mode_w, turn_w, fwd_w, stop_pos, 0.0)
    before = float(before.item())
    print(f"overfit loss_init {json.dumps(parts0)}", flush=True)
    for step in range(200):
        opt.zero_grad()
        out_b = heads_out(model, batch)
        loss, parts = nav_loss(out_b, batch, wts, pos_w, mode_w, turn_w, fwd_w, stop_pos, 0.0)
        loss.backward()
        opt.step()
        if step in (0, 49, 99, 199):
            print(f"overfit step={step} loss={float(loss.item()):.4f} parts={json.dumps(parts)}", flush=True)
    metrics = offline_metrics(model, subset, tokens, device, spec, cfg, bearings)
    after = float(parts["total"])
    print(f"overfit done before={before:.4f} after={after:.4f} metrics={json.dumps(metrics, default=str)}", flush=True)
    return {"before": before, "after": after, "metrics": metrics}


def dry_run_torch() -> float:
    import torch

    cfg = load_config()
    spec = spec_from_cfg(cfg)
    spec["dim"] = 16
    spec["grid"] = (4, 6)
    spec["hidden"] = 32
    spec["channels"] = 8
    b, h, w, d = 8, 4, 6, 16
    k = len(spec["objects"])
    model = C2NavNet(spec)
    tokens = torch.randn(b, h, w, d)
    batch = {
        "tokens": tokens,
        "present": torch.zeros(b, k, h, w),
        "log_dist": torch.zeros(b, k, h, w),
        "ignore": torch.zeros(b, h, w),
        "biome": torch.zeros(b, dtype=torch.long),
        "time": torch.zeros(b, dtype=torch.long),
        "weather": torch.zeros(b, dtype=torch.long),
        "brightness": torch.zeros(b, dtype=torch.long),
        "s_vec": torch.zeros(b, spec["s_dim"]),
        "cond": torch.zeros(b, spec["cond_dim"]),
        "geom": torch.zeros(b, 4),
        "target_index": torch.zeros(b, dtype=torch.long),
        "bearings": torch.zeros(h, w),
        "mode": torch.zeros(b, dtype=torch.long),
        "turn": torch.zeros(b, dtype=torch.long),
        "forward": torch.ones(b, dtype=torch.long),
        "stop": torch.zeros(b),
        "used_wide": torch.zeros(b),
        "has_c": torch.ones(b),
    }
    batch["present"][:, 0, 1, 2] = 1.0
    out = model(batch)
    pos_w = torch.ones(k)
    mode_w = torch.ones(len(MODE_NAMES))
    turn_w = torch.ones(N_TURN)
    fwd_w = torch.ones(len(FORWARD_BINS))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    wts = {"a": 1.0, "mode": 1.0, "wp": 1.0, "stop": 1.0}
    before = float(nav_loss(out, batch, wts, pos_w, mode_w, turn_w, fwd_w, torch.tensor(1.0), 0.1)[0].item())
    for _ in range(8):
        opt.zero_grad()
        out = model(batch)
        loss, _ = nav_loss(out, batch, wts, pos_w, mode_w, turn_w, fwd_w, torch.tensor(1.0), 0.1)
        loss.backward()
        opt.step()
    out = model(batch)
    after = float(nav_loss(out, batch, wts, pos_w, mode_w, turn_w, fwd_w, torch.tensor(1.0), 0.1)[0].item())
    print(f"dry-run c2nav torch {before:.4f} -> {after:.4f}", flush=True)
    payload = structure_record(spec, model=model.state_dict())
    if "meta" not in payload:
        payload["meta"] = {"epoch": 0, "metrics": {}}
    return after


def train_loop(cfg: dict, feat_dir: Path, out: Path, epochs: int, batch_size: int, lr: float, seed: int, cfg_path: Path | None = None) -> Path:
    import torch

    spec = spec_from_cfg(cfg)
    train, val, stats = load_split(cfg, feat_dir)
    if not train:
        raise RuntimeError("no c2nav training samples")
    tokens, _, index = load_memmaps(feat_dir)
    spec["dim"] = int(index.get("dim") or spec["dim"])
    spec["grid"] = tuple(index.get("grid") or spec["grid"])
    spec["token_pool"] = int(index.get("token_pool") or spec["token_pool"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = C2NavNet(spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float((cfg.get("c2nav") or {}).get("weight_decay") or 0.05))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    pos_w = torch.as_tensor(_presence_pos_weight(train, len(spec["objects"])), device=device)
    usable = [r for r in train if r.has_c and not r.used_wide]
    mode_w, mode_n = class_weights([r.mode for r in usable], len(MODE_NAMES))
    turn_w, turn_n = class_weights([r.turn for r in usable], N_TURN)
    fwd_w, fwd_n = class_weights([r.forward for r in usable], len(FORWARD_BINS))
    mode_w = torch.as_tensor(mode_w, device=device)
    turn_w = torch.as_tensor(turn_w, device=device)
    fwd_w = torch.as_tensor(fwd_w, device=device)
    stop_scale = stop_pos_weight(train)
    stop_pos = torch.tensor(stop_scale, device=device)
    wts = dict((cfg.get("c2nav") or {}).get("loss_weights") or {"a": 1, "mode": 1, "wp": 1, "stop": 1})
    smooth = float((cfg.get("c2nav") or {}).get("label_smoothing") or 0.1)
    bearings = patch_bearings(spec["grid"][0], spec["grid"][1], cfg)
    print(
        "c2nav weights "
        + json.dumps({
            "present_pos": pos_w.detach().cpu().tolist(),
            "mode": mode_w.detach().cpu().tolist(),
            "mode_n": mode_n.tolist(),
            "turn": turn_w.detach().cpu().tolist(),
            "turn_n": turn_n.tolist(),
            "forward": fwd_w.detach().cpu().tolist(),
            "forward_n": fwd_n.tolist(),
            "stop_pos_weight": stop_scale,
        }),
        flush=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    last = out / "last.pt"
    first_parts = None
    last_parts = None
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        running = 0.0
        part_sum = {}
        n_b = 0
        for start in range(0, len(order), batch_size):
            chunk = [train[i] for i in order[start : start + batch_size]]
            batch = _to_torch(collate(chunk, tokens, True, rng), device, bearings=bearings)
            opt.zero_grad()
            out_b = heads_out(model, batch)
            loss, parts = nav_loss(out_b, batch, wts, pos_w, mode_w, turn_w, fwd_w, stop_pos, smooth)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_b += 1
            for key, val in parts.items():
                part_sum[key] = part_sum.get(key, 0.0) + float(val)
        sched.step()
        avg_parts = {key: val / max(n_b, 1) for key, val in part_sum.items()}
        if first_parts is None:
            first_parts = avg_parts
            print(f"c2nav loss epoch_1 {json.dumps(first_parts)}", flush=True)
        last_parts = avg_parts
        metrics = offline_metrics(model, val or train[: min(64, len(train))], tokens, device, spec, cfg, bearings)
        metrics["train_loss"] = running / max(n_b, 1)
        metrics["loss_parts"] = avg_parts
        ckpt = out / f"epoch_{epoch:02d}.pt"
        save_checkpoint(ckpt, model, spec, epoch, metrics, extra={"filter": stats})
        save_checkpoint(last, model, spec, epoch, metrics, extra={"filter": stats})
        print(f"epoch {epoch} loss={metrics['train_loss']:.4f} metrics={json.dumps(metrics, default=str)}", flush=True)
    if last_parts is not None:
        print(f"c2nav loss last_epoch {json.dumps(last_parts)}", flush=True)
    return last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", required=False, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overfit", action="store_true", help="64 samples, 200 steps, expect loss~0 acc~1")
    args = parser.parse_args()
    if args.dry_run:
        try:
            dry_run_torch()
        except ImportError:
            print("dry-run c2nav skipped (no torch)", flush=True)
        return
    if args.overfit:
        cfg_path = Path(args.config).resolve() if args.config else None
        cfg = load_config(cfg_path) if cfg_path else load_config()
        nav = cfg.get("c2nav") or {}
        feat = Path(args.features) if args.features else Path(os.environ.get("SCR") or f"/scratch/{os.environ.get('USER', 'user')}") / "lfnav" / "features" / str(nav.get("features_version") or "nav_v3_easy")
        overfit_check(cfg, feat, int(args.seed))
        return
    cfg_path = Path(args.config).resolve() if args.config else None
    cfg = load_config(cfg_path) if cfg_path else load_config()
    nav = cfg.get("c2nav") or {}
    feat = Path(args.features) if args.features else Path(os.environ.get("SCR") or f"/scratch/{os.environ.get('USER', 'user')}") / "lfnav" / "features" / str(nav.get("features_version") or "joint_v1")
    out = Path(args.out) if args.out else Path(os.environ.get("SCR") or f"/scratch/{os.environ.get('USER', 'user')}") / "lfnav" / "ckpt" / "c2nav_easy"
    epochs = int(args.epochs if args.epochs is not None else nav.get("epochs") or 20)
    batch = int(args.batch if args.batch is not None else nav.get("batch") or 16)
    lr = float(args.lr if args.lr is not None else nav.get("lr") or 3e-4)
    train_loop(cfg, feat, out, epochs, batch, lr, int(args.seed), cfg_path=cfg_path)


if __name__ == "__main__":
    main()
