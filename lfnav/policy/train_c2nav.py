"""Train C2Nav. A and C share demo_dirs from the config. --dry-run needs no features."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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
    PRESENT_POS_CAP,
    VIS_POS_CAP,
    a_extra_demos_of,
    a_extra_features_of,
    a_extra_per_epoch_of,
    ckpt_dir_of,
    encode_match,
    encode_s,
    far_class_ignore,
    geom_from_obs,
    gt_target_obs,
    horizontal_fov_deg,
    patch_bearings,
    patch_targets,
    pool_to_grid,
    spec_from_cfg,
    structure_record,
    torch_threads_of,
    vis_labels,
    wide_radar_field,
)
from lfnav.policy.c2nav_labels import (
    FORWARD_BINS,
    MODE_NAMES,
    N_TURN,
    branch_is_mapped,
    flip_turn_index,
    forward_index,
    labels_from_step,
    mode_from_branch,
)
from lfnav.policy.state_tracker import (
    STEPS_NORM,
    TargetObs,
    TrackerNoise,
    apply_c_obs_noise,
    flip_vector,
    replay,
)


def git_hash() -> str | None:
    git_dir = _ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = git_dir / head.split(":", 1)[1].strip()
            return ref.read_text().strip() or None
        return head or None
    except Exception:
        return None


def _argmax_index(table, names, empty=0) -> int:
    vals = [float((table or {}).get(n) or 0.0) for n in names]
    if not vals or max(vals) <= 0.0:
        return empty
    return int(np.argmax(vals))




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
    vis: np.ndarray | None = None
    vis_ignore: np.ndarray | None = None
    a_only: bool = False
    feat_store: int = 0


def _label_int(labels: dict | None, key: str, default: int) -> int:
    if labels is None:
        return int(default)
    value = labels.get(key)
    if value is None:
        return int(default)
    return int(value)


def policy_supervised(row) -> bool:
    if bool(row.has_c) is False:
        return False
    if bool(row.used_wide) is True:
        return False
    if int(row.stop) != 0:
        return False
    return True


def _forward_class(labels: dict | None) -> tuple[int, bool]:
    if labels is None or labels.get("forward_index") is None:
        if labels is None or labels.get("forward") is None:
            return 1, False
        return forward_index(labels["forward"]), False
    idx = int(labels["forward_index"])
    if idx < 0 or idx >= len(FORWARD_BINS):
        raise RuntimeError(f"forward class {idx} outside 0..{len(FORWARD_BINS) - 1}")
    return idx, False


_geom_from_obs = geom_from_obs


def _pose_noise(noise: TrackerNoise | None) -> TrackerNoise | None:
    if noise is None:
        return None
    return TrackerNoise(yaw_std=float(noise.yaw_std), disp_std=float(noise.disp_std), odom_std=float(noise.odom_std))


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
    judge = cfg.get("judge") if cfg.get("judge") is not None else {}
    radius_raw = judge.get("radius")
    radius = 3.0 if radius_raw is None else float(radius_raw)
    bearings = patch_bearings(int(grid[0]), int(grid[1]), cfg)
    rng = random.Random(seed)
    parsed = []
    gt_obs: list[TargetObs] = []
    hfov = horizontal_fov_deg(cfg)
    for i, step in enumerate(packed["steps"]):
        decision = int(step["decision"] if step.get("decision") is not None else i)
        feat_row = lookup.get((stem, decision))
        scene = step.get("scene") if step.get("scene") is not None else {}
        present, log_d, ignore = patch_targets(scene, objects, pool, grid)
        class_ig = far_class_ignore(step, objects, hfov)
        vis, vis_ig = vis_labels(present, ignore, class_ig)
        obs_clean = gt_target_obs(present, ignore, target_index, bearings, step.get("target_distance_num"))
        gt_obs.append(obs_clean)
        parsed.append((i, step, feat_row, present, log_d, ignore, obs_clean, vis, vis_ig))
    if noise is None:
        noisy_obs = list(gt_obs)
    else:
        noisy_obs = apply_c_obs_noise(gt_obs, packed["steps"], noise, rng)
    states = replay(
        packed["steps"],
        max_steps=STEPS_NORM,
        noise=_pose_noise(noise),
        rng=rng,
        observations=noisy_obs,
    )
    rows = []
    source = "gt_noisy"
    last_feat = None
    for i, step, feat_row, present, log_d, ignore, obs_clean, vis, vis_ig in parsed:
        if feat_row is None:
            if last_feat is None:
                continue
            feat_row = last_feat
        else:
            last_feat = feat_row
        labels = labels_from_step(step, radius=radius)
        has_c = labels is not None
        used_wide = bool((labels or {}).get("used_wide"))
        fwd_i, fwd_clip = _forward_class(labels)
        if labels is not None:
            fwd_clip = bool(labels.get("forward_clipped")) or fwd_clip
        obs = noisy_obs[i]
        s_vec = encode_s(states[i] if i < len(states) else states[-1])
        geom = _geom_from_obs(obs)
        geom_clean = _geom_from_obs(obs_clean)
        scene = step.get("scene") if step.get("scene") is not None else {}
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
            geom_clean=geom_clean,
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
            obs_source=source,
            vis=vis,
            vis_ignore=vis_ig,
            a_only=False,
        ))
    return rows


VAL_SEEDS_DEFAULT = (1, 13, 23, 26, 37, 45)


def _nav_dict(cfg: dict | None) -> dict:
    if cfg is None:
        return {}
    nav = cfg.get("c2nav")
    if nav is None:
        return {}
    return nav


def _noise_from_cfg(cfg: dict) -> TrackerNoise:
    nav = _nav_dict(cfg)
    noise_cfg = nav.get("noise")
    if noise_cfg is None:
        noise_cfg = {}
    drop_visible = float(noise_cfg["drop_visible"] if noise_cfg.get("drop_visible") is not None else 0.15)
    if drop_visible > 0.15:
        drop_visible = 0.15
    p_fp = noise_cfg.get("p_fp_start")
    return TrackerNoise(
        drop_visible=drop_visible,
        bearing_std=float(noise_cfg["bearing_std"] if noise_cfg.get("bearing_std") is not None else 3.0),
        log_dist_std=float(noise_cfg["log_dist_std"] if noise_cfg.get("log_dist_std") is not None else 0.1),
        yaw_std=float(noise_cfg["yaw_std"] if noise_cfg.get("yaw_std") is not None else 1.0),
        disp_std=float(noise_cfg["disp_std"] if noise_cfg.get("disp_std") is not None else 0.05),
        p_fp_start=0.03 if p_fp is None else float(p_fp),
        hfov_deg=float(horizontal_fov_deg(cfg)),
    )


def collect_log_dist_bank(episodes, spec: dict, cfg: dict, val_seeds: set[int]) -> list[float]:
    bank = []
    objects = list(spec["objects"])
    grid = tuple(spec["grid"])
    pool = int(spec["token_pool"])
    bearings = patch_bearings(int(grid[0]), int(grid[1]), cfg)
    for folder in episodes:
        packed = read_episode(folder)
        rec = packed["record"]
        seed = int(rec.get("seed") if rec.get("seed") is not None else Path(folder).name)
        if seed in val_seeds:
            continue
        task = load_task(rec, Path(folder), cfg)
        target = str(task.get("target") if task.get("target") is not None else "pig")
        t_i = objects.index(target) if target in objects else 0
        for step in packed["steps"]:
            scene = step.get("scene") if step.get("scene") is not None else {}
            present, _log_d, ignore = patch_targets(scene, objects, pool, grid)
            obs = gt_target_obs(present, ignore, t_i, bearings, step.get("target_distance_num"))
            if obs.visible and obs.log_distance is not None:
                bank.append(float(obs.log_distance))
    return bank


def count_stop_neg_invisible_close(rows: list[NavSample], max_dist: float = 3.5) -> dict:
    n_close_inv = 0
    n_pos = 0
    for row in rows:
        if not row.has_c:
            continue
        vis = 0.0
        if row.geom_clean is not None:
            vis = float(row.geom_clean[0])
        elif row.geom is not None:
            vis = float(row.geom[0])
        log_d = float(row.s_vec[2]) if row.s_vec is not None and len(row.s_vec) > 2 else 0.0
        dist = math.exp(log_d)
        if vis < 0.5 and dist <= float(max_dist):
            n_close_inv += 1
            if int(row.stop) != 0:
                n_pos += 1
    return {"n": n_close_inv, "stop_pos": n_pos, "all_neg": n_pos == 0}


def load_split(cfg: dict, feat_dir: Path, apply_noise: bool = True) -> tuple[list[NavSample], list[NavSample], dict]:
    spec = spec_from_cfg(cfg)
    tokens, _, index = load_memmaps(feat_dir)
    lookup = _index_lookup(index)
    nav = _nav_dict(cfg)
    demo_dirs = nav.get("demo_dirs")
    if demo_dirs:
        roots = demo_roots(demo_dirs, {"c2": {"demo_dirs": demo_dirs}})
    else:
        roots = demo_roots(None, cfg)
    folders = []
    for root in roots:
        if Path(root).is_dir():
            folders.extend(iter_success_episodes(root))
    episodes, stats = select_training_episodes(folders, cfg)
    val_raw = nav.get("val_seeds")
    val_seeds = {int(s) for s in (VAL_SEEDS_DEFAULT if val_raw is None else val_raw)}
    train_noise = _noise_from_cfg(cfg)
    if apply_noise:
        train_noise.log_dist_bank = collect_log_dist_bank(episodes, spec, cfg, val_seeds)
    train, val = [], []
    for folder in episodes:
        seed = int(Path(folder).name) if str(Path(folder).name).isdigit() else 0
        rec = read_episode(folder)["record"]
        seed = int(rec.get("seed") if rec.get("seed") is not None else seed)
        is_val = seed in val_seeds
        use_noise = train_noise if (apply_noise and not is_val) else None
        rows = samples_from_episode(folder, lookup, spec, cfg, use_noise)
        dest = val if is_val else train
        dest.extend(rows)
    stats["train_rows"] = len(train)
    stats["val_rows"] = len(val)
    stats["grid"] = list(index.get("grid") if index.get("grid") is not None else tokens.shape[1:3])
    stats["dim"] = int(index.get("dim") if index.get("dim") is not None else tokens.shape[-1])
    close_inv = count_stop_neg_invisible_close(train)
    stats["stop_neg_invisible_close"] = close_inv
    print(f"c2nav samples train={len(train)} val={len(val)} episodes={len(episodes)}", flush=True)
    print(f"c2nav stop-neg invisible belief<=3.5 n={close_inv['n']} stop_pos={close_inv['stop_pos']} all_neg={close_inv['all_neg']}", flush=True)
    print_label_stats("train", train, spec)
    print_label_stats("val", val, spec)
    return train, val, stats


def check_a_extra_format(feat_dir: Path, spec: dict) -> dict:
    tokens, _pooled, index = load_memmaps(feat_dir)
    if tokens.ndim != 4:
        raise RuntimeError(f"{feat_dir} tokens ndim={tokens.ndim}, expected 4")
    expect = (8, 13, 384)
    got = tuple(int(x) for x in tokens.shape[1:])
    if got != expect:
        raise RuntimeError(f"{feat_dir} tokens grid/dim {got}, expected {expect}")
    if str(tokens.dtype) not in ("float16", "float32"):
        raise RuntimeError(f"{feat_dir} tokens dtype={tokens.dtype}, expected float16")
    objs = index.get("objects")
    if objs is not None and list(objs) != list(spec["objects"]):
        raise RuntimeError(f"{feat_dir} objects {objs} != {spec['objects']}")
    return {"n": int(tokens.shape[0]), "shape": list(tokens.shape), "dtype": str(tokens.dtype)}


def _assert_old_patch_hw(scene: dict, folder: Path) -> None:
    patches = scene.get("patches")
    ignore = scene.get("ignore")
    if patches is None:
        return
    sample_p = next(iter(patches.values())) if isinstance(patches, dict) and patches else patches
    arr = np.asarray(sample_p)
    if arr.ndim != 2 or tuple(int(x) for x in arr.shape) != (16, 26):
        raise RuntimeError(f"old demo patches shape {arr.shape}, expected (16, 26) in {folder}")
    if ignore is None:
        return
    ig = np.asarray(ignore)
    if ig.ndim == 2 and tuple(int(x) for x in ig.shape) != (16, 26):
        raise RuntimeError(f"old demo ignore shape {ig.shape}, expected (16, 26) in {folder}")


def load_a_extra(cfg: dict, spec: dict) -> tuple[list[NavSample], list[NavSample]]:
    feat_dirs = a_extra_features_of(cfg)
    if not feat_dirs:
        return [], []
    demo_dirs = a_extra_demos_of(cfg)
    if not demo_dirs:
        raise RuntimeError("c2nav.a_extra_features is set but c2nav.a_extra_demos is empty")
    nav = _nav_dict(cfg)
    val_raw = nav.get("val_seeds")
    val_seeds = {int(s) for s in (VAL_SEEDS_DEFAULT if val_raw is None else val_raw)}
    train, val = [], []
    for feat_dir in feat_dirs:
        path = Path(feat_dir)
        check_a_extra_format(path, spec)
        _tokens, _, index = load_memmaps(path)
        lookup = _index_lookup(index)
        folders = []
        for root in demo_dirs:
            root_p = Path(root)
            if root_p.is_dir():
                folders.extend(iter_success_episodes(root_p))
        for folder in folders:
            packed = read_episode(folder)
            rec = packed["record"]
            seed = int(rec.get("seed") if rec.get("seed") is not None else Path(folder).name)
            for step in packed["steps"]:
                scene = step.get("scene") if step.get("scene") is not None else {}
                _assert_old_patch_hw(scene, Path(folder))
            rows = samples_from_episode(folder, lookup, spec, cfg, None)
            store_i = 1 + feat_dirs.index(feat_dir)
            for row in rows:
                row.a_only = True
                row.feat_store = store_i
            dest = val if seed in val_seeds else train
            dest.extend(rows)
    print(f"c2nav a_extra train={len(train)} val={len(val)}", flush=True)
    return train, val


def print_label_stats(name: str, rows: list[NavSample], spec: dict) -> dict:
    from collections import Counter

    n = len(rows)
    ignored = sum(1 for r in rows if not r.has_c)
    wide = sum(1 for r in rows if r.has_c and r.used_wide)
    usable = [r for r in rows if policy_supervised(r)]
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
        vis_ig = row.vis_ignore
        for k in range(n_obj):
            if vis_ig is not None and k < len(vis_ig) and float(vis_ig[k]) >= 0.5:
                continue
            m = (row.ignore < 0.5)
            p = row.present[k] > 0.5
            pos[k] += float((p & m).sum())
            neg[k] += float((~p & m).sum())
    w = np.ones(n_obj, np.float32)
    for k in range(n_obj):
        if pos[k] > 0:
            w[k] = min(float(PRESENT_POS_CAP), neg[k] / max(pos[k], 1.0))
    return w


def _vis_pos_weight(rows: list[NavSample], n_obj: int) -> np.ndarray:
    pos = np.zeros(n_obj, np.float64)
    neg = np.zeros(n_obj, np.float64)
    for row in rows:
        vis, vis_ig = _row_vis(row, n_obj)
        for k in range(n_obj):
            if float(vis_ig[k]) >= 0.5:
                continue
            if float(vis[k]) > 0.5:
                pos[k] += 1.0
            else:
                neg[k] += 1.0
    w = np.ones(n_obj, np.float32)
    for k in range(n_obj):
        if pos[k] > 0:
            w[k] = min(float(VIS_POS_CAP), neg[k] / max(pos[k], 1.0))
    return w


def _row_vis(row: NavSample, n_obj: int) -> tuple[np.ndarray, np.ndarray]:
    if row.vis is None:
        return vis_labels(row.present, row.ignore)
    vis = np.asarray(row.vis, np.float32).reshape(-1)
    if row.vis_ignore is None:
        vis_ig = np.zeros((n_obj,), np.float32)
    else:
        vis_ig = np.asarray(row.vis_ignore, np.float32).reshape(-1)
    if vis.shape[0] < n_obj:
        vis = np.pad(vis, (0, n_obj - vis.shape[0]))
    if vis_ig.shape[0] < n_obj:
        vis_ig = np.pad(vis_ig, (0, n_obj - vis_ig.shape[0]))
    return vis[:n_obj], vis_ig[:n_obj]


def class_counts(values: list[int], n: int) -> np.ndarray:
    counts = np.zeros(n, np.float64)
    for v in values:
        if 0 <= int(v) < n:
            counts[int(v)] += 1
    return counts


def class_weights(values: list[int], n: int, cap: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """w_c = min(cap, sqrt(N/(K n_c))), then mean 1. Empty class raises."""
    counts = class_counts(values, n)
    total = float(counts.sum())
    empty = [i for i in range(int(n)) if counts[i] == 0]
    if len(empty) != 0 or total <= 0 or int(n) <= 0:
        raise RuntimeError(f"empty class counts={counts.tolist()} N={total} K={int(n)} empty={empty}")
    raw = np.sqrt(total / (float(n) * counts))
    raw = np.minimum(raw, float(cap))
    mean = float(raw.mean())
    if mean > 0:
        raw = raw / mean
    return raw.astype(np.float32), counts


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
        vis=None if sample.vis is None else np.flip(np.asarray(sample.vis), axis=-1).copy() if np.asarray(sample.vis).ndim == 2 else np.asarray(sample.vis).copy(),
        vis_ignore=None if sample.vis_ignore is None else np.asarray(sample.vis_ignore).copy(),
        a_only=sample.a_only,
        feat_store=sample.feat_store,
    )


def collate(rows: list[NavSample], tokens, flip: bool, rng: random.Random) -> dict:
    chosen = []
    flips = []
    for row in rows:
        use = flip_sample(row) if flip and rng.random() < 0.5 else row
        chosen.append(use)
        flips.append(use is not row and flip)
    if isinstance(tokens, (list, tuple)):
        tok = np.stack([np.asarray(tokens[int(r.feat_store)][int(r.row)], np.float32) for r in chosen])
    else:
        tok = rows_float32(tokens, [r.row for r in chosen])
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
        "obs_source": [r.obs_source for r in chosen],
        "geom_clean": np.stack([
            r.geom if r.geom_clean is None else r.geom_clean for r in chosen
        ]),
        "vis": np.stack([_row_vis(r, chosen[0].present.shape[0])[0] for r in chosen]),
        "vis_ignore": np.stack([_row_vis(r, chosen[0].present.shape[0])[1] for r in chosen]),
        "a_only": np.asarray([1.0 if r.a_only else 0.0 for r in chosen], np.float32),
    }


def _mask_unseen_logits(logits, weight):
    import torch

    valid = weight > 0
    if bool(valid.all()):
        return logits
    fill = torch.full_like(logits, -1.0e9)
    return torch.where(valid.unsqueeze(0), logits, fill)


def nav_loss(out, batch, weights, pos_w, mode_w, turn_w, fwd_w, stop_pos, smooth: float, vis_w=None):
    import torch
    from torch.nn import functional as F

    skip_a = float(weights["a"] if weights.get("a") is not None else 1.0) == 0.0
    if skip_a:
        z = out["mode"].new_zeros(())
        loss_p = loss_d = loss_g = loss_vis = loss_a = z
        loss_biome = loss_time = loss_weather = loss_bright = z
        present = None
    else:
        present = out["present_logit"]
        b, k, h, w = present.shape
        ignore = batch["ignore"].unsqueeze(1)
        class_ok = (batch["vis_ignore"] < 0.5).float().view(b, k, 1, 1)
        valid = (ignore < 0.5).float() * class_ok
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
        loss_g = torch.stack([loss_biome, loss_time, loss_weather, loss_bright]).mean()
        y_vis = batch["vis"]
        v_ig = batch["vis_ignore"]
        valid_v = (v_ig < 0.5).float()
        bce_v = F.binary_cross_entropy_with_logits(out["vis_logit"], y_vis, reduction="none")
        if vis_w is None:
            vis_w = torch.ones(k, device=present.device, dtype=present.dtype)
        bce_v = torch.where(y_vis > 0.5, bce_v * vis_w.view(1, k), bce_v)
        loss_vis = (bce_v * valid_v).sum() / valid_v.sum().clamp_min(1.0)
        loss_a = loss_p + loss_d + loss_g + loss_vis
        z = present.new_zeros(())
    a_only = batch.get("a_only")
    if a_only is None:
        a_only = z
    c_ok = (batch["has_c"] > 0.5) & (batch["used_wide"] < 0.5) & (batch["stop"] < 0.5) & (a_only < 0.5)
    if c_ok.any():
        mode_logits = _mask_unseen_logits(out["mode"][c_ok], mode_w)
        turn_logits = _mask_unseen_logits(out["turn"][c_ok], turn_w)
        fwd_logits = _mask_unseen_logits(out["forward"][c_ok], fwd_w)
        loss_mode = F.cross_entropy(mode_logits, batch["mode"][c_ok], weight=mode_w, label_smoothing=smooth)
        loss_turn = F.cross_entropy(turn_logits, batch["turn"][c_ok], weight=turn_w, label_smoothing=smooth)
        loss_fwd = F.cross_entropy(fwd_logits, batch["forward"][c_ok], weight=fwd_w, label_smoothing=smooth)
        loss_wp = (loss_turn + loss_fwd) / 2.0
    else:
        loss_mode = z
        loss_wp = z
        loss_turn = z
        loss_fwd = z
    stop_ok = (batch["has_c"] > 0.5) & (a_only < 0.5)
    if stop_ok.any():
        loss_stop = F.binary_cross_entropy_with_logits(
            out["stop"][stop_ok], batch["stop"][stop_ok], pos_weight=stop_pos,
        )
    else:
        loss_stop = z
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
        "vis": float(loss_vis.detach().item()),
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

    skip = {"obs_source"}
    out = {}
    for key, value in batch.items():
        if key in skip:
            out[key] = value
            continue
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


def heads_out(model, batch, skip_a: bool = False):
    """A from tokens; C from GT geom so policy metrics match the training objective."""
    if skip_a:
        return model.policy(batch["geom"], batch["s_vec"], batch["cond"])
    a = model.perception(batch["tokens"])
    c = model.policy(batch["geom"], batch["s_vec"], batch["cond"])
    out = dict(a)
    out.update(c)
    return out


def _pr_fpr(y, p, tau: float) -> dict:
    y = np.asarray(y, dtype=np.int32)
    p = np.asarray(p, dtype=np.float64)
    if y.size == 0:
        return {"precision": None, "recall": None, "fpr": None, "n": 0, "tau": float(tau)}
    pred = p >= float(tau)
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & (~pred)).sum())
    tn = int(((y == 0) & (~pred)).sum())
    prec = None if (tp + fp) == 0 else tp / float(tp + fp)
    rec = None if (tp + fn) == 0 else tp / float(tp + fn)
    fpr = None if (fp + tn) == 0 else fp / float(fp + tn)
    return {"precision": prec, "recall": rec, "fpr": fpr, "n": int(y.size), "tau": float(tau)}


def _best_stop_f1(y, p) -> dict:
    y = np.asarray(y, dtype=np.int32)
    p = np.asarray(p, dtype=np.float64)
    if y.size == 0 or int((y == 1).sum()) == 0:
        return {"f1": None, "tau": None, "n": int(y.size)}
    best = {"f1": -1.0, "tau": 0.5, "n": int(y.size)}
    for tau in np.linspace(0.05, 0.95, 19):
        rec = _pr_fpr(y, p, float(tau))
        prec = rec["precision"]
        recall = rec["recall"]
        if prec is None or recall is None:
            continue
        f1 = 2.0 * prec * recall / max(prec + recall, 1e-9)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "tau": float(tau), "n": int(y.size)}
    if best["f1"] < 0:
        return {"f1": None, "tau": None, "n": int(y.size)}
    return best


def _metric(value, n: int) -> dict:
    return {"value": None if n <= 0 else value, "n": int(n)}


def offline_metrics(model, rows: list[NavSample], tokens, device, spec: dict, cfg: dict, bearings, warn_empty_stop: bool = True, skip_a: bool = False) -> dict:
    import torch

    empty_mode = {name: _metric(None, 0) for name in MODE_NAMES}
    if not rows:
        return {
            "mode_acc": _metric(None, 0),
            "mode_acc_by_name": empty_mode,
            "turn_acc": _metric(None, 0),
            "forward_acc": _metric(None, 0),
            "stop_f1": _metric(None, 0),
            "stop_auc": _metric(None, 0),
            "present_auc": _metric(None, 0),
            "log_dist_mae": _metric(None, 0),
            "n": 0,
        }
    model.eval()
    rng = random.Random(0)
    y_mode, p_mode = [], []
    y_stop, p_stop = [], []
    y_turn, p_turn = [], []
    y_fwd, p_fwd = [], []
    y_turn_by_mode = {name: ([], []) for name in MODE_NAMES}
    y_fwd_by_mode = {name: ([], []) for name in MODE_NAMES}
    present_y, present_p = [], []
    vis_y, vis_p = [], []
    vis_y_tgt, vis_p_tgt = [], []
    dist_err = []
    n_dist = 0
    k_obj = len(spec["objects"])
    bs = 16
    with torch.no_grad():
        for start in range(0, len(rows), bs):
            chunk = rows[start : start + bs]
            batch = _to_torch(collate(chunk, tokens, False, rng), device, bearings=bearings)
            out = heads_out(model, batch, skip_a=skip_a)
            for i, row in enumerate(chunk):
                if policy_supervised(row) and not row.a_only:
                    y_mode.append(row.mode)
                    p_mode.append(int(out["mode"][i].argmax().item()))
                    y_turn.append(row.turn)
                    p_turn.append(int(out["turn"][i].argmax().item()))
                    y_fwd.append(row.forward)
                    p_fwd.append(int(out["forward"][i].argmax().item()))
                    name = MODE_NAMES[row.mode] if 0 <= row.mode < len(MODE_NAMES) else None
                    if name is not None:
                        y_turn_by_mode[name][0].append(row.turn)
                        y_turn_by_mode[name][1].append(int(out["turn"][i].argmax().item()))
                        y_fwd_by_mode[name][0].append(row.forward)
                        y_fwd_by_mode[name][1].append(int(out["forward"][i].argmax().item()))
                if row.has_c and not row.a_only:
                    y_stop.append(int(row.stop))
                    p_stop.append(float(torch.sigmoid(out["stop"][i]).item()))
                if skip_a or "vis_logit" not in out:
                    continue
                vis, vis_ig = _row_vis(row, k_obj)
                vis_logit = out["vis_logit"][i].detach().cpu().numpy()
                vis_prob = 1.0 / (1.0 + np.exp(-vis_logit))
                for k in range(k_obj):
                    if float(vis_ig[k]) >= 0.5:
                        continue
                    vis_y.append(int(vis[k] > 0.5))
                    vis_p.append(float(vis_prob[k]))
                t_i = int(row.target_index)
                if t_i < k_obj and float(vis_ig[t_i]) < 0.5:
                    vis_y_tgt.append(int(vis[t_i] > 0.5))
                    vis_p_tgt.append(float(vis_prob[t_i]))
                mask = row.ignore < 0.5
                if mask.any():
                    present_y.append(row.present[:, mask].reshape(-1))
                    present_p.append(out["present_p"][i].detach().cpu().numpy()[:, mask].reshape(-1))
                    exist = (row.present > 0.5) & mask
                    if exist.any():
                        pred = out["log_dist"][i].detach().cpu().numpy()[exist]
                        dist_err.append(float(np.mean(np.abs(pred - row.log_dist[exist]))))
                        n_dist += int(exist.sum())
    def acc(y, p):
        n = len(y)
        if n == 0:
            return _metric(None, 0)
        return _metric(float(np.mean(np.asarray(y) == np.asarray(p))), n)

    n_stop = len(y_stop)
    n_pos = int(sum(y_stop))
    f1_val = None
    auc_val = None
    stop_best = {"f1": None, "tau": None, "n": n_stop}
    if n_stop == 0 or n_pos == 0:
        if warn_empty_stop and n_stop >= 0:
            print("验证集无停止正样本", flush=True)
        f1_val = None
        auc_val = None
    else:
        pred = [1 if v >= 0.5 else 0 for v in p_stop]
        tp = sum(1 for a, b in zip(y_stop, pred) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_stop, pred) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_stop, pred) if a == 1 and b == 0)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1_val = 2 * prec * rec / max(prec + rec, 1e-9)
        auc_val = _binary_auc(y_stop, p_stop)
        stop_best = _best_stop_f1(y_stop, p_stop)
    mode_acc = {}
    if y_mode:
        y_mode_a = np.asarray(y_mode)
        p_mode_a = np.asarray(p_mode)
        for i, name in enumerate(MODE_NAMES):
            m = y_mode_a == i
            n_c = int(m.sum())
            mode_acc[name] = _metric(None if n_c == 0 else float(np.mean(p_mode_a[m] == i)), n_c)
    else:
        mode_acc = empty_mode
    turn_by_mode = {name: acc(y_turn_by_mode[name][0], y_turn_by_mode[name][1]) for name in MODE_NAMES}
    fwd_by_mode = {name: acc(y_fwd_by_mode[name][0], y_fwd_by_mode[name][1]) for name in MODE_NAMES}
    py = np.concatenate(present_y) if present_y else np.zeros(0)
    pp = np.concatenate(present_p) if present_p else np.zeros(0)
    present_auc = None if py.size == 0 else _binary_auc((py > 0.5).astype(int), pp)
    vis_auc = _binary_auc(vis_y, vis_p) if vis_y else None
    vis_at = {str(t): _pr_fpr(vis_y_tgt, vis_p_tgt, t) for t in (0.5, 0.7, 0.9)}
    vis_fpr_05 = vis_at["0.5"]["fpr"]
    chosen_vis = None
    for t in (0.5, 0.7, 0.9):
        fpr = vis_at[str(t)]["fpr"]
        if fpr is not None and fpr <= 0.03:
            chosen_vis = float(t)
            break
    return {
        "mode_acc": acc(y_mode, p_mode),
        "mode_acc_by_name": mode_acc,
        "turn_acc": acc(y_turn, p_turn),
        "forward_acc": acc(y_fwd, p_fwd),
        "turn_acc_by_mode": turn_by_mode,
        "forward_acc_by_mode": fwd_by_mode,
        "stop_f1": _metric(f1_val, n_pos),
        "stop_auc": _metric(auc_val, n_pos),
        "stop_best_f1": stop_best,
        "present_auc": _metric(present_auc, int(py.size)),
        "vis_auc": _metric(vis_auc, len(vis_y)),
        "vis_target": vis_at,
        "vis_tau_fpr_le_3": chosen_vis,
        "log_dist_mae": _metric(None if not dist_err else float(np.mean(dist_err)), n_dist),
        "pred_mode": {MODE_NAMES[i]: int(p_mode.count(i)) for i in range(len(MODE_NAMES))} if p_mode else None,
        "n": len(rows),
        "n_mode": len(y_mode),
        "n_stop": n_stop,
        "n_stop_pos": n_pos,
        "vis_fpr_0.5": vis_fpr_05,
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
    metrics = offline_metrics(model, subset, tokens, device, spec, cfg, bearings, warn_empty_stop=False)
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
        "vis": torch.zeros(b, k),
        "vis_ignore": torch.zeros(b, k),
        "a_only": torch.zeros(b),
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


TURN_WEIGHT_CAP = 3.0
A_PARAM_PREFIX = ("stem", "present", "dist_conv", "log_dist", "biome", "time", "weather", "brightness", "vis_fc1", "vis_fc2")


def resume_start_epoch(path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu")
    meta = payload.get("meta") if payload.get("meta") is not None else {}
    epoch = meta.get("epoch")
    if epoch is None:
        return 1
    return int(epoch) + 1


def _freeze_a(model, freeze: bool) -> None:
    for name, param in model.named_parameters():
        head = name.split(".", 1)[0]
        if head in A_PARAM_PREFIX:
            param.requires_grad = not freeze


def _brief_val(metrics: dict, extra_a=None) -> dict:
    vis_at = metrics.get("vis_target") if metrics.get("vis_target") is not None else {}
    brief = {
        "vis_auc": metrics.get("vis_auc"),
        "vis_at": vis_at,
        "turn_acc_by_mode": metrics.get("turn_acc_by_mode"),
        "forward_acc_by_mode": metrics.get("forward_acc_by_mode"),
        "stop_best_f1": metrics.get("stop_best_f1"),
    }
    if extra_a is not None:
        brief["a_extra_val"] = {
            "vis_auc": extra_a.get("vis_auc"),
            "vis_at": extra_a.get("vis_target"),
        }
    return brief


BEARING_BUCKETS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 45.0), (45.0, 181.0))


def _bearing_abs(geom) -> float:
    g = np.asarray(geom, np.float32)
    if g.shape[0] < 3 or float(g[0]) < 0.5:
        return float("nan")
    return abs(float(math.degrees(math.atan2(float(g[1]), float(g[2])))))


def _bucket_name(abs_deg: float) -> str:
    if abs_deg != abs_deg:
        return "unseen"
    for lo, hi in BEARING_BUCKETS:
        if lo <= abs_deg < hi:
            if hi >= 180:
                return f">{int(lo)}"
            return f"{int(lo)}-{int(hi)}"
    return f">{int(BEARING_BUCKETS[-1][0])}"


def c_only_diag(model, rows: list[NavSample], tokens, device, spec: dict, cfg: dict, bearings) -> dict:
    import torch
    from collections import Counter

    from lfnav.policy.c2nav_labels import TURN_BINS

    rng = random.Random(0)
    buckets = {}
    for lo, hi in BEARING_BUCKETS:
        key = f">{int(lo)}" if hi >= 180 else f"{int(lo)}-{int(hi)}"
        buckets[key] = {"expert": Counter(), "pred": Counter(), "n": 0}
    n_back = n_back_hit = 0
    skip_a = True
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 16):
            chunk = rows[start : start + 16]
            batch = _to_torch(collate(chunk, tokens, False, rng), device, bearings=bearings)
            out = heads_out(model, batch, skip_a=skip_a)
            for i, row in enumerate(chunk):
                if not policy_supervised(row) or row.a_only:
                    continue
                pred_t = int(out["turn"][i].argmax().item())
                pred_f = int(out["forward"][i].argmax().item())
                if int(row.forward) == 0:
                    n_back += 1
                    if pred_f == 0:
                        n_back_hit += 1
                if row.mode != MODE_NAMES.index("approach"):
                    continue
                src = row.geom_clean if row.geom_clean is not None else row.geom
                key = _bucket_name(_bearing_abs(src))
                if key not in buckets:
                    buckets[key] = {"expert": Counter(), "pred": Counter(), "n": 0}
                buckets[key]["expert"][int(TURN_BINS[row.turn])] += 1
                buckets[key]["pred"][int(TURN_BINS[pred_t])] += 1
                buckets[key]["n"] += 1
    packed = {}
    for key, rec in buckets.items():
        packed[key] = {
            "n": rec["n"],
            "expert": {str(k): int(v) for k, v in sorted(rec["expert"].items())},
            "pred": {str(k): int(v) for k, v in sorted(rec["pred"].items())},
        }
    return {
        "approach_turn_buckets": packed,
        "backward_recall": None if n_back == 0 else n_back_hit / float(n_back),
        "n_backward": n_back,
    }


def _epoch_score(metrics: dict) -> tuple[bool, float, float | None, float | None]:
    vis_tau = metrics.get("vis_tau_fpr_le_3")
    ok = vis_tau is not None
    turn = metrics.get("turn_acc_by_mode") if metrics.get("turn_acc_by_mode") is not None else {}
    approach = turn.get("approach") if isinstance(turn, dict) else None
    approach_acc = 0.0
    if isinstance(approach, dict) and approach.get("value") is not None:
        approach_acc = float(approach["value"])
    stop = metrics.get("stop_best_f1") if metrics.get("stop_best_f1") is not None else {}
    stop_f1 = 0.0 if stop.get("f1") is None else float(stop["f1"])
    stop_tau = stop.get("tau")
    return ok, approach_acc + stop_f1, vis_tau, stop_tau


def train_loop(
    cfg: dict,
    feat_dir: Path,
    out: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    cfg_path: Path | None = None,
    resume=None,
    c_only: bool = False,
    apply_noise: bool = True,
) -> Path:
    import torch

    spec = spec_from_cfg(cfg)
    n_th = torch_threads_of(cfg)
    if n_th > 0:
        torch.set_num_threads(n_th)
    train, val, stats = load_split(cfg, feat_dir, apply_noise=apply_noise)
    if not train:
        raise RuntimeError("no c2nav training samples")
    tokens, _, index = load_memmaps(feat_dir)
    spec["dim"] = int(index.get("dim") if index.get("dim") is not None else spec["dim"])
    spec["grid"] = tuple(index.get("grid") if index.get("grid") is not None else spec["grid"])
    spec["token_pool"] = int(index.get("token_pool") if index.get("token_pool") is not None else spec["token_pool"])
    extra_train, extra_val = load_a_extra(cfg, spec)
    token_arg = tokens
    if extra_train or extra_val:
        stores = [tokens]
        for extra_dir in a_extra_features_of(cfg):
            extra_tok, _, _ = load_memmaps(Path(extra_dir))
            stores.append(extra_tok)
        token_arg = stores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = C2NavNet(spec).to(device)
    nav = _nav_dict(cfg)
    wd_raw = nav.get("weight_decay")
    wd = 0.05 if wd_raw is None else float(wd_raw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    start_epoch = 1
    if resume is not None:
        payload = torch.load(resume, map_location=device)
        model.load_state_dict(payload["model"])
        start_epoch = resume_start_epoch(resume)
        print(f"c2nav resume {resume} start_epoch={start_epoch}", flush=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    pos_w = torch.as_tensor(_presence_pos_weight(train + extra_train, len(spec["objects"])), device=device)
    vis_w = torch.as_tensor(_vis_pos_weight(train + extra_train, len(spec["objects"])), device=device)
    usable = [r for r in train if policy_supervised(r)]
    mode_w, mode_n = class_weights([r.mode for r in usable], len(MODE_NAMES))
    turn_w, turn_n = class_weights([r.turn for r in usable], N_TURN, cap=TURN_WEIGHT_CAP)
    fwd_w, fwd_n = class_weights([r.forward for r in usable], len(FORWARD_BINS), cap=TURN_WEIGHT_CAP)
    mode_w = torch.as_tensor(mode_w, device=device)
    turn_w = torch.as_tensor(turn_w, device=device)
    fwd_w = torch.as_tensor(fwd_w, device=device)
    stop_scale = stop_pos_weight(train)
    stop_pos = torch.tensor(stop_scale, device=device)
    wts_raw = nav.get("loss_weights")
    wts = dict(wts_raw) if wts_raw is not None else {"a": 1, "mode": 1, "wp": 1, "stop": 1}
    if c_only:
        wts = dict(wts)
        wts["a"] = 0.0
        _freeze_a(model, True)
    smooth_raw = nav.get("label_smoothing")
    smooth = 0.1 if smooth_raw is None else float(smooth_raw)
    bearings = patch_bearings(spec["grid"][0], spec["grid"][1], cfg)
    extra_n = a_extra_per_epoch_of(cfg)
    print(
        "c2nav weights "
        + json.dumps({
            "present_pos": pos_w.detach().cpu().tolist(),
            "vis_pos": vis_w.detach().cpu().tolist(),
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
    aug = nav.get("augment")
    if aug is None:
        aug = {}
    use_flip = bool(aug.get("flip"))
    best_pack = None
    skip_a = bool(c_only)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_rows = list(train)
        if extra_train and extra_n > 0:
            take = extra_n if extra_n <= len(extra_train) else len(extra_train)
            epoch_rows.extend(rng.sample(extra_train, k=take) if take < len(extra_train) else list(extra_train))
        order = list(range(len(epoch_rows)))
        rng.shuffle(order)
        running = 0.0
        part_sum = {}
        n_b = 0
        for start in range(0, len(order), batch_size):
            chunk = [epoch_rows[i] for i in order[start : start + batch_size]]
            batch = _to_torch(collate(chunk, token_arg, use_flip, rng), device, bearings=bearings)
            opt.zero_grad()
            out_b = heads_out(model, batch, skip_a=skip_a)
            loss, parts = nav_loss(out_b, batch, wts, pos_w, mode_w, turn_w, fwd_w, stop_pos, smooth, vis_w=vis_w)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_b += 1
            for key, part in parts.items():
                part_sum[key] = part_sum.get(key, 0.0) + float(part)
        sched.step()
        avg_parts = {key: part / max(n_b, 1) for key, part in part_sum.items()}
        if first_parts is None:
            first_parts = avg_parts
            print(f"c2nav loss epoch_{epoch} {json.dumps(first_parts)}", flush=True)
        last_parts = avg_parts
        eval_rows = val if len(val) != 0 else train[: min(64, len(train))]
        metrics = offline_metrics(
            model=model,
            rows=eval_rows,
            tokens=token_arg,
            device=device,
            spec=spec,
            cfg=cfg,
            bearings=bearings,
            warn_empty_stop=len(val) != 0,
            skip_a=skip_a,
        )
        extra_metrics = None
        if extra_val:
            extra_metrics = offline_metrics(
                model=model,
                rows=extra_val,
                tokens=token_arg,
                device=device,
                spec=spec,
                cfg=cfg,
                bearings=bearings,
                warn_empty_stop=False,
                skip_a=skip_a,
            )
        metrics["train_loss"] = running / max(n_b, 1)
        metrics["loss_parts"] = avg_parts
        brief = _brief_val(metrics, extra_metrics)
        print(f"c2nav val epoch {epoch} {json.dumps(brief, default=str)}", flush=True)
        ok, score, vis_tau, stop_tau = _epoch_score(metrics)
        extra_meta = {
            "filter": stats,
            "chosen_vis_tau": vis_tau,
            "chosen_stop_tau": stop_tau,
            "chosen_tau": stop_tau,
        }
        ckpt = out / f"epoch_{epoch:02d}.pt"
        save_checkpoint(ckpt, model, spec, epoch, metrics, extra=extra_meta)
        save_checkpoint(last, model, spec, epoch, metrics, extra=extra_meta)
        if ok and (best_pack is None or score > best_pack[0]):
            best_pack = (score, epoch, vis_tau, stop_tau, metrics)
            save_checkpoint(out / "best.pt", model, spec, epoch, metrics, extra={
                **extra_meta,
                "best_score": score,
            })
        print(f"epoch {epoch} loss={metrics['train_loss']:.4f} metrics={json.dumps(metrics, default=str)}", flush=True)
    if last_parts is not None:
        print(f"c2nav loss last_epoch {json.dumps(last_parts)}", flush=True)
    if best_pack is None:
        print("c2nav best skipped: no epoch with frame vis FPR <= 0.03", flush=True)
    else:
        print(f"c2nav best epoch={best_pack[1]} score={best_pack[0]:.4f} vis_tau={best_pack[2]} stop_tau={best_pack[3]}", flush=True)
    if c_only:
        diag_rows = val if val else train
        diag = c_only_diag(model, diag_rows, token_arg, device, spec, cfg, bearings)
        print("c2nav c_only_diag " + json.dumps(diag, default=str), flush=True)
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
    parser.add_argument("--resume", default=None)
    parser.add_argument("--c-only", action="store_true")
    parser.add_argument("--no-noise", action="store_true")
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
    nav = _nav_dict(cfg)
    feat_ver = nav.get("features_version")
    if feat_ver is None:
        feat_ver = "joint_v1"
    feat = Path(args.features) if args.features is not None else Path(os.environ.get("SCR") if os.environ.get("SCR") is not None else f"/scratch/{os.environ.get('USER', 'user')}") / "lfnav" / "features" / str(feat_ver)
    out_name = ckpt_dir_of(cfg)
    out = Path(args.out) if args.out is not None else Path(os.environ.get("SCR") if os.environ.get("SCR") is not None else f"/scratch/{os.environ.get('USER', 'user')}") / "lfnav" / "ckpt" / out_name
    ep_raw = nav.get("epochs")
    epochs = int(args.epochs if args.epochs is not None else (15 if ep_raw is None else ep_raw))
    batch_raw = nav.get("batch")
    batch = int(args.batch if args.batch is not None else (16 if batch_raw is None else batch_raw))
    lr_raw = nav.get("lr")
    lr = float(args.lr if args.lr is not None else (3e-4 if lr_raw is None else lr_raw))
    train_loop(
        cfg, feat, out, epochs, batch, lr, int(args.seed), cfg_path=cfg_path,
        resume=args.resume, c_only=bool(args.c_only), apply_noise=not bool(args.no_noise),
    )


if __name__ == "__main__":
    main()
