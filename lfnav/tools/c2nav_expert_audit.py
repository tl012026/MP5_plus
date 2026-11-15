"""Audit C1 expert info vs C2Nav C observations. Analysis only; --only T0..T5."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lfnav.aggregate import column_direction
from lfnav.config import load_config
from lfnav.data.demo_format import read_episode
from lfnav.perception.extract_features import episode_feature_name
from lfnav.policy.c1 import RuleBased, _target_visible, sector8_counts
from lfnav.policy.c2_data import demo_roots, iter_success_episodes, load_task, select_training_episodes
from lfnav.policy.c2nav import gt_target_obs, horizontal_fov_deg, patch_bearings, patch_targets, spec_from_cfg
from lfnav.policy.c2nav_labels import MODE_NAMES, compose_waypoint, labels_from_step
from lfnav.policy.expert import wide_target
from lfnav.policy.train_c2nav import VAL_SEEDS_DEFAULT, _nav_dict, load_split
from lfnav.matcher import Matcher

N_TAKE = 5000
TRAIN_SEED = 0


def _print(tag: str, obj) -> None:
    if isinstance(obj, str):
        print(f"{tag} {obj}", flush=True)
        return
    print(f"{tag} {json.dumps(obj, default=str)}", flush=True)


def _loc(fn) -> str:
    src = inspect.getsourcefile(fn) or "?"
    try:
        lines, start = inspect.getsourcelines(fn)
    except OSError:
        return f"{src} {fn.__qualname__}"
    end = start + len(lines) - 1
    rel = src
    if str(_ROOT) in src:
        rel = str(Path(src).resolve().relative_to(_ROOT))
    return f"{rel} {fn.__qualname__} L{start}-{end}"


def _cfg(args):
    path = Path(args.config).resolve() if args.config else _ROOT / "configs/lfnav/c2nav_easy.yaml"
    return load_config(path), path


def _feat(args, cfg) -> Path:
    if args.features:
        return Path(args.features)
    nav = _nav_dict(cfg)
    ver = nav.get("features_version")
    if ver is None:
        ver = "nav_v3_easy"
    return Path("/scratch/yl13095/lfnav/features") / str(ver)


def _hfov_bounds(cfg: dict) -> tuple[float, float, float]:
    hfov = float(horizontal_fov_deg(cfg))
    half = math.radians(hfov / 2.0)

    def yaw_at_u(u: float) -> float:
        return math.degrees(math.atan(2.0 * (float(u) - 0.5) * math.tan(half)))

    return hfov, yaw_at_u(1.0 / 3.0), yaw_at_u(2.0 / 3.0)


def bearing_to_lcr(yaw, cfg: dict) -> str:
    _hfov, left_b, right_b = _hfov_bounds(cfg)
    y = float(yaw)
    if y < left_b:
        return "left"
    if y < right_b:
        return "center"
    return "right"


def expert_approach_actions(match: dict, mid_cap: int) -> list[str]:
    dummy = RuleBased({"policy": {"c1_opening_scan": False, "c1_mid_cap": int(mid_cap)}})
    if match.get("status") == "found":
        return dummy._approach(match.get("target_direction"), match.get("target_distance"))
    if match.get("missing"):
        return ["turn_left_45", "forward"]
    return dummy._approach(match.get("target_direction"), match.get("target_distance"))


def _connected_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    n = 0
    for i in range(h):
        for j in range(w):
            if (not mask[i, j]) or seen[i, j]:
                continue
            n += 1
            stack = [(i, j)]
            while stack:
                x, y = stack.pop()
                if x < 0 or y < 0 or x >= h or y >= w or seen[x, y] or (not mask[x, y]):
                    continue
                seen[x, y] = True
                stack.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    return n


def _raw_patch(scene: dict, target: str) -> tuple[np.ndarray, np.ndarray]:
    patches = scene.get("patches") if scene.get("patches") is not None else {}
    grid = np.asarray(patches.get(target) if isinstance(patches, dict) else [], np.float32)
    if grid.ndim != 2:
        grid = np.zeros((16, 26), np.float32)
    ig = scene.get("ignore")
    ignore = np.zeros(grid.shape, np.float32) if ig is None else np.asarray(ig, np.float32)
    if ignore.ndim != 2 or ignore.shape != grid.shape:
        ignore = np.zeros(grid.shape, np.float32)
    return grid, ignore


def _our_obs(scene: dict, spec: dict, cfg: dict, target: str, dist_num):
    objects = list(spec["objects"])
    t_i = objects.index(target) if target in objects else 0
    grid = tuple(spec["grid"])
    pool = int(spec["token_pool"])
    present, _log_d, ignore = patch_targets(scene, objects, pool, grid)
    bearings = patch_bearings(int(grid[0]), int(grid[1]), cfg)
    return gt_target_obs(present, ignore, t_i, bearings, dist_num), present[t_i], ignore


def _iter_train_folders(cfg: dict):
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
    episodes, _stats = select_training_episodes(folders, cfg)
    val_raw = nav.get("val_seeds")
    val_seeds = {int(s) for s in (VAL_SEEDS_DEFAULT if val_raw is None else val_raw)}
    train, val = [], []
    for folder in episodes:
        rec = read_episode(folder)["record"]
        seed = int(rec.get("seed") if rec.get("seed") is not None else Path(folder).name)
        (val if seed in val_seeds else train).append(Path(folder))
    return train, val


def run_t0(args) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(4)
    cfg, _ = _cfg(args)
    feat = _feat(args, cfg)
    train, _val, _stats = load_split(cfg, feat, apply_noise=True)
    g = torch.Generator()
    g.manual_seed(int(TRAIN_SEED))
    loader = DataLoader(
        TensorDataset(torch.arange(len(train))),
        batch_size=16,
        shuffle=True,
        generator=g,
    )
    src = Counter()
    n_gt_inv = 0
    n_fp = 0
    n = 0
    for (idx,) in loader:
        for i in idx.tolist():
            row = train[int(i)]
            src[str(row.obs_source)] += 1
            gt_vis = False if row.geom_clean is None else float(row.geom_clean[0]) >= 0.5
            if not gt_vis:
                n_gt_inv += 1
                if str(row.obs_source) == "gt_noisy_fp":
                    n_fp += 1
            n += 1
            if n >= N_TAKE:
                break
        if n >= N_TAKE:
            break
    rate = None if n_gt_inv == 0 else n_fp / float(n_gt_inv)
    _print("T0", {
        "n": n,
        "seed": TRAIN_SEED,
        "batch": 16,
        "obs_source": dict(src),
        "n_gt_invisible": n_gt_inv,
        "n_gt_noisy_fp": n_fp,
        "fp_rate_among_gt_invisible": rate,
        "expected": "0.05-0.15",
        "pass": False if rate is None else (0.05 <= rate <= 0.15),
    })


def run_t1(args) -> None:
    cfg, _ = _cfg(args)
    hfov, left_b, right_b = _hfov_bounds(cfg)
    print("T1 code", flush=True)
    print("T1 a_see", _loc(_target_visible), flush=True)
    print("T1 a_see_matcher", _loc(Matcher.match), flush=True)
    print("T1 b_wide_nearest", _loc(wide_target), flush=True)
    print("T1 c_column_thirds", _loc(column_direction), flush=True)
    print("T1 c_approach", _loc(RuleBased._approach), flush=True)
    print("T1 d_mid_cap_act", _loc(RuleBased.act), flush=True)
    print("T1 e_reacquire", _loc(RuleBased.act), flush=True)
    print("T1 f_sectors", _loc(sector8_counts), flush=True)
    print("T1 g_stop", _loc(Matcher.match), flush=True)
    print("T1 a. seeing the target: RuleBased uses _target_visible: match.target_direction is not None, or status==found, or satisfied.target. Matcher sets target_ok from scene.object(target).present >= tau_obj (0.5), which is max non-ignore 16x26 patch of that class.", flush=True)
    print("T1 b. multiple same-class objects: no instance id. aggregate_object sums every positive patch of the class; direction is argmax of that mass. wide_target and privileged_ray_distance pick the nearest ray.", flush=True)
    print(
        f"T1 c. left/center/right: column_direction splits the image into thirds (u=(col+0.5)/n_w; left u<1/3, center u<2/3, right else). At image_hw=224x364 fov=70 this is yaw {left_b:.2f}° / {right_b:.2f}° (hfov={hfov:.2f}°). Matcher.target_direction is argmax of those thirds. C1._approach maps left->turn_left_15, right->turn_right_15, center->no turn. The left-45 vs right-15 asymmetry is not this split: when the target is visible but match.missing is nonempty, act() always emits turn_left_45+forward (c1.py) and never a right-45. Right 15° is only the found-and-right approach.",
        flush=True,
    )
    print("T1 d. mid/near forward: if distance==mid, append forward*c1_mid_cap (default 2) after the 15° turn. If not mid and there is a turn (near left/right), return only the turn. If not mid and no turn (near/far center), return one forward.", flush=True)
    print("T1 e. reacquire: not visible, last_seen within lost_horizon=20 env steps, last_seen.distance in {{near,mid}}: first action is backward, then 24x turn_right_15. Opening scan is 24x turn_right_15. recover_scan default true.", flush=True)
    print("T1 f. explore sectors: sector8_counts, 45° bins, index 0 is forward (ahead, side) via atan2(side, ahead), remaining clockwise. Ties: all bins with the minimum visit count; rng.choice(tied). Stored in steps as memory.sector8.", flush=True)
    print("T1 g. stop: Matcher stop_mode default sr_b / stop_rule nearest: found (no missing after optional smooth) AND aligned_distance <= judge.radius (3). Expert branch 'stop' when match.stop. labels_from_step stop=1 iff expert_branch==stop.", flush=True)
    print("T1 sign: column_bearing and wide yaw_offset are lfnav degrees, + is right. match left is the left image third (negative yaw).", flush=True)
    train, val = _iter_train_folders(cfg)
    keys = Counter()
    examples = defaultdict(list)
    n_steps = 0
    for folder in (train + val)[:40]:
        packed = read_episode(folder)
        for step in packed["steps"]:
            match = step.get("match") if isinstance(step.get("match"), dict) else {}
            n_steps += 1
            for k, v in match.items():
                keys[k] += 1
                if len(examples[k]) < 3:
                    examples[k].append({"episode": str(folder), "decision": step.get("decision"), "value": v})
    _print("T1 match_keys", {"n_steps_scanned": n_steps, "keys": dict(keys)})
    for k in sorted(examples):
        _print(f"T1 match.{k}", examples[k])


def _confusion(rows: list[tuple[str, str]]) -> dict:
    labels = ("left", "center", "right")
    table = {a: {b: 0 for b in labels} for a in labels}
    n = 0
    agree = 0
    for expert, ours in rows:
        if expert not in table or ours not in table[expert]:
            continue
        table[expert][ours] += 1
        n += 1
        if expert == ours:
            agree += 1
    return {"n": n, "agree": None if n == 0 else agree / float(n), "matrix_expert_row_our_col": table}


def run_t2(args) -> None:
    cfg, _ = _cfg(args)
    spec = spec_from_cfg(cfg)
    pol = cfg.get("policy") if cfg.get("policy") is not None else {}
    mid_raw = pol.get("c1_mid_cap")
    mid_cap = 2 if mid_raw is None else int(mid_raw)
    hfov, left_b, right_b = _hfov_bounds(cfg)
    train, val = _iter_train_folders(cfg)
    n_turn = n_turn_ok = n_fwd = n_fwd_ok = 0
    pairs = []
    n_approach = 0
    for folder in train:
        packed = read_episode(folder)
        rec = packed["record"]
        task = load_task(rec, folder, cfg)
        target = str(task.get("target") if task.get("target") is not None else "pig")
        for step in packed["steps"]:
            if str(step.get("expert_branch") or "") != "approach":
                continue
            n_approach += 1
            match = step.get("match") if isinstance(step.get("match"), dict) else {}
            gold = labels_from_step(step)
            if gold is None:
                continue
            pred_act = expert_approach_actions(match, mid_cap)
            pred = compose_waypoint(pred_act)
            n_turn += 1
            n_fwd += 1
            if int(pred["turn_index"]) == int(gold["turn_index"]):
                n_turn_ok += 1
            if int(pred["forward_index"]) == int(gold["forward_index"]):
                n_fwd_ok += 1
            scene = step.get("scene") if step.get("scene") is not None else {}
            obs, _p, _ig = _our_obs(scene, spec, cfg, target, step.get("target_distance_num"))
            expert_d = match.get("target_direction")
            if expert_d in ("left", "center", "right") and obs.visible and obs.bearing_deg is not None:
                pairs.append((str(expert_d), bearing_to_lcr(obs.bearing_deg, cfg)))
    _print("T2", {
        "split": "train",
        "n_approach": n_approach,
        "turn_agree": None if n_turn == 0 else n_turn_ok / float(n_turn),
        "forward_agree": None if n_fwd == 0 else n_fwd_ok / float(n_fwd),
        "n": n_turn,
        "mid_cap": mid_cap,
        "our_lcr_cut_deg": {"left_lt": left_b, "right_ge": right_b, "hfov": hfov},
        "sign": "our bearing + is right; match left is negative yaw / left image third",
        "confusion_match_direction_vs_our_bearing_lcr": _confusion(pairs),
    })


def run_t3(args) -> None:
    cfg, _ = _cfg(args)
    spec = spec_from_cfg(cfg)
    train, _val = _iter_train_folders(cfg)
    one, many = [], []
    n_one = n_many = 0
    for folder in train:
        packed = read_episode(folder)
        rec = packed["record"]
        task = load_task(rec, folder, cfg)
        target = str(task.get("target") if task.get("target") is not None else "pig")
        for step in packed["steps"]:
            if str(step.get("expert_branch") or "") != "approach":
                continue
            scene = step.get("scene") if step.get("scene") is not None else {}
            raw, _ig = _raw_patch(scene, target)
            ncc = _connected_count(raw > 0.5)
            match = step.get("match") if isinstance(step.get("match"), dict) else {}
            expert_d = match.get("target_direction")
            obs, _p, _ig2 = _our_obs(scene, spec, cfg, target, step.get("target_distance_num"))
            if expert_d not in ("left", "center", "right") or (not obs.visible) or obs.bearing_deg is None:
                continue
            pair = (str(expert_d), bearing_to_lcr(obs.bearing_deg, cfg))
            if ncc <= 1:
                one.append(pair)
                n_one += 1
            else:
                many.append(pair)
                n_many += 1
    _print("T3", {
        "n_approach_with_both_dirs_1cc": n_one,
        "n_approach_with_both_dirs_multicc": n_many,
        "agree_1_region": _confusion(one),
        "agree_multi_region": _confusion(many),
        "connectivity": "4-connected on 16x26 present>0.5, target class, ignore not applied",
    })


def _t4_reason(raw: np.ndarray, ignore: np.ndarray, pooled_present: np.ndarray, pooled_ignore: np.ndarray) -> str:
    pos16 = raw > 0.5
    if not np.any(pos16):
        return "other_no_16x26_positive"
    non_ig = pos16 & (ignore < 0.5)
    if not np.any(non_ig):
        return "all_in_ignore"
    pooled_hit = (pooled_present > 0.5) & (pooled_ignore < 0.5)
    if not np.any(pooled_hit):
        return "lost_after_pool_8x13"
    return "other"


def run_t4(args) -> None:
    cfg, _ = _cfg(args)
    spec = spec_from_cfg(cfg)
    train, val = _iter_train_folders(cfg)

    def scan(folders, split: str) -> None:
        approach_inv = Counter()
        stop_inv = Counter()
        samples = {"approach_inv": defaultdict(list), "stop_inv": defaultdict(list)}
        n_explore = 0
        n_tie = 0
        n_app_inv = n_stop_inv = 0
        for folder in folders:
            packed = read_episode(folder)
            rec = packed["record"]
            task = load_task(rec, folder, cfg)
            target = str(task.get("target") if task.get("target") is not None else "pig")
            for step in packed["steps"]:
                branch = str(step.get("expert_branch") or "")
                scene = step.get("scene") if step.get("scene") is not None else {}
                obs, pooled_p, pooled_ig = _our_obs(scene, spec, cfg, target, step.get("target_distance_num"))
                raw, ign = _raw_patch(scene, target)
                if branch == "explore":
                    n_explore += 1
                    mem = step.get("memory") if isinstance(step.get("memory"), dict) else {}
                    counts = mem.get("sector8")
                    if counts is None:
                        counts = sector8_counts([])
                    counts = [float(x) for x in counts]
                    if counts:
                        lo = min(counts)
                        tied = [i for i, n in enumerate(counts) if n == lo]
                        if len(tied) > 1:
                            n_tie += 1
                if obs.visible:
                    continue
                reason = _t4_reason(raw, ign, pooled_p, pooled_ig)
                rec_s = {
                    "episode": f"{task.get('id')}/{rec.get('difficulty')}/{rec.get('seed')}",
                    "decision": step.get("decision"),
                    "branch": branch,
                    "match_direction": (step.get("match") or {}).get("target_direction"),
                    "match_status": (step.get("match") or {}).get("status"),
                    "n_pos16": int((raw > 0.5).sum()),
                    "n_pos16_nonignore": int(((raw > 0.5) & (ign < 0.5)).sum()),
                    "n_pos8": int(((pooled_p > 0.5) & (pooled_ig < 0.5)).sum()),
                    "reason": reason,
                }
                if branch == "approach":
                    n_app_inv += 1
                    approach_inv[reason] += 1
                    if len(samples["approach_inv"][reason]) < 2:
                        samples["approach_inv"][reason].append(rec_s)
                gold = labels_from_step(step)
                if gold is not None and int(gold.get("stop") or 0) == 1:
                    n_stop_inv += 1
                    stop_inv[reason] += 1
                    if len(samples["stop_inv"][reason]) < 2:
                        samples["stop_inv"][reason].append(rec_s)
        _print(f"T4 {split}", {
            "n_approach_our_invisible": n_app_inv,
            "approach_reasons": dict(approach_inv),
            "n_stop1_our_invisible": n_stop_inv,
            "stop_reasons": dict(stop_inv),
            "explore_n": n_explore,
            "explore_min_sector_tie_frac": None if n_explore == 0 else n_tie / float(n_explore),
            "samples": {k: dict(v) for k, v in samples.items()},
        })

    scan(train, "train")
    scan(val, "val")


def run_t5(args) -> None:
    cfg, _ = _cfg(args)
    spec = spec_from_cfg(cfg)
    train, val = _iter_train_folders(cfg)
    n_train = n_val = 0
    for split, folders in (("train", train), ("val", val)):
        n = 0
        for folder in folders:
            packed = read_episode(folder)
            rec = packed["record"]
            task = load_task(rec, folder, cfg)
            target = str(task.get("target") if task.get("target") is not None else "pig")
            for step in packed["steps"]:
                gold = labels_from_step(step)
                if gold is None or int(gold.get("stop") or 0) != 1:
                    continue
                scene = step.get("scene") if step.get("scene") is not None else {}
                obs, _p, _ig = _our_obs(scene, spec, cfg, target, step.get("target_distance_num"))
                if not obs.visible:
                    n += 1
        if split == "train":
            n_train = n
        else:
            n_val = n
    _print("T5", {
        "n_stop1_our_invisible_train": n_train,
        "n_stop1_our_invisible_val": n_val,
        "loss": "stop loss masked when stop==1 and geom_clean visible is false",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", required=True, choices=["T0", "T1", "T2", "T3", "T4", "T5"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--features", default=None)
    args = parser.parse_args()
    {"T0": run_t0, "T1": run_t1, "T2": run_t2, "T3": run_t3, "T4": run_t4, "T5": run_t5}[args.only](args)


if __name__ == "__main__":
    main()
