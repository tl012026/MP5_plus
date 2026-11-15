"""Check C obs dir_lcr / dist_bin / sector8 vs expert, and explore soft support."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lfnav.config import load_config
from lfnav.data.demo_format import read_episode
from lfnav.policy.c2_data import demo_roots, iter_success_episodes, load_task, select_training_episodes
from lfnav.policy.c2nav import (
    attach_lcr_dist,
    encode_s,
    gt_target_obs,
    patch_bearings,
    patch_targets,
    spec_from_cfg,
)
from lfnav.policy.c2nav_labels import MODE_NAMES, explore_turn_soft, labels_from_step
from lfnav.policy.train_c2nav import VAL_SEEDS_DEFAULT, _nav_dict


def _folders(cfg):
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
    train = []
    for folder in episodes:
        rec = read_episode(folder)["record"]
        seed = int(rec.get("seed") if rec.get("seed") is not None else Path(folder).name)
        if seed not in val_seeds:
            train.append(Path(folder))
    return train


def main():
    cfg = load_config(_ROOT / "configs/lfnav/c2nav_easy.yaml")
    spec = spec_from_cfg(cfg)
    objects = list(spec["objects"])
    grid = tuple(spec["grid"])
    pool = int(spec["token_pool"])
    bearings = patch_bearings(int(grid[0]), int(grid[1]), cfg)
    n_dir = n_dir_ok = 0
    n_dist = n_dist_ok = 0
    n_sec = n_sec_ok = 0
    n_ex = n_ex_ok = 0
    for folder in _folders(cfg):
        packed = read_episode(folder)
        rec = packed["record"]
        task = load_task(rec, folder, cfg)
        target = str(task.get("target") if task.get("target") is not None else "pig")
        t_i = objects.index(target) if target in objects else 0
        for step in packed["steps"]:
            gold = labels_from_step(step)
            match = step.get("match") if isinstance(step.get("match"), dict) else {}
            mem = step.get("memory") if isinstance(step.get("memory"), dict) else {}
            scene = step.get("scene") if step.get("scene") is not None else {}
            present, _log_d, ignore = patch_targets(scene, objects, pool, grid)
            patches = scene.get("patches") if scene.get("patches") is not None else {}
            raw = np.asarray(patches.get(target) if isinstance(patches, dict) else present[t_i], np.float32)
            if raw.ndim != 2:
                raw = present[t_i]
            bel = (scene.get("objects") or {}).get(target) if isinstance(scene.get("objects"), dict) else None
            dist_mass = (bel or {}).get("distance") if isinstance(bel, dict) else None
            obs = attach_lcr_dist(
                gt_target_obs(present, ignore, t_i, bearings, step.get("target_distance_num")),
                raw, scene.get("ignore") if scene.get("ignore") is not None else ignore,
                step.get("target_distance_num"), cfg, dist_mass=dist_mass,
            )
            branch = str(step.get("expert_branch") or "")
            if branch == "approach":
                expert_d = match.get("target_direction")
                if expert_d in ("left", "center", "right"):
                    n_dir += 1
                    if obs.dir_lcr == expert_d:
                        n_dir_ok += 1
                expert_dist = match.get("target_distance")
                if expert_dist in ("near", "mid", "far"):
                    n_dist += 1
                    if obs.dist_bin == expert_dist:
                        n_dist_ok += 1
            sec = mem.get("sector8")
            if sec is not None:
                n_sec += 1
                vec = encode_s(np.zeros(18, np.float32), sector8=sec)
                counts = [float(x) for x in list(sec)[:8]]
                if len(counts) < 8:
                    counts = counts + [0.0] * (8 - len(counts))
                total = sum(counts) or 1.0
                gold_s = np.asarray([c / total for c in counts], np.float32)
                if np.allclose(vec[7:15], gold_s, atol=1e-6):
                    n_sec_ok += 1
            if gold is not None and int(gold.get("mode")) == MODE_NAMES.index("explore"):
                n_ex += 1
                soft = explore_turn_soft(sec, extra_turn=int(gold["turn_index"]))
                ti = int(gold["turn_index"])
                if 0 <= ti < len(soft) and float(soft[ti]) > 0:
                    n_ex_ok += 1
    payload = {
        "a_dir_lcr": {
            "n": n_dir,
            "agree": None if n_dir == 0 else n_dir_ok / float(n_dir),
            "pass": n_dir > 0 and n_dir_ok / float(n_dir) >= 0.98,
        },
        "a_dist_bin": {
            "n": n_dist,
            "agree": None if n_dist == 0 else n_dist_ok / float(n_dist),
            "pass": n_dist > 0 and n_dist_ok / float(n_dist) >= 0.98,
        },
        "a_sector8": {
            "n": n_sec,
            "agree": None if n_sec == 0 else n_sec_ok / float(n_sec),
            "pass": n_sec > 0 and n_sec_ok / float(n_sec) >= 0.98,
        },
        "b_explore_support": {
            "n": n_ex,
            "agree": None if n_ex == 0 else n_ex_ok / float(n_ex),
            "pass": n_ex > 0 and n_ex_ok / float(n_ex) >= 0.98,
        },
    }
    print("ALIGN " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
