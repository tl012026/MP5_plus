"""Val-set stop labels vs dir-visible / dist_bin=near / conditions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lfnav.config import load_config
from lfnav.data.demo_format import read_episode
from lfnav.policy.c2_data import demo_roots, iter_success_episodes, select_training_episodes
from lfnav.policy.c2nav import DIR_LCR_NAMES, conditions_satisfied, dist_bin_from_num
from lfnav.policy.c2nav_labels import labels_from_step
from lfnav.policy.train_c2nav import VAL_SEEDS_DEFAULT, _nav_dict


def _val_folders(cfg):
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
    val = []
    for folder in episodes:
        rec = read_episode(folder)["record"]
        seed = int(rec.get("seed") if rec.get("seed") is not None else Path(folder).name)
        if seed in val_seeds:
            val.append(Path(folder))
    return val


def rule_pred(step: dict, cfg: dict) -> bool:
    match = step.get("match") if isinstance(step.get("match"), dict) else {}
    dir_visible = match.get("target_direction") in DIR_LCR_NAMES
    dist_near = dist_bin_from_num(step.get("target_distance_num"), cfg) == "near"
    return bool(dir_visible and dist_near and conditions_satisfied(match))


def main():
    cfg = load_config(_ROOT / "configs/lfnav/c2nav_easy.yaml")
    n = tp = fp = fn = n_pos = n_pred = 0
    for folder in _val_folders(cfg):
        packed = read_episode(folder)
        for step in packed["steps"]:
            gold = labels_from_step(step)
            if gold is None:
                continue
            y = int(gold.get("stop") or 0) == 1
            pred = rule_pred(step, cfg)
            n += 1
            if y:
                n_pos += 1
            if pred:
                n_pred += 1
            if pred and y:
                tp += 1
            elif pred and not y:
                fp += 1
            elif y and not pred:
                fn += 1
    prec = None if (tp + fp) == 0 else tp / float(tp + fp)
    rec = None if (tp + fn) == 0 else tp / float(tp + fn)
    f1 = None
    if prec is not None and rec is not None and (prec + rec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    print("STOP_RULE " + json.dumps({
        "n": n,
        "n_pos": n_pos,
        "n_pred": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }), flush=True)


if __name__ == "__main__":
    main()
