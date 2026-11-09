"""Sum per-patch object labels into present / direction / distance. No torch.

Also summarizes baseline evaluation trees (policy/difficulty/task JSON).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from lfnav.eval.metrics import mcnemar, wilson_interval
from lfnav.types import ObjectBelief, SceneState

DIRS = ("left", "center", "right")
DISTS = ("near", "mid", "far")


def _blank_dist():
    return {k: 0.0 for k in DISTS}


def _blank_dir():
    return {k: 0.0 for k in DIRS}


def column_direction(col: int, n_w: int) -> str:
    """Third of the image that contains the patch center. A single column is center."""
    if n_w <= 0:
        raise ValueError("n_w must be positive")
    u = (col + 0.5) / n_w
    if u < 1.0 / 3.0:
        return "left"
    if u < 2.0 / 3.0:
        return "center"
    return "right"


def distance_bin(blocks: float, near: float = 3.0, mid: float = 10.0) -> str:
    if blocks <= near:
        return "near"
    if blocks <= mid:
        return "mid"
    return "far"


def aggregate_object(present_hw, dist_hw=None, ignore_hw=None) -> ObjectBelief:
    """present_hw[row][col] in [0, 1]. dist_hw[row][col] is near/mid/far or a {bin: prob} map.

    present is the max over patches. Direction mass is the sum of present in each horizontal
    third, normalized. Distance mass is present-weighted over patches. Ignored HUD/hand
    patches are skipped.
    """
    if not present_hw or not present_hw[0]:
        return ObjectBelief()
    n_h = len(present_hw)
    n_w = len(present_hw[0])
    present = 0.0
    dir_mass = _blank_dir()
    dist_mass = _blank_dist()
    for i in range(n_h):
        for j in range(n_w):
            if ignore_hw and i < len(ignore_hw) and j < len(ignore_hw[i]) and ignore_hw[i][j]:
                continue
            p = float(present_hw[i][j])
            present = max(present, p)
            dir_mass[column_direction(j, n_w)] += p
            if dist_hw is None or p <= 0:
                continue
            cell = dist_hw[i][j]
            if isinstance(cell, str):
                dist_mass[cell] = dist_mass.get(cell, 0.0) + p
            else:
                for name, val in cell.items():
                    dist_mass[name] = dist_mass.get(name, 0.0) + p * float(val)
    dir_sum = sum(dir_mass.values())
    dist_sum = sum(dist_mass.values())
    if dir_sum > 0:
        dir_mass = {k: v / dir_sum for k, v in dir_mass.items()}
    if dist_sum > 0:
        dist_mass = {k: v / dist_sum for k, v in dist_mass.items()}
    return ObjectBelief(present=present, direction=dir_mass, distance=dist_mass)


def scene_from_patches(per_object, globals_probs=None) -> SceneState:
    """per_object: name -> (present_hw, dist_hw). globals_probs holds biome/time/weather/brightness."""
    objects = {name: aggregate_object(pres, dist) for name, (pres, dist) in per_object.items()}
    g = globals_probs or {}
    return SceneState(
        objects=objects,
        biome=dict(g.get("biome") or {}),
        time=dict(g.get("time") or {}),
        weather=dict(g.get("weather") or {}),
        brightness=dict(g.get("brightness") or {}),
    )


def argmax_name(probs: dict) -> str | None:
    if not probs:
        return None
    return max(probs.items(), key=lambda kv: kv[1])[0]


def _numeric_distance(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.startswith(">"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_baseline_rows(root: Path) -> list[dict]:
    rows = []
    root = Path(root)
    for path in sorted(root.glob("*/*/*/*.json")):
        if path.name in ("run_meta.json",):
            continue
        rec = json.loads(path.read_text())
        if "seed" not in rec:
            continue
        rec = dict(rec)
        rec.setdefault("policy_name", path.parts[-4])
        rec.setdefault("difficulty", path.parts[-3])
        rec.setdefault("task_name", path.parts[-2])
        rows.append(rec)
    if not rows:
        for path in sorted(root.rglob("*.json")):
            if path.name in ("run_meta.json",):
                continue
            rec = json.loads(path.read_text())
            if "seed" not in rec:
                continue
            rows.append(rec)
    return rows


def _rate(rows: list[dict], key: str) -> dict:
    vals = [bool(r.get(key)) for r in rows]
    return wilson_interval(sum(vals), len(vals))


def _median_steps(rows: list[dict]):
    steps = [int(r["steps"]) for r in rows if r.get("steps") is not None]
    if not steps:
        return None
    return statistics.median(steps)


def _spl_stats(rows: list[dict]) -> dict:
    values = [r.get("spl") for r in rows]
    known = [float(v) for v in values if v is not None]
    beyond = sum(1 for r in rows if r.get("spl") is None or r.get("initial_distance_beyond_range"))
    return {
        "n_known": len(known),
        "n_beyond_range": beyond,
        "mean": (sum(known) / len(known)) if known else None,
        "median": statistics.median(known) if known else None,
    }


def _distance_stats(rows: list[dict]) -> dict:
    nums = []
    beyond = 0
    for row in rows:
        num = _numeric_distance(row.get("initial_target_distance"))
        if num is None:
            beyond += 1
        else:
            nums.append(num)
    out = {
        "n_numeric": len(nums),
        "n_beyond_range": beyond,
        "min": min(nums) if nums else None,
        "max": max(nums) if nums else None,
        "median": statistics.median(nums) if nums else None,
    }
    return out


def summarize_baseline(root: Path) -> dict:
    rows = load_baseline_rows(root)
    groups = {}
    for row in rows:
        key = (row.get("task_name"), row.get("difficulty"), row.get("policy_name"))
        groups.setdefault(key, []).append(row)
    table = []
    for (task, difficulty, policy), chunk in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2]))):
        counted = [row for row in chunk if not row.get("spawn_failed")]
        table.append({
            "task": task,
            "difficulty": difficulty,
            "policy": policy,
            "n": len(chunk),
            "n_scored": len(counted),
            "sr_b": _rate(counted, "sr_b"),
            "sr_a": _rate(counted, "sr_a"),
            "false_stop": _rate(counted, "false_stop"),
            "timeout": _rate(counted, "timeout"),
            "steps_median": _median_steps(counted),
            "spl": _spl_stats(counted),
            "initial_target_distance": _distance_stats(counted),
            "difficulty_unmet": _rate(counted, "difficulty_unmet"),
            "spawn_failed": _rate(chunk, "spawn_failed"),
        })
    paired = []
    by_cell = {}
    failed = set()
    for row in rows:
        cell = (row.get("task_name"), row.get("difficulty"), row.get("seed"), row.get("policy_name"))
        by_cell[cell] = bool(row.get("sr_b") or row.get("success"))
        if row.get("spawn_failed"):
            failed.add(cell[:3])
    c1, c0 = [], []
    for task, difficulty, seed, policy in list(by_cell):
        if policy != "c1":
            continue
        if (task, difficulty, seed) in failed:
            continue
        other = by_cell.get((task, difficulty, seed, "c0"))
        if other is None:
            continue
        c1.append(by_cell[(task, difficulty, seed, "c1")])
        c0.append(other)
    report = {
        "n_episodes": len(rows),
        "by_task_difficulty_policy": table,
        "mcnemar_c1_vs_c0": mcnemar(c0, c1) if c1 else None,
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    report = summarize_baseline(Path(args.baseline))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
