"""Sweep matcher thresholds on a JSON list of {task, scene, found, stop}."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from lfnav.matcher import Matcher
from lfnav.types import SceneState, TaskSpec


def score(rows, taus, stop_mode: str) -> dict:
    cfg = {"match": {"tau_obj": taus[0], "tau_cond": taus[1], "tau_near": taus[2], "stop_mode": stop_mode, "stop_rule": "fraction", "smooth_n": 1}}
    matcher = Matcher(cfg)
    found_ok = stop_ok = 0
    for row in rows:
        pred = matcher.match(TaskSpec.from_dict(row["task"]), SceneState.from_dict(row["scene"]))
        found_ok += int((pred.status == "found") == bool(row["found"]))
        stop_ok += int(pred.stop == bool(row["stop"]))
    n = max(len(rows), 1)
    return {"tau_obj": taus[0], "tau_cond": taus[1], "tau_near": taus[2],
            "found_acc": found_ok / n, "stop_acc": stop_ok / n}


def sweep(rows, grid, stop_mode="sr_b") -> list[dict]:
    out = [score(rows, taus, stop_mode) for taus in product(grid, grid, grid)]
    out.sort(key=lambda r: (r["stop_acc"], r["found_acc"]), reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--grid", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--stop-mode", default="sr_b")
    args = ap.parse_args()
    rows = json.loads(Path(args.data).read_text())
    ranked = sweep(rows, args.grid, args.stop_mode)
    print(json.dumps(ranked[:10], indent=2))


if __name__ == "__main__":
    main()
