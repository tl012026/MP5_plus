"""Summarize C2Nav validation episodes and pick a stop threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TAUS = (0.3, 0.5, 0.7)
TURN_NAMES = ("turn_left_15", "turn_right_15", "turn_left_45", "turn_right_45")


def _load_events(folder: Path) -> list[dict]:
    path = folder / "events.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _executed(event: dict) -> list[str]:
    names = []
    info = (event.get("policy") or {}).get("info") or {}
    if info.get("executed_actions"):
        names.extend(info["executed_actions"])
    for control in event.get("control") or []:
        names.extend(control.get("executed") or [])
    if not names:
        names.extend(((event.get("policy") or {}).get("action_chunk")) or [])
    return [str(n) for n in names]


def episode_stats(record: dict, events: list[dict]) -> dict:
    n_turn_hidden = 0
    n_hidden = 0
    for event in events:
        match = event.get("match") or {}
        visible = bool(match.get("target_direction") or (match.get("satisfied") or {}).get("target"))
        executed = _executed(event)
        if not visible:
            n_hidden += max(len(executed), 1)
            n_turn_hidden += sum(1 for n in executed if n in TURN_NAMES)
    p_stops = []
    gt_stops = []
    for event in events:
        info = (event.get("policy") or {}).get("info") or {}
        if "p_stop" in info:
            p_stops.append(float(info["p_stop"]))
            gt_stops.append(1 if (event.get("match") or {}).get("stop") else 0)
    return {
        "task": record.get("task_name") or record.get("task"),
        "difficulty": record.get("difficulty"),
        "sr_b": bool(record.get("sr_b")),
        "steps": int(record.get("steps") or record.get("self_reported_steps") or 0),
        "path_length": float(record.get("path_length") or 0.0),
        "stop_distance": record.get("distance_blocks"),
        "turn_when_hidden": None if n_hidden == 0 else n_turn_hidden / n_hidden,
        "p_stop": p_stops,
        "gt_stop": gt_stops,
    }


def _f1(y, p, tau: float) -> float:
    pred = [1 if v >= tau else 0 for v in p]
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def _offline_sr(rows: list[dict], tau: float) -> float:
    wins = 0
    for row in rows:
        ps = row.get("p_stop") or []
        gts = row.get("gt_stop") or []
        stopped = False
        success = False
        for p, g in zip(ps, gts):
            if p >= tau:
                stopped = True
                success = bool(g)
                break
        if not stopped:
            success = bool(row.get("sr_b"))
        wins += int(success)
    return wins / max(len(rows), 1)


def summarize_val(run_dir, ckpt=None) -> dict:
    run_dir = Path(run_dir)
    recs = []
    rows = []
    for path in sorted(run_dir.glob("*/*.json")):
        if path.name in ("val.json", "run_meta.json"):
            continue
        rec = json.loads(path.read_text())
        if "sr_b" not in rec and "success" not in rec:
            continue
        events = _load_events(path.parent / path.stem)
        if not events:
            events = _load_events(path.parent / str(rec.get("seed")))
        recs.append(rec)
        rows.append(episode_stats(rec, events))
    n = max(len(recs), 1)
    sr = sum(1 for r in recs if r.get("sr_b") or r.get("success")) / n
    steps = [int(r.get("steps") or 0) for r in recs]
    paths = [float(r.get("path_length") or 0.0) for r in recs]
    hidden = [r["turn_when_hidden"] for r in rows if r.get("turn_when_hidden") is not None]
    y, p = [], []
    for row in rows:
        y.extend(row.get("gt_stop") or [])
        p.extend(row.get("p_stop") or [])
    tau_rows = []
    for tau in TAUS:
        tau_rows.append({
            "tau": tau,
            "stop_f1": _f1(y, p, tau) if y else None,
            "offline_sr_b": _offline_sr(rows, tau),
        })
    best = max(tau_rows, key=lambda r: (r["offline_sr_b"] if r["offline_sr_b"] is not None else -1, r["stop_f1"] or -1))
    summary = {
        "ckpt": None if ckpt is None else str(ckpt),
        "n": len(recs),
        "sr_b": sr,
        "mean_steps": sum(steps) / n,
        "mean_path_length": sum(paths) / n,
        "mean_stop_distance": None
        if not any(r.get("stop_distance") is not None for r in rows)
        else sum(float(r["stop_distance"] or 0.0) for r in rows) / n,
        "turn_when_hidden": None if not hidden else sum(hidden) / len(hidden),
        "episodes": rows,
        "tau_sweep": tau_rows,
        "chosen_tau": best["tau"],
        "stop_f1": best["stop_f1"],
    }
    return summary


def choose_best(ckpt_dir) -> dict | None:
    ckpt_dir = Path(ckpt_dir)
    scored = []
    for path in sorted(ckpt_dir.glob("epoch_*.val.json")):
        data = json.loads(path.read_text())
        scored.append((float(data.get("sr_b") or 0.0), float(data.get("stop_f1") or 0.0), path, data))
    if not scored:
        return None
    scored.sort(reverse=True)
    _sr, _f1, path, data = scored[0]
    data["best_from"] = str(path)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--choose-best", action="store_true")
    args = parser.parse_args()
    if args.choose_best:
        print(json.dumps(choose_best(args.inp), indent=2))
        return
    print(json.dumps(summarize_val(args.inp, ckpt=args.ckpt), indent=2))


if __name__ == "__main__":
    main()
