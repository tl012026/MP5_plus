#!/usr/bin/env python3
"""Batch eval entry. One process per worker; each process builds its own env via run_episode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGENT = REPO / "MP5_agent" / "agent"
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))

from config import load_config  # noqa: E402
from run_agent import run_episode  # noqa: E402


def read_seeds(path):
    seeds = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        seeds.append(int(line))
    return seeds


def read_tasks(values):
    out = []
    for v in values:
        p = Path(v)
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        else:
            out.append(p)
    return [str(p) for p in out]


def task_name(task_file):
    return Path(task_file).stem


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return None


def _one(args):
    task_file, seed, config, out_root = args
    name = task_name(task_file)
    dest = Path(out_root) / name / f"{seed}.json"
    ep_dir = Path(out_root) / name / f"{seed}"
    rec = run_episode(task_file, seed, config, ep_dir)
    rec["task_name"] = name
    atomic_write_json(dest, rec)
    return str(dest), rec.get("outcome")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--mllm-url", default="")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.mllm_url:
        config["percipient"]["mllm_url"] = args.mllm_url

    tasks = read_tasks(args.tasks)
    seeds = read_seeds(args.seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    atomic_write_json(out / "run_meta.json", {
        "config_path": str(Path(args.config).resolve()),
        "config": {k: v for k, v in config.items() if k != "_path"},
        "git_commit": git_hash(),
        "argv": sys.argv,
        "tasks": tasks,
        "seeds": seeds,
        "workers": args.workers,
    })

    jobs = []
    for task in tasks:
        for seed in seeds:
            dest = out / task_name(task) / f"{seed}.json"
            if args.skip_done and dest.is_file():
                continue
            jobs.append((task, seed, config, str(out)))

    if args.workers <= 1:
        for job in jobs:
            _one(job)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, job) for job in jobs]
            for fut in as_completed(futs):
                try:
                    dest, outcome = fut.result()
                    print(dest, outcome, flush=True)
                except Exception as e:
                    print("worker failed:", repr(e), flush=True)

    from summarize import write_summary
    write_summary(out)
    print("done", out / "summary.csv")


if __name__ == "__main__":
    main()
