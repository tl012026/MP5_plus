#!/usr/bin/env python3
"""Rebuild summary.csv from OUT/<task>/<seed>.json (safe to run on a partial directory)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "task_name", "seed", "task_kind", "outcome", "success", "steps", "elapsed_s",
    "failure_category", "self_reported_steps",
]


def collect(out_dir):
    rows = []
    for path in sorted(Path(out_dir).glob("*/*.json")):
        if path.name in ("run_meta.json",) or path.name.endswith(".tmp"):
            continue
        if path.parent.name in ("",):
            continue
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "seed" not in rec:
            continue
        row = {k: rec.get(k) for k in FIELDS}
        row["task_name"] = rec.get("task_name") or path.parent.name
        rows.append(row)
    return rows


def write_summary(out_dir):
    out_dir = Path(out_dir)
    rows = collect(out_dir)
    dest = out_dir / "summary.csv"
    with open(dest, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    args = ap.parse_args()
    dest = write_summary(args.out_dir)
    print(dest)


if __name__ == "__main__":
    main()
