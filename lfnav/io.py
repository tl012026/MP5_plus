"""Small JSON helpers. Mirrors eval/run_batch.atomic_write_json without importing the agent."""

from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def read_seeds(path) -> list[int]:
    seeds = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        seeds.append(int(line))
    return seeds


def read_tasks(values) -> list[str]:
    out = []
    for v in values:
        p = Path(v)
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        else:
            out.append(p)
    return [str(p) for p in out]
