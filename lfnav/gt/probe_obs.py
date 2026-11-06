"""Print one MineDojo observation. Run this inside the container; do not guess key names.

Writes a JSON summary of keys, shapes, and a few scalar samples. Also prints action_space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lfnav.io import atomic_write_json


def summarize(value, depth: int = 0):
    if depth > 4:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {str(k): summarize(v, depth + 1) for k, v in list(value.items())[:60]}
    shape = getattr(value, "shape", None)
    if shape is not None:
        flat = None
        try:
            flat = value.reshape(-1)[:3].tolist()
        except Exception:
            flat = None
        return {"type": type(value).__name__, "dtype": str(getattr(value, "dtype", "")), "shape": list(shape), "head": flat}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "len": len(value), "head": summarize(value[0], depth + 1) if value else None}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)[:300]}


def build_env(make_json: str | None):
    import minedojo
    if make_json:
        kwargs = json.loads(Path(make_json).read_text())
    else:
        kwargs = {"task_id": "harvest_milk", "image_size": [288, 512], "world_seed": 0, "seed": 0}
    if "image_size" in kwargs:
        kwargs["image_size"] = tuple(kwargs["image_size"])
    env = minedojo.make(**kwargs)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--make-json", default=None, help="optional MineDojo make() kwargs. Defaults to the small harvest_milk check used by validate_minedojo.py")
    args = parser.parse_args()
    env = build_env(args.make_json)
    obs = env.reset()
    summary = {
        "keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
        "observation": summarize(obs),
        "action_space": summarize(getattr(env, "action_space", None)),
        "note": "Fill configs/lfnav/obs_keys.yaml from these keys. Leave a key null if it is absent.",
    }
    atomic_write_json(args.out, summary)
    print(json.dumps({"keys": summary["keys"], "out": args.out}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
