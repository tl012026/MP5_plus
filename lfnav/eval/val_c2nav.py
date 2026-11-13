"""Run the four C2Nav validation episodes and write val.json next to the checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lfnav.config import load_config
from lfnav.eval.run import EnvReuse, open_episode, run_episode
from lfnav.eval.summarize_c2nav import summarize_val
from lfnav.io import atomic_write_json
from lfnav.types import TaskSpec

CELLS = (
    ("1-4", "easy"),
    ("2-3", "medium"),
    ("3-4", "hard"),
    ("4-4", "medium"),
)
SEED = 13
MAX_STEPS = 1000


def spec_path(root: Path, task_id: str) -> Path:
    return root / "lfnav" / "tasks" / "specs" / f"{task_id}.json"


def run_val(cfg: dict, ckpt: Path, out: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    cfg = dict(cfg)
    cfg["policy_name"] = "c2nav"
    cfg["perception_backend"] = "c2nav"
    cfg["c2nav"] = dict(cfg.get("c2nav") or {})
    cfg["c2nav"]["checkpoint"] = str(ckpt)
    cfg["judge"] = dict(cfg.get("judge") or {})
    cfg["judge"]["max_steps"] = MAX_STEPS
    cfg["frames"] = dict(cfg.get("frames") or {}, enabled=False)
    cfg["save_decision_frames"] = False
    out.mkdir(parents=True, exist_ok=True)
    session = EnvReuse(cfg)
    records = []
    try:
        for task_id, difficulty in CELLS:
            cfg["difficulty"] = difficulty
            task = TaskSpec.from_dict(json.loads(spec_path(root, task_id).read_text()))
            dest = out / task_id / f"{SEED}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            env, obs, meta = open_episode(session, task, cfg, SEED)
            rec, events = run_episode(env, task, cfg, seed=SEED, obs=obs, episode_meta=meta)
            atomic_write_json(dest, rec)
            event_path = out / task_id / str(SEED) / "events.jsonl"
            event_path.parent.mkdir(parents=True, exist_ok=True)
            event_path.write_text("".join(json.dumps(ev) + "\n" for ev in events))
            records.append(rec)
            session.release(keep=True)
    finally:
        session.close()
    summary = summarize_val(out, ckpt=ckpt)
    atomic_write_json(out / "val.json", summary)
    if ckpt.is_file():
        atomic_write_json(ckpt.parent / (ckpt.stem + ".val.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", default=os.environ.get("CKPT"))
    parser.add_argument("--out", default=os.environ.get("VAL_OUT"))
    parser.add_argument("--backend", default=None)
    args = parser.parse_args()
    if not args.ckpt or not args.out:
        raise SystemExit("need --ckpt and --out (or CKPT and VAL_OUT)")
    cfg = load_config(args.config) if args.config else load_config()
    if args.config is None:
        from lfnav.config import repo_root
        cfg = load_config(repo_root() / "configs" / "lfnav" / "main.yaml")
    if args.backend:
        cfg["backend"] = args.backend
    print(run_val(cfg, Path(args.ckpt), Path(args.out)))


if __name__ == "__main__":
    main()
