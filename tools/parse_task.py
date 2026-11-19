#!/usr/bin/env python3
"""Generate a knowledge-grounded process task file with a real chat request; no game required."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "MP5_agent/agent"))
from config import load_config
from task_parser import parse_task


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs/openrouter.yaml"))
    ap.add_argument("--request", default="Starting with an empty inventory, obtain one wooden pickaxe.",
                    help="Natural-language objective; must agree with --goals")
    ap.add_argument("--goals", default='{"wooden pickaxe": 1}', help="Final inventory requirements as JSON")
    ap.add_argument("--inventory", default="{}", help="Initial inventory JSON; nonempty produces plan only")
    ap.add_argument("--knowledge", default=None, help="Knowledge JSON path; default bundled wood coverage")
    ap.add_argument("--out", default=None, help="New output directory; existing directories are not overwritten")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dest = Path(args.out or REPO / "logs/parser" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    model_cfg = cfg["models"].get("parser", cfg["models"]["planner"])
    print("Online Parser: covered process goals; up to two paid chat requests (generation + one validation repair).", flush=True)
    task = parse_task(args.request, model_cfg, dest, goals=json.loads(args.goals),
                      initial_inventory=json.loads(args.inventory), knowledge_path=args.knowledge)
    print("PASS Parser schema, order and recipe accounting")
    print("Generated artifact (see validation status):", task)
    print("Review execution_order.json and validation.json in:", dest)


if __name__ == "__main__":
    main()
