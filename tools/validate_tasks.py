#!/usr/bin/env python3
"""Check task JSON field completeness and, for process tasks, recipe-chain coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGENT = REPO / "MP5_agent" / "agent"

# Same mapping as utils.common_utils.update_inventory_obj_name (avoid importing
# common_utils, which pulls the OpenAI SDK).
def update_inventory_obj_name(obj_name):
    if obj_name == "wood" or obj_name == "tree":
        return "log"
    if obj_name == "stone":
        return "cobblestone"
    if obj_name == "diamond ore":
        return "diamond"
    return obj_name

REQUIRED = ("task", "quantity", "material", "tool", "platform", "tips", "description")
# Craft yields used only for chain accounting (vanilla 1.11 / MineDojo).
YIELD = {
    "planks": 4,  # per 1 log, handled specially
    "stick": 4,   # per 2 planks -> 4 stick; we count output quantity field
}


def load_tasks(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty list")
    return data


def check_fields(path, items):
    errors = []
    for i, item in enumerate(items):
        for k in REQUIRED:
            if k not in item:
                errors.append(f"{path}[{i}] missing field {k}")
        if not item.get("task"):
            errors.append(f"{path}[{i}] empty task")
        if item.get("quantity") is not None:
            try:
                int(item["quantity"])
            except (TypeError, ValueError):
                errors.append(f"{path}[{i}] quantity not int")
        if item.get("material") is not None and not isinstance(item["material"], dict):
            errors.append(f"{path}[{i}] material must be dict or null")
    return errors


def check_process_chain(path, items):
    """items[0] is the final goal (file order). Execution is reversed."""
    errors = []
    produced = {}
    for item in reversed(items):
        name = update_inventory_obj_name(item["task"])
        qty = int(item.get("quantity") or 1)
        mats = item.get("material") or {}
        for mat, need in mats.items():
            mat_n = update_inventory_obj_name(mat)
            have = produced.get(mat_n, 0)
            if have < int(need) and mat_n not in ("coal",):
                # coal is provided by code_default start inventory; other gaps are real.
                errors.append(
                    f"{path}: when crafting {name}, need {need} {mat_n} but chain only produced {have} before this step"
                )
            elif mat_n not in ("coal",):
                produced[mat_n] = have - int(need)
        produced[name] = produced.get(name, 0) + qty
        produced[item["task"]] = produced.get(item["task"], 0) + qty
    final = update_inventory_obj_name(items[0]["task"])
    if produced.get(final, 0) < int(items[0]["quantity"] or 1):
        errors.append(f"{path}: final goal {final} not produced")
    return errors


def iter_task_files(roots):
    files = []
    for root in roots:
        p = Path(root)
        if p.is_file():
            files.append(p)
        else:
            files.extend(sorted(p.rglob("*.json")))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "roots",
        nargs="*",
        default=[str(AGENT / "tasks" / "creative"), str(AGENT / "tasks" / "process")],
    )
    args = ap.parse_args()
    errors = []
    n = 0
    for path in iter_task_files(args.roots):
        n += 1
        try:
            items = load_tasks(path)
        except Exception as e:
            errors.append(str(e))
            continue
        errors.extend(check_fields(path, items))
        if "process" in path.parts:
            errors.extend(check_process_chain(path, items))
    if errors:
        print("FAIL")
        for e in errors:
            print(" ", e)
        return 1
    print(f"OK {n} task files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
