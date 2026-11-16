"""Format of the file the MineDreamer performer writes for every MP5 action.

One JSON per action (`<out>/<index>_<action>.json`) plus a per-step `<...>.trace.jsonl`.
`check_result` has exactly the keys MP5's Performer/Patroller exchange
(success / feedback / suggestion), so run_agent.py can consume it unchanged.

{
  "schema": "mp5.minedreamer_performer.result/v1",
  "created": "2026-11-15T12:00:00Z",
  "action": {"name": "mine", "args": {"obj": "wood", "tool": null}},     # verbatim from the planner
  "instruction": {                                                        # instructions.Instruction
    "action": "mine", "text_prompt": "chop a tree", "in_distribution": true,
    "max_steps": 1200, "done": {"kind": "inventory_gain", "item": "log", "count": 1},
    "supported": true, "note": ""},
  "backend": {"env": "...", "imaginator": "...", "prompt_generator": "...", "policy": "...",
              "freq": 25, "seed": 0},
  "executor": "minedreamer | command_craft | none",                        # who ran the action
  "commands": ["/clear @p minecraft:oak_planks 3", "/give @p minecraft:wooden_pickaxe 1"],  # command_craft only
  "outcome": "done | budget_exhausted | unsupported | noop | precheck_fail | error",
  "steps": 137,                         # env steps spent on this action
  "imagine_calls": 6,
  "imagine_latency_s": {"total": 0.01, "max": 0.002},
  "inventory_before": {}, "inventory_after": {"log": 1}, "inventory_delta": {"log": 1},   # MP5 names
  "pose_before": {"pos": [0, 64, 0], "yaw": 10.0, "pitch": 0.0}, "pose_after": {...},
  "done_check": {"kind": "inventory_gain", "satisfied": true, "value": 1, "checked_at_step": 137},
  "imaginations": [{"step": 0, "goal_image": "frames/0000_goal_0000.png" | null, "latency_s": 0.0}],
  "trace_file": "0000_mine.trace.jsonl",
  "check_result": {"success": true, "feedback": "...", "suggestion": ""},
  "error": null
}

Trace rows: {"t": 12, "minerl": {"forward": 1, "camera": [0, 5.0]}, "minedojo": [1,0,0,12,12,0,0,0],
             "pos": [...], "yaw": ..., "pitch": ..., "event": "mined:oak_log" | null}
"""

from __future__ import annotations

SCHEMA_ID = "mp5.minedreamer_performer.result/v1"

OUTCOMES = ("done", "budget_exhausted", "unsupported", "noop", "precheck_fail", "error")

_REQUIRED = {
    "schema": str, "created": str, "action": dict, "instruction": dict, "backend": dict,
    "outcome": str, "steps": int, "imagine_calls": int, "imagine_latency_s": dict,
    "inventory_before": dict, "inventory_after": dict, "inventory_delta": dict,
    "pose_before": (dict, type(None)), "pose_after": (dict, type(None)),
    "done_check": dict, "imaginations": list, "trace_file": (str, type(None)),
    "check_result": dict,
}


def validate(record: dict) -> list[str]:
    """Return a list of problems; empty means the record matches v1."""
    problems = []
    for key, typ in _REQUIRED.items():
        if key not in record:
            problems.append(f"missing {key}")
        elif not isinstance(record[key], typ):
            problems.append(f"{key} has type {type(record[key]).__name__}")
    if record.get("schema") != SCHEMA_ID:
        problems.append(f"schema is {record.get('schema')!r}")
    if record.get("outcome") not in OUTCOMES:
        problems.append(f"outcome {record.get('outcome')!r} not in {OUTCOMES}")
    cr = record.get("check_result") or {}
    for key in ("success", "feedback", "suggestion"):
        if key not in cr:
            problems.append(f"check_result missing {key}")
    if "success" in cr and not isinstance(cr["success"], bool):
        problems.append("check_result.success must be bool")
    return problems
