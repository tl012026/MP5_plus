"""Inventory success check and failure_category rules. No LLM."""

from __future__ import annotations

from pathlib import Path

from utils import update_inventory_obj_name


def classify_task_kind(task_file):
    parts = Path(task_file).parts
    if "process" in parts:
        return "process"
    if "creative" in parts:
        return "creative"
    return "unknown"


def inventory_has_item(inventory, task_information):
    """True if the final-goal item is in inventory at the required quantity."""
    raw = task_information.get("task")
    qty = int(task_information.get("quantity") or 1)
    names = {raw, update_inventory_obj_name(raw)}
    if raw:
        names.add(raw.replace("_", " "))
    for name in names:
        if name and inventory.get(name, 0) >= qty:
            return True
    return False


def snapshot_obs(obs):
    """Record privileged signals that exist on a MineDojo observation, if present."""
    if not isinstance(obs, dict):
        return {}
    snap = {}
    loc = obs.get("location_stats")
    if isinstance(loc, dict):
        for k in ("pos", "biome_id", "is_raining", "can_see_sky", "temperature", "rainfall", "sea_level"):
            if k in loc:
                snap[k] = _jsonable(loc[k])
    rays = obs.get("rays")
    if isinstance(rays, dict):
        if "block_name" in rays:
            snap["lidar_blocks"] = _unique(rays["block_name"])
        if "entity_name" in rays:
            snap["lidar_entities"] = _unique(rays["entity_name"])
    life = obs.get("life_stats")
    if isinstance(life, dict) and "life" in life:
        snap["life"] = _jsonable(life["life"])
    return snap


def _unique(arr):
    try:
        import numpy as np
        vals = [str(x) for x in np.asarray(arr).reshape(-1).tolist()]
    except Exception:
        vals = [str(x) for x in arr]
    return sorted({v for v in vals if v and v not in ("null", "None", "air")})


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
    except Exception:
        pass
    return x


def failure_category(outcome, last_action_find, planner_parse_failed, retries_exhausted):
    """
    Rules (first match wins):
      death                    -> death
      error                    -> exception
      timeout and last find    -> timeout_in_find
      timeout                  -> timeout_other
      planner returned empty / parse failed and retries exhausted -> parse_error
      retries exhausted        -> planning_retry_exhausted
      otherwise                -> None (success / self_reported / fail-with-inventory)
    """
    if outcome == "death":
        return "death"
    if outcome == "error":
        return "exception"
    if outcome == "timeout":
        return "timeout_in_find" if last_action_find else "timeout_other"
    if retries_exhausted and planner_parse_failed:
        return "parse_error"
    if retries_exhausted:
        return "planning_retry_exhausted"
    return None
