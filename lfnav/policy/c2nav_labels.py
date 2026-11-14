"""Mode, waypoint, and stop labels for C2Nav. Old demos are not used for mode/waypoints."""

from __future__ import annotations

from lfnav.policy.state_tracker import FORWARD_ACTIONS, TURN_ACTIONS, wrap_deg

MODE_NAMES = ("scan", "explore", "approach", "reacquire")
FORWARD_BINS = (-1, 0, 1, 2)
TURN_BINS = (-135, -90, -45, -15, 0, 15, 45, 90, 135, 180)
N_TURN = len(TURN_BINS)
USED_WIDE_MODES = {"wide_turn"}
KNOWN_BRANCHES = set(MODE_NAMES) | USED_WIDE_MODES | {"stop"}

YAW_TO_ACTION = {
    -45.0: "turn_left_45",
    -15.0: "turn_left_15",
    15.0: "turn_right_15",
    45.0: "turn_right_45",
}


def is_v3_demo(record: dict, folder=None) -> bool:
    from pathlib import Path

    if record.get("expert_branch") is not None or (record.get("demo_version") == "v3"):
        return True
    parts = [] if folder is None else Path(folder).parts
    return "demos_main_v3" in parts


def mode_from_branch(branch: str | None, used_wide: bool = False) -> tuple[int, bool]:
    """Return (mode_index, mask_wide). stop is not a mode class."""
    name = "" if branch is None else str(branch)
    wide = False
    if used_wide is True:
        wide = True
    if name in USED_WIDE_MODES:
        wide = True
    if name in MODE_NAMES:
        return MODE_NAMES.index(name), wide
    return 0, wide


def branch_is_mapped(branch: str | None) -> bool:
    if branch is None:
        return False
    return str(branch) in KNOWN_BRANCHES


def net_yaw_forward(actions: list[str]) -> tuple[float, int]:
    yaw = 0.0
    forward = 0
    for name in actions or []:
        yaw += float(TURN_ACTIONS.get(str(name), 0.0))
        if str(name) in FORWARD_ACTIONS:
            forward += 1
        elif str(name) == "backward":
            forward -= 1
    return wrap_deg(yaw), int(forward)


def turn_index(yaw_deg: float) -> int:
    yaw = wrap_deg(yaw_deg)
    best_i = 0
    best_dist = abs(yaw - float(TURN_BINS[0]))
    best_abs = abs(float(TURN_BINS[0]))
    for i in range(1, len(TURN_BINS)):
        center = float(TURN_BINS[i])
        dist = abs(yaw - center)
        center_abs = abs(center)
        if dist < best_dist:
            best_i = i
            best_dist = dist
            best_abs = center_abs
        elif dist == best_dist and center_abs > best_abs:
            best_i = i
            best_abs = center_abs
    return int(best_i)


def turn_center(index: int) -> float:
    return float(TURN_BINS[int(index) % N_TURN])


def forward_index(blocks: float) -> int:
    n = int(round(float(blocks)))
    if n <= -1:
        return 0
    elif n == 0:
        return 1
    elif n == 1:
        return 2
    else:
        return 3


def forward_blocks_from_class(index: int) -> int:
    return int(FORWARD_BINS[int(index) % len(FORWARD_BINS)])


def compose_waypoint(actions: list[str]) -> dict:
    yaw, forward = net_yaw_forward(actions)
    t_i = turn_index(yaw)
    f_i = forward_index(forward)
    return {
        "turn_index": t_i,
        "forward_index": f_i,
        "yaw_deg": yaw,
        "forward": FORWARD_BINS[f_i],
        "turn_center": turn_center(t_i),
        "forward_raw": int(forward),
    }


def invert_waypoint(turn_i: int, forward_i: int) -> list[str]:
    remaining = turn_center(int(turn_i))
    actions: list[str] = []
    sign = 1 if remaining >= 0 else -1
    n45 = int(abs(remaining) // 45.0)
    actions.extend(["turn_right_45" if sign > 0 else "turn_left_45"] * n45)
    remaining = wrap_deg(remaining - sign * 45.0 * n45)
    n15 = int(round(remaining / 15.0))
    if n15 != 0:
        actions.extend(["turn_right_15" if n15 > 0 else "turn_left_15"] * abs(n15))
    blocks = forward_blocks_from_class(int(forward_i))
    if blocks < 0:
        actions.extend(["backward"] * abs(blocks))
    elif blocks > 0:
        actions.extend(["forward"] * int(blocks))
    return actions


def flip_turn_index(index: int) -> int:
    return turn_index(-turn_center(int(index)))


def gt_stop(step: dict, radius: float = 3.0) -> bool:
    """Target visible, nearest numeric distance <= radius, every condition holds."""
    match = step.get("match") or {}
    satisfied = dict(match.get("satisfied") or {})
    missing = list(match.get("missing") or [])
    visible = bool(
        match.get("target_direction")
        or satisfied.get("target")
        or (step.get("target_distance_num") is not None)
    )
    dist = step.get("target_distance_num")
    if dist is None:
        dist = match.get("aligned_distance")
    in_range = dist is not None and float(dist) <= float(radius)
    cond_ok = (len(missing) == 0) and (len(satisfied) == 0 or all(bool(v) for v in satisfied.values()))
    if len(missing) != 0:
        cond_ok = False
    return bool(visible and in_range and cond_ok)


def labels_from_step(step: dict, radius: float = 3.0) -> dict | None:
    """None when the step cannot supervise mode/waypoints (legacy demos)."""
    del radius
    if step.get("expert_branch") is None and not step.get("executed_actions") and step.get("pos") is None:
        return None
    branch = step.get("expert_branch")
    if branch is None:
        return None
    name = str(branch)
    used_wide = False
    if step.get("used_wide") is True:
        used_wide = True
    if name in USED_WIDE_MODES:
        used_wide = True
    mode_i, wide = mode_from_branch(name, used_wide)
    actions = list(step.get("executed_actions") or step.get("action_chunk") or [])
    wp = compose_waypoint(actions)
    is_stop = name == "stop"
    return {
        "mode": mode_i,
        "used_wide": wide,
        "turn_index": wp["turn_index"],
        "forward_index": wp["forward_index"],
        "stop": 1 if is_stop else 0,
        "has_policy": (is_stop is False) and (wide is False),
    }
