"""Mode, waypoint, and stop labels for C2Nav. Old demos are not used for mode/waypoints."""

from __future__ import annotations

from lfnav.policy.state_tracker import FORWARD_ACTIONS, TURN_ACTIONS, wrap_deg

MODE_NAMES = ("scan", "approach", "reacquire", "explore", "stop")
FORWARD_BINS = (-1, 0, 1, 2, 3, 4)
# Explicit class ids. Do not use `x or 1`: forward 0 must stay class 1.
FORWARD_TO_CLASS = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
CLASS_TO_FORWARD = {v: k for k, v in FORWARD_TO_CLASS.items()}
N_TURN = 12
TURN_WIDTH = 30.0
USED_WIDE_MODES = {"wide_turn"}
KNOWN_BRANCHES = set(MODE_NAMES) | USED_WIDE_MODES

YAW_TO_ACTION = {
    -45.0: "turn_left_45",
    -15.0: "turn_left_15",
    15.0: "turn_right_15",
    45.0: "turn_right_45",
}


def is_v3_demo(record: dict, folder=None) -> bool:
    from pathlib import Path

    if record.get("expert_branch") or (record.get("demo_version") == "v3"):
        return True
    parts = [] if folder is None else Path(folder).parts
    return "demos_main_v3" in parts


def mode_from_branch(branch: str | None, used_wide: bool = False) -> tuple[int, bool]:
    """Return (mode_index, used_wide). wide_turn maps to approach with used_wide."""
    name = "" if branch is None else str(branch)
    wide = bool(used_wide) or name in USED_WIDE_MODES
    if name in USED_WIDE_MODES:
        name = "approach"
    if name not in MODE_NAMES:
        name = "explore"
    return MODE_NAMES.index(name), wide


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
    return int((yaw + TURN_WIDTH / 2.0) % 360.0 // TURN_WIDTH) % N_TURN


def turn_center(index: int) -> float:
    return wrap_deg(int(index) * TURN_WIDTH)


def clip_forward_blocks(blocks: float) -> tuple[int, bool]:
    raw = int(round(float(blocks)))
    lo, hi = int(FORWARD_BINS[0]), int(FORWARD_BINS[-1])
    clipped = lo if raw < lo else hi if raw > hi else raw
    return clipped, clipped != raw


def forward_index(blocks: float, clip_counts: list[int] | None = None) -> int:
    clipped, was_clipped = clip_forward_blocks(blocks)
    if was_clipped and clip_counts is not None:
        clip_counts.append(1)
    return int(FORWARD_TO_CLASS[clipped])


def forward_blocks_from_class(index: int) -> int:
    return int(CLASS_TO_FORWARD[int(index)])


def compose_waypoint(actions: list[str]) -> dict:
    yaw, forward = net_yaw_forward(actions)
    t_i = turn_index(yaw)
    clipped, was_clipped = clip_forward_blocks(forward)
    f_i = forward_index(clipped)
    return {
        "turn_index": t_i,
        "forward_index": f_i,
        "yaw_deg": yaw,
        "forward": FORWARD_BINS[f_i],
        "turn_center": turn_center(t_i),
        "forward_clipped": was_clipped,
        "forward_raw": int(round(float(forward))),
    }


def invert_waypoint(turn_i: int, forward_i: int) -> list[str]:
    remaining = turn_center(int(turn_i))
    actions: list[str] = []
    sign = 1 if remaining >= 0 else -1
    n45 = int(abs(remaining) // 45.0)
    actions.extend(["turn_right_45" if sign > 0 else "turn_left_45"] * n45)
    remaining = wrap_deg(remaining - sign * 45.0 * n45)
    n15 = int(round(remaining / 15.0))
    if n15:
        actions.extend(["turn_right_15" if n15 > 0 else "turn_left_15"] * abs(n15))
    blocks = forward_blocks_from_class(int(forward_i) % len(FORWARD_BINS))
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
    cond_ok = (not missing) and (not satisfied or all(bool(v) for v in satisfied.values()))
    if missing:
        cond_ok = False
    return bool(visible and in_range and cond_ok)


def labels_from_step(step: dict, radius: float = 3.0) -> dict | None:
    """None when the step cannot supervise mode/waypoints (legacy demos)."""
    if step.get("expert_branch") is None and not step.get("executed_actions") and step.get("pos") is None:
        return None
    branch = step.get("expert_branch")
    if branch is None:
        return None
    used_wide = bool(step.get("used_wide"))
    mode_i, wide = mode_from_branch(branch, used_wide)
    actions = list(step.get("executed_actions") or step.get("action_chunk") or [])
    wp = compose_waypoint(actions)
    stop = True if branch == "stop" else gt_stop(step, radius=radius)
    if branch == "stop":
        mode_i = MODE_NAMES.index("stop")
        wide = False
    return {
        "mode": mode_i,
        "used_wide": wide,
        "turn_index": wp["turn_index"],
        "forward_index": wp["forward_index"],
        "forward_clipped": bool(wp.get("forward_clipped")),
        "stop": int(bool(stop)),
        "has_policy": branch != "stop",
    }
