"""MineRL 1.0 action dicts (what STEVE-1 emits) and a MineDojo MultiDiscrete view for logs.

MineRL camera is [pitch_delta_deg, yaw_delta_deg]. MineDojo camera bins are 15 degrees
with 12 meaning zero (cam_interval=15), so the conversion is lossy. It is only for
logging and for a future MineDojo port; STEVE-1 itself runs in MineRL 1.16.5.
"""

from __future__ import annotations

BUTTONS = (
    "forward", "back", "left", "right", "jump", "sneak", "sprint",
    "attack", "use", "drop", "inventory", "ESC", "swapHands", "pickItem",
) + tuple(f"hotbar.{i}" for i in range(1, 10))


def noop() -> dict:
    act = {k: 0 for k in BUTTONS}
    act["camera"] = [0.0, 0.0]
    return act


def make(**kwargs) -> dict:
    act = noop()
    for key, value in kwargs.items():
        if key == "camera":
            act["camera"] = [float(value[0]), float(value[1])]
        else:
            if key not in act:
                raise KeyError(f"unknown MineRL action key {key!r}")
            act[key] = int(value)
    return act


def compact(act: dict) -> dict:
    """Only the active keys, for small trace rows."""
    out = {k: v for k, v in act.items() if k != "camera" and v}
    cam = act.get("camera") or [0.0, 0.0]
    if cam[0] or cam[1]:
        out["camera"] = [round(float(cam[0]), 2), round(float(cam[1]), 2)]
    return out


def _cam_bin(deg: float, interval: float = 15.0) -> int:
    return max(0, min(24, int(round(float(deg) / interval)) + 12))


def to_minedojo(act: dict) -> list[int]:
    """[fwd/back, left/right, jump/sneak/sprint, pitch, yaw, functional, craft_arg, inv_arg]."""
    move = 1 if act.get("forward") else 2 if act.get("back") else 0
    strafe = 1 if act.get("left") else 2 if act.get("right") else 0
    jss = 1 if act.get("jump") else 2 if act.get("sneak") else 3 if act.get("sprint") else 0
    cam = act.get("camera") or [0.0, 0.0]
    functional = 3 if act.get("attack") else 1 if act.get("use") else 2 if act.get("drop") else 0
    return [move, strafe, jss, _cam_bin(cam[0]), _cam_bin(cam[1]), functional, 0, 0]
