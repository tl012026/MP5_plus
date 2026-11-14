"""Pure-function pose tracker S. Offline replay and online inference share update()."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

N_SECTORS = 8
DISP_WINDOW = 60
UNSEEN_HORIZON = 40.0
LOG_DIST_UNSEEN = math.log(32.0)
# Feature scale for env_steps. Closed-loop val may cap the episode at 1000;
# the tracker still divides by this constant so train and val stay aligned.
STEPS_NORM = 3000
TURN_ACTIONS = {
    "turn_left_15": -15.0,
    "turn_right_15": 15.0,
    "turn_left_45": -45.0,
    "turn_right_45": 45.0,
}
FORWARD_ACTIONS = {"forward", "jump_forward"}
BACKWARD_ACTIONS = {"backward"}


def wrap_deg(angle: float) -> float:
    x = (float(angle) + 180.0) % 360.0 - 180.0
    if x == -180.0:
        return 180.0
    return x


def _norm(x: float) -> float:
    return 1.0 if abs(x) < 1e-9 else x


@dataclass
class TargetObs:
    visible: bool = False
    bearing_deg: float | None = None
    log_distance: float | None = None

    def to_dict(self) -> dict:
        return {
            "visible": bool(self.visible),
            "bearing_deg": None if self.bearing_deg is None else float(self.bearing_deg),
            "log_distance": None if self.log_distance is None else float(self.log_distance),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "TargetObs":
        data = data or {}
        bearing = data.get("bearing_deg")
        log_d = data.get("log_distance")
        return cls(
            visible=bool(data.get("visible")),
            bearing_deg=None if bearing is None else float(bearing),
            log_distance=None if log_d is None else float(log_d),
        )


@dataclass
class TrackerNoise:
    drop_visible: float = 0.0
    bearing_std: float = 0.0
    log_dist_std: float = 0.0
    odom_std: float = 0.0

    def apply_obs(self, obs: TargetObs, rng) -> TargetObs:
        visible = bool(obs.visible)
        if visible and rng.random() < float(self.drop_visible):
            return TargetObs(visible=False)
        if not visible:
            return TargetObs(visible=False)
        bearing = float(obs.bearing_deg or 0.0)
        log_d = float(obs.log_distance if obs.log_distance is not None else LOG_DIST_UNSEEN)
        if self.bearing_std:
            bearing = wrap_deg(bearing + rng.gauss(0.0, float(self.bearing_std)))
        if self.log_dist_std:
            log_d = log_d + rng.gauss(0.0, float(self.log_dist_std))
        return TargetObs(visible=True, bearing_deg=bearing, log_distance=log_d)

    def apply_pose(self, pos, yaw: float, rng) -> tuple[list[float], float]:
        if not self.odom_std:
            return [float(v) for v in pos], float(yaw)
        s = float(self.odom_std)
        noisy = [float(pos[0]) + rng.gauss(0.0, s), float(pos[1]), float(pos[-1]) + rng.gauss(0.0, s)]
        return noisy, wrap_deg(float(yaw) + rng.gauss(0.0, s * 5.0))


@dataclass
class TrackerState:
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    yaw: float = 0.0
    belief_sin: float = 0.0
    belief_cos: float = 1.0
    log_distance: float = LOG_DIST_UNSEEN
    unseen_steps: float = UNSEEN_HORIZON
    confidence: float = 0.0
    scan_deg: float = 0.0
    turn_since_forward: float = 0.0
    sector_raw: list[float] = field(default_factory=lambda: [0.0] * N_SECTORS)
    pos_hist: list[tuple[float, float]] = field(default_factory=list)
    env_steps: int = 0
    stuck: bool = False
    max_steps: int = 3000
    started: bool = False

    def features(self) -> dict:
        total = sum(self.sector_raw) or 1.0
        sectors = [c / total for c in self.sector_raw]
        if len(self.pos_hist) >= 2:
            x0, z0 = self.pos_hist[0]
            x1, z1 = self.pos_hist[-1]
            disp = math.hypot(x1 - x0, z1 - z0)
        else:
            disp = 0.0
        return {
            "bearing_sin": float(self.belief_sin),
            "bearing_cos": float(self.belief_cos),
            "log_distance": float(self.log_distance),
            "unseen": min(float(self.unseen_steps) / UNSEEN_HORIZON, 1.0),
            "confidence": float(self.confidence),
            "scan_frac": min(abs(float(self.scan_deg)) / 360.0, 1.0),
            "turn_since_forward": wrap_deg(self.turn_since_forward) / 180.0,
            "sectors": sectors,
            "displacement_60": float(disp),
            "stuck": 1.0 if self.stuck else 0.0,
            "steps_frac": min(float(self.env_steps) / float(STEPS_NORM), 1.0),
        }

    def vector(self) -> list[float]:
        feat = self.features()
        return [
            feat["bearing_sin"],
            feat["bearing_cos"],
            feat["log_distance"],
            feat["unseen"],
            feat["confidence"],
            feat["scan_frac"],
            feat["turn_since_forward"],
            *feat["sectors"],
            feat["displacement_60"],
            feat["stuck"],
            feat["steps_frac"],
        ]

    def to_dict(self) -> dict:
        return {
            "pos": list(self.pos),
            "yaw": float(self.yaw),
            "belief_sin": float(self.belief_sin),
            "belief_cos": float(self.belief_cos),
            "log_distance": float(self.log_distance),
            "unseen_steps": float(self.unseen_steps),
            "confidence": float(self.confidence),
            "scan_deg": float(self.scan_deg),
            "turn_since_forward": float(self.turn_since_forward),
            "sector_raw": list(self.sector_raw),
            "pos_hist": [list(p) for p in self.pos_hist],
            "env_steps": int(self.env_steps),
            "stuck": bool(self.stuck),
            "max_steps": int(self.max_steps),
            "started": bool(self.started),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "TrackerState":
        data = data or {}
        hist = []
        for item in data.get("pos_hist") or []:
            hist.append((float(item[0]), float(item[-1])))
        return cls(
            pos=[float(v) for v in (data.get("pos") or [0.0, 0.0, 0.0])],
            yaw=float(data.get("yaw") or 0.0),
            belief_sin=float(data.get("belief_sin") or 0.0),
            belief_cos=float(data.get("belief_cos") if data.get("belief_cos") is not None else 1.0),
            log_distance=float(data.get("log_distance") if data.get("log_distance") is not None else LOG_DIST_UNSEEN),
            unseen_steps=float(data.get("unseen_steps") if data.get("unseen_steps") is not None else UNSEEN_HORIZON),
            confidence=float(data.get("confidence") or 0.0),
            scan_deg=float(data.get("scan_deg") or 0.0),
            turn_since_forward=float(data.get("turn_since_forward") or 0.0),
            sector_raw=[float(v) for v in (data.get("sector_raw") or [0.0] * N_SECTORS)],
            pos_hist=hist,
            env_steps=int(data.get("env_steps") or 0),
            stuck=bool(data.get("stuck")),
            max_steps=int(data.get("max_steps") or 3000),
            started=bool(data.get("started")),
        )


def _sector_index(rel_yaw: float) -> int:
    ang = (float(rel_yaw) % 360.0 + 360.0) % 360.0
    return int((ang + 22.5) // 45.0) % N_SECTORS


def _action_yaw(name: str) -> float:
    return float(TURN_ACTIONS.get(str(name), 0.0))


def _is_forward(name: str) -> bool:
    return str(name) in FORWARD_ACTIONS or str(name) in BACKWARD_ACTIONS


def _clone(state: TrackerState) -> TrackerState:
    return replace(
        state,
        pos=list(state.pos),
        sector_raw=list(state.sector_raw),
        pos_hist=list(state.pos_hist),
    )


def _step_one(
    state: TrackerState,
    pos,
    yaw: float,
    name: str | None,
    obs: TargetObs,
) -> TrackerState:
    nxt = _clone(state)
    pose = [float(v) for v in pos]
    yaw_v = float(yaw)
    if not nxt.started:
        nxt.started = True
        nxt.pos = pose
        nxt.yaw = yaw_v
        nxt.pos_hist = [(pose[0], pose[-1])]
        return _apply_observation(nxt, obs)

    dyaw = _action_yaw(name) if name else wrap_deg(yaw_v - nxt.yaw)
    rad = math.radians(dyaw)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    bx, bz = nxt.belief_sin, nxt.belief_cos
    nxt.belief_sin = bx * cos_t - bz * sin_t
    nxt.belief_cos = bx * sin_t + bz * cos_t
    nrm = _norm(math.hypot(nxt.belief_sin, nxt.belief_cos))
    nxt.belief_sin /= nrm
    nxt.belief_cos /= nrm

    moved = math.hypot(pose[0] - nxt.pos[0], pose[-1] - nxt.pos[-1])
    if moved > 0.05:
        world = math.degrees(math.atan2(pose[0] - nxt.pos[0], pose[-1] - nxt.pos[-1]))
        rel = wrap_deg(world - yaw_v)
        nxt.sector_raw[_sector_index(rel)] += 1.0

    nxt.scan_deg = min(float(nxt.scan_deg) + abs(dyaw), 360.0)
    if (name and _is_forward(name)) or moved > 0.25:
        nxt.turn_since_forward = 0.0
    else:
        nxt.turn_since_forward = wrap_deg(nxt.turn_since_forward + dyaw)

    nxt.pos = pose
    nxt.yaw = yaw_v
    nxt.env_steps += 1
    nxt.pos_hist.append((pose[0], pose[-1]))
    nxt.pos_hist = nxt.pos_hist[-DISP_WINDOW:]
    if len(nxt.pos_hist) >= DISP_WINDOW:
        x0, z0 = nxt.pos_hist[0]
        x1, z1 = nxt.pos_hist[-1]
        nxt.stuck = math.hypot(x1 - x0, z1 - z0) < 1.0
    else:
        nxt.stuck = False
    return _apply_observation(nxt, obs)


def update(
    state: TrackerState,
    pos,
    yaw: float,
    executed_action,
    observation: TargetObs | dict | None,
    max_steps: int | None = None,
    noise: TrackerNoise | None = None,
    rng=None,
) -> TrackerState:
    """Advance S to the pose after executed_action. A list of names is applied one by one."""
    import random as _random

    obs = observation if isinstance(observation, TargetObs) else TargetObs.from_dict(observation)
    if isinstance(executed_action, (list, tuple)):
        names = [str(n) for n in executed_action if n is not None]
    elif executed_action is None or executed_action == "":
        names = []
    else:
        names = [str(executed_action)]
    nxt = _clone(state)
    if max_steps is not None:
        nxt.max_steps = int(max_steps)
    rng = rng or _random.Random(0)
    pose = [float(v) for v in pos]
    yaw_v = float(yaw)
    if noise is not None:
        pose, yaw_v = noise.apply_pose(pose, yaw_v, rng)
        obs = noise.apply_obs(obs, rng)
    if not names:
        return _step_one(nxt, pose, yaw_v, None, obs)
    last = len(names) - 1
    for i, name in enumerate(names):
        use = obs if i == last else TargetObs(visible=False)
        nxt = _step_one(nxt, pose, yaw_v, name, use)
    return nxt


def _apply_observation(state: TrackerState, obs: TargetObs) -> TrackerState:
    if obs.visible and obs.bearing_deg is not None:
        rad = math.radians(wrap_deg(obs.bearing_deg))
        state.belief_sin = math.sin(rad)
        state.belief_cos = math.cos(rad)
        if obs.log_distance is not None:
            state.log_distance = float(obs.log_distance)
        state.unseen_steps = 0.0
        state.confidence = 1.0
        return state
    state.unseen_steps = min(float(state.unseen_steps) + 1.0, UNSEEN_HORIZON)
    state.confidence = math.exp(-state.unseen_steps / UNSEEN_HORIZON)
    return state


def observation_from_step(step: dict) -> TargetObs:
    if step.get("target_obs"):
        return TargetObs.from_dict(step["target_obs"])
    match = step.get("match") or {}
    visible = bool(match.get("target_direction") or (match.get("satisfied") or {}).get("target"))
    dist = step.get("target_distance_num")
    if not visible:
        return TargetObs(visible=False)
    bearing = 0.0
    if match.get("target_direction") == "left":
        bearing = -30.0
    elif match.get("target_direction") == "right":
        bearing = 30.0
    log_d = math.log(max(float(dist), 0.05)) if dist is not None else LOG_DIST_UNSEEN
    return TargetObs(visible=True, bearing_deg=bearing, log_distance=log_d)


def replay(steps: list[dict], max_steps: int = 3000, noise: TrackerNoise | None = None, rng=None) -> list[TrackerState]:
    """S at each decision: current pose and observation, after the previous chunk's actions."""
    state = TrackerState(max_steps=max_steps)
    out = []
    prev_actions: list[str] = []
    for i, step in enumerate(steps):
        pos = step.get("pos") or [0.0, 0.0, 0.0]
        yaw = float(step.get("yaw") or 0.0)
        obs = observation_from_step(step)
        action = [] if i == 0 else prev_actions
        state = update(state, pos, yaw, action, obs, max_steps=max_steps, noise=noise, rng=rng)
        out.append(state)
        prev_actions = list(step.get("executed_actions") or step.get("action_chunk") or [])
    return out


def flip_features(feat: dict) -> dict:
    sectors = list(feat.get("sectors") or [0.0] * N_SECTORS)
    mirrored = [sectors[0]] + [sectors[N_SECTORS - i] for i in range(1, N_SECTORS)]
    return {
        **feat,
        "bearing_sin": -float(feat.get("bearing_sin") or 0.0),
        "turn_since_forward": -float(feat.get("turn_since_forward") or 0.0),
        "sectors": mirrored,
    }


def flip_vector(vec: list[float]) -> list[float]:
    out = list(vec)
    out[0] = -out[0]
    out[6] = -out[6]
    sectors = out[7:15]
    out[7:15] = [sectors[0]] + [sectors[N_SECTORS - i] for i in range(1, N_SECTORS)]
    return out
