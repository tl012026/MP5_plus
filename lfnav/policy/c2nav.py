"""C2Nav: patch-wise A heads plus a mode/waypoint/stop policy C."""

from __future__ import annotations

import math

import numpy as np

from lfnav.geom import ray_directions
from lfnav.gt.fov import ray_angles_deg
from lfnav.policy.c2nav_labels import FORWARD_BINS, MODE_NAMES, N_TURN
from lfnav.policy.state_tracker import TrackerState

VERSION = "c2nav_v1"
CHANNELS = 64
S_DIM = 18
CKPT_STRUCTURE = (
    "version",
    "hidden",
    "channels",
    "dim",
    "grid",
    "token_pool",
    "objects",
    "biomes",
    "times",
    "weathers",
    "brightness",
    "modes",
    "n_turn",
    "forward_bins",
    "chunk_max_steps",
    "dropout",
    "s_dim",
    "cond_dim",
)


def spec_from_cfg(cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    labels = cfg.get("labels") or {}
    nav = cfg.get("c2nav") or {}
    objects = list(labels.get("objects") or ["tree", "grass", "water", "pig", "cow", "sheep", "chicken", "sand"])
    return {
        "objects": objects,
        "biomes": list(labels.get("biomes") or ["forest", "plains", "desert", "savanna", "taiga", "jungle", "other"]),
        "times": list(labels.get("time") or ["day", "night"]),
        "weathers": list(labels.get("weather") or ["clear", "rain"]),
        "brightness": list(labels.get("brightness") or ["sufficient", "insufficient"]),
        "hidden": int(nav.get("hidden") or 256),
        "channels": int(nav.get("channels") or CHANNELS),
        "dim": int(nav.get("dim") or 384),
        "grid": tuple(nav.get("grid") or (8, 13)),
        "token_pool": int(nav.get("token_pool") or ((cfg.get("joint") or {}).get("token_pool") or 2)),
        "dropout": float(nav.get("dropout") or 0.3),
        "chunk_max_steps": int(nav.get("chunk_max_steps") or 8),
        "modes": list(MODE_NAMES),
        "n_turn": int(N_TURN),
        "forward_bins": list(FORWARD_BINS),
        "s_dim": S_DIM,
        "cond_dim": 1 + len(objects) + 4,
        "version": str(nav.get("version") or VERSION),
    }


def patch_bearings(n_h: int, n_w: int, cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or {}
    dirs = ray_directions(n_h, n_w, float(cfg.get("fov_deg") or 70), str(cfg.get("fov_axis") or "vertical"))
    yaw = np.zeros((n_h, n_w), dtype=np.float32)
    for i in range(n_h):
        for j in range(n_w):
            yaw[i, j] = ray_angles_deg(dirs[i][j])[0]
    return yaw


def encode_match(match: dict | None, objects: list[str]) -> np.ndarray:
    match = match or {}
    sat = match.get("satisfied") or {}
    vec = [1.0 if sat.get("target") else 0.0]
    for name in objects:
        vec.append(1.0 if sat.get(f"near:{name}") else 0.0)
    for key in ("biome", "time", "weather", "brightness"):
        hits = [v for k, v in sat.items() if str(k).startswith(f"{key}:") or str(k).startswith(f"{key}")]
        if hits:
            vec.append(1.0 if all(hits) else 0.0)
        else:
            missing = [m for m in (match.get("missing") or []) if str(m).startswith(key)]
            vec.append(0.0 if missing else 1.0)
    return np.asarray(vec, dtype=np.float32)


def encode_s(state: TrackerState | dict | list[float]) -> np.ndarray:
    if isinstance(state, TrackerState):
        return np.asarray(state.vector(), dtype=np.float32)
    if isinstance(state, dict):
        return np.asarray(TrackerState.from_dict(state).vector(), dtype=np.float32)
    return np.asarray(state, dtype=np.float32).reshape(-1)


def target_geometry(present_p, log_dist, target_index: int, bearings: np.ndarray):
    """present_p, log_dist: (B, K, H, W). Returns vis, sin, cos, log_d each (B,)."""
    import torch

    p = present_p[:, int(target_index)].clamp(0.0, 1.0)
    vis = 1.0 - (1.0 - p).clamp(0.0, 1.0).prod(dim=(1, 2))
    w = p
    mass = w.sum(dim=(1, 2)).clamp_min(1e-6)
    yaw = torch.as_tensor(bearings, device=p.device, dtype=p.dtype)
    yaw_hat = (w * yaw).sum(dim=(1, 2)) / mass
    log_d = (w * log_dist[:, int(target_index)]).sum(dim=(1, 2)) / mass
    unseen = vis < 1e-4
    log_d = torch.where(unseen, torch.full_like(log_d, math.log(32.0)), log_d)
    yaw_hat = torch.where(unseen, torch.zeros_like(yaw_hat), yaw_hat)
    rad = yaw_hat * math.pi / 180.0
    return vis, torch.sin(rad), torch.cos(rad), log_d


def assert_structure(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("c2nav checkpoint must be a dict")
    missing = [key for key in CKPT_STRUCTURE if key not in payload]
    if missing:
        raise RuntimeError("checkpoint missing " + ", ".join(missing))
    return payload


def C2NavNet(spec: dict):
    import torch
    from torch import nn

    n_obj = len(spec["objects"])
    ch = int(spec["channels"])
    dim = int(spec["dim"])
    hidden = int(spec["hidden"])
    drop = float(spec["dropout"])
    n_mode = len(spec["modes"])
    n_turn = int(spec["n_turn"])
    n_fwd = len(spec["forward_bins"])
    cond_dim = int(spec["cond_dim"])
    s_dim = int(spec["s_dim"])
    c_in = 4 + s_dim + cond_dim

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec
            self.stem = nn.Conv2d(dim, ch, kernel_size=1)
            self.present = nn.Conv2d(ch, n_obj, kernel_size=1)
            self.dist_conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
            self.log_dist = nn.Conv2d(ch, n_obj, kernel_size=1)
            pooled = ch
            self.biome = nn.Linear(pooled, len(spec["biomes"]))
            self.time = nn.Linear(pooled, len(spec["times"]))
            self.weather = nn.Linear(pooled, len(spec["weathers"]))
            self.brightness = nn.Linear(pooled, len(spec["brightness"]))
            self.c1 = nn.Linear(c_in, hidden)
            self.c2 = nn.Linear(hidden, hidden)
            self.drop = nn.Dropout(drop)
            self.mode = nn.Linear(hidden, n_mode)
            self.turn = nn.Linear(hidden, n_turn)
            self.forward = nn.Linear(hidden, n_fwd)
            self.stop = nn.Linear(hidden, 1)

        def perception(self, tokens):
            x = tokens.permute(0, 3, 1, 2)
            h = nn.functional.relu(self.stem(x))
            present_logit = self.present(h)
            log_d = self.log_dist(nn.functional.relu(self.dist_conv(h)))
            pooled = h.mean(dim=(2, 3))
            return {
                "present_logit": present_logit,
                "present_p": torch.sigmoid(present_logit),
                "log_dist": log_d,
                "biome": self.biome(pooled),
                "time": self.time(pooled),
                "weather": self.weather(pooled),
                "brightness": self.brightness(pooled),
                "feat": h,
            }

        def policy(self, geom, s_vec, cond):
            x = torch.cat([geom, s_vec, cond], dim=-1)
            h = nn.functional.relu(self.c1(x))
            h = self.drop(h)
            h = nn.functional.relu(self.c2(h))
            h = self.drop(h)
            return {
                "mode": self.mode(h),
                "turn": self.turn(h),
                "forward": self.forward(h),
                "stop": self.stop(h).squeeze(-1),
            }

        def forward(self, tokens, s_vec, cond, target_index, bearings):
            a = self.perception(tokens)
            vis, sn, cs, log_d = target_geometry(a["present_p"], a["log_dist"], int(target_index), bearings)
            geom = torch.stack([vis, sn, cs, log_d], dim=-1)
            c = self.policy(geom, s_vec, cond)
            out = dict(a)
            out.update(c)
            out["geom"] = geom
            out["vis"] = vis
            return out

    return _Net()


def structure_record(spec: dict, **extra) -> dict:
    payload = {key: spec[key] for key in CKPT_STRUCTURE if key in spec}
    payload["grid"] = list(spec["grid"])
    payload.update(extra)
    missing = [key for key in CKPT_STRUCTURE if key not in payload]
    if missing:
        raise RuntimeError("checkpoint must include " + ", ".join(missing))
    return payload


def load_net(payload: dict, map_location="cpu"):
    import torch

    payload = assert_structure(payload)
    spec = {key: payload[key] for key in CKPT_STRUCTURE}
    spec["grid"] = tuple(spec["grid"])
    spec["forward_bins"] = list(spec["forward_bins"])
    net = C2NavNet(spec)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint missing model")
    net.load_state_dict(state)
    net.eval()
    return net, spec
