"""C2Nav: patch-wise A heads plus a mode/waypoint/stop policy C."""

from __future__ import annotations

import math

import numpy as np

from lfnav.geom import ray_directions
from lfnav.gt.fov import ray_angles_deg
from lfnav.policy.base import Policy
from lfnav.policy.c2nav_labels import FORWARD_BINS, MODE_NAMES, N_TURN
from lfnav.policy.state_tracker import STEPS_NORM, TrackerState

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


FORWARD_KEYS = ("tokens", "s_vec", "cond", "target_index", "bearings")


def target_geometry(present_p, log_dist, target_index, bearings):
    """present_p, log_dist: (B, K, H, W). target_index: int or (B,). Returns vis, sin, cos, log_d each (B,)."""
    import torch

    b, _k, height, width = present_p.shape
    idx = target_index
    if not torch.is_tensor(idx):
        idx = torch.as_tensor(idx, device=present_p.device, dtype=torch.long)
    else:
        idx = idx.to(device=present_p.device, dtype=torch.long)
    if idx.ndim == 0:
        p = present_p[:, int(idx)].clamp(0.0, 1.0)
        dist = log_dist[:, int(idx)]
    else:
        idx = idx.reshape(-1)
        if idx.numel() == 1 and b > 1:
            idx = idx.expand(b)
        gather = idx.view(b, 1, 1, 1).expand(b, 1, height, width)
        p = present_p.gather(1, gather).squeeze(1).clamp(0.0, 1.0)
        dist = log_dist.gather(1, gather).squeeze(1)
    vis = 1.0 - (1.0 - p).clamp(0.0, 1.0).flatten(1).prod(dim=1)
    mass = p.flatten(1).sum(dim=1).clamp_min(1e-6)
    yaw = torch.as_tensor(bearings, device=p.device, dtype=p.dtype)
    yaw_hat = (p * yaw).flatten(1).sum(dim=1) / mass
    log_d = (p * dist).flatten(1).sum(dim=1) / mass
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
            self.fwd = nn.Linear(hidden, n_fwd)
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
                "forward": self.fwd(h),
                "stop": self.stop(h).squeeze(-1),
            }

        def forward(self, batch):
            missing = [key for key in FORWARD_KEYS if key not in batch]
            if missing:
                raise RuntimeError("c2nav batch missing " + ", ".join(missing))
            a = self.perception(batch["tokens"])
            vis, sn, cs, log_d = target_geometry(
                a["present_p"], a["log_dist"], batch["target_index"], batch["bearings"],
            )
            geom_from_a = torch.stack([vis, sn, cs, log_d], dim=-1)
            if self.training and batch.get("geom") is not None:
                geom = batch["geom"]
                vis_out = geom[:, 0]
            else:
                geom = geom_from_a
                vis_out = vis
            c = self.policy(geom, batch["s_vec"], batch["cond"])
            out = dict(a)
            out.update(c)
            out["geom"] = geom
            out["vis"] = vis_out
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


def _scene_from_present(present_p, log_dist, spec: dict, bearings: np.ndarray):
    import torch

    from lfnav.aggregate import distance_bin
    from lfnav.types import ObjectBelief, SceneState

    objects = {}
    t_p = torch.from_numpy(np.asarray(present_p, np.float32)).unsqueeze(0)
    t_d = torch.from_numpy(np.asarray(log_dist, np.float32)).unsqueeze(0)
    for i, name in enumerate(spec["objects"]):
        pmap = present_p[i]
        p = float(np.clip(pmap.max() if pmap.size else 0.0, 0.0, 1.0))
        vis, sn, cs, log_d = target_geometry(t_p, t_d, i, bearings)
        yaw = math.degrees(math.atan2(float(sn), float(cs))) if float(vis) > 0.05 else 0.0
        if yaw < -20:
            direction = {"left": 1.0, "center": 0.0, "right": 0.0}
        elif yaw > 20:
            direction = {"left": 0.0, "center": 0.0, "right": 1.0}
        else:
            direction = {"left": 0.0, "center": 1.0, "right": 0.0}
        dist_m = math.exp(float(log_d))
        name_bin = distance_bin(dist_m)
        distance = {key: 1.0 if key == name_bin else 0.0 for key in ("near", "mid", "far")}
        objects[name] = ObjectBelief(present=p, direction=direction, distance=distance)
    return SceneState(objects=objects)


class C2NavAgent(Policy):
    name = "c2nav"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        nav = self.cfg.get("c2nav") or {}
        self.tau = float(nav.get("stop_tau") or 0.5)
        self.chunk_max = int(nav.get("chunk_max_steps") or 8)
        self._tokens = None
        self._scene = None
        self._last_a = None
        self._s = TrackerState(max_steps=STEPS_NORM)
        self._prev_actions: list[str] = []
        self.last_info: dict = {}
        self.model = None
        self.spec = spec_from_cfg(self.cfg)
        self.bearings = patch_bearings(self.spec["grid"][0], self.spec["grid"][1], self.cfg)
        try:
            import torch
        except ImportError:
            torch = None
        self._torch = torch
        ckpt = nav.get("checkpoint")
        payload = None
        if ckpt:
            if torch is None:
                raise RuntimeError("c2nav checkpoint requires torch")
            payload = torch.load(ckpt, map_location="cpu")
            self.model, self.spec = load_net(payload)
            meta = payload.get("meta") or {}
            if meta.get("chosen_tau") is not None:
                self.tau = float(meta["chosen_tau"])
            self.bearings = patch_bearings(self.spec["grid"][0], self.spec["grid"][1], self.cfg)
        from lfnav.perception.encoder import FakeEncoder, load_encoder

        enc_name = nav.get("encoder") or ((self.cfg.get("perception") or {}).get("encoder") or "fake")
        if self.cfg.get("dry_run") or enc_name in ("fake", "dry-run") or torch is None:
            self.encoder = FakeEncoder(dim=int(self.spec["dim"]), grid=tuple(self.spec["grid"]))
        else:
            self.encoder = load_encoder(enc_name, self.cfg)
        if torch is not None and self.model is None:
            self.model = C2NavNet(self.spec)
            self.model.eval()
        self.reset()

    def reset(self) -> None:
        self._tokens = None
        self._scene = None
        self._last_a = None
        self._s = TrackerState(max_steps=STEPS_NORM)
        self._prev_actions = []
        self.last_info = {}

    def features(self, frame, force: bool = False) -> dict:
        from lfnav.data.demo_format import preprocess_frame
        from lfnav.perception.joint_features import encode_joint_tokens

        if not force and self._tokens is not None:
            return {"tokens": self._tokens}
        rgb = frame
        if isinstance(frame, dict):
            from lfnav.env.frames import rgb_hwc

            rgb = rgb_hwc(frame, self.cfg)
            if rgb is None:
                rgb = frame.get("rgb")
        rgb = preprocess_frame(rgb)
        encoded = encode_joint_tokens(rgb, self.encoder, int(self.spec["token_pool"]))
        self._tokens = encoded["tokens"]
        return encoded

    def numeric_distance(self, target: str):
        if self._last_a is None or self._torch is None:
            return None
        objects = list(self.spec["objects"])
        if target not in objects:
            return None
        vis, _sn, _cs, log_d = target_geometry(
            self._last_a["present_p"], self._last_a["log_dist"], objects.index(target), self.bearings,
        )
        if float(vis) < 0.5:
            return None
        return math.exp(float(log_d))

    def perceive(self, frame):
        from lfnav.types import SceneState

        self.features(frame, force=True)
        if self.model is None or self._torch is None or self._tokens is None:
            self._scene = SceneState()
            return self._scene
        torch = self._torch
        with torch.no_grad():
            tokens = torch.from_numpy(np.asarray(self._tokens, np.float32)).unsqueeze(0)
            a = self.model.perception(tokens)
        self._last_a = a
        self._scene = _scene_from_present(
            a["present_p"][0].cpu().numpy(),
            a["log_dist"][0].cpu().numpy(),
            self.spec,
            self.bearings,
        )
        return self._scene

    def act(self, policy_input: PolicyInput) -> PolicyOutput:
        from lfnav.policy.c2nav_labels import FORWARD_BINS, MODE_NAMES, invert_waypoint, turn_center
        from lfnav.policy.state_tracker import TargetObs, update
        from lfnav.types import PolicyOutput

        mem = policy_input.memory or {}
        pos = mem.get("pos") or [0.0, 0.0, 0.0]
        yaw = float(mem.get("yaw") or 0.0)
        match = policy_input.match or {}
        target = str((policy_input.task or {}).get("target") or "")
        objects = list(self.spec["objects"])
        t_i = objects.index(target) if target in objects else 0
        obs = TargetObs(visible=False)
        geom_np = np.zeros(4, np.float32)
        mode_i = turn_i = fwd_i = 0
        p_stop = 0.0
        if self.model is not None and self._torch is not None and self._tokens is not None:
            torch = self._torch
            device = next(self.model.parameters()).device
            with torch.no_grad():
                batch = {
                    "tokens": torch.from_numpy(np.asarray(self._tokens, np.float32)).unsqueeze(0).to(device),
                    "s_vec": torch.from_numpy(encode_s(self._s)).unsqueeze(0).to(device),
                    "cond": torch.from_numpy(encode_match(match, objects)).unsqueeze(0).to(device),
                    "target_index": torch.tensor([t_i], device=device, dtype=torch.long),
                    "bearings": torch.as_tensor(self.bearings, device=device, dtype=torch.float32),
                }
                out = self.model(batch)
            vis_f = float(out["vis"][0])
            sn = float(out["geom"][0, 1])
            cs = float(out["geom"][0, 2])
            log_d = float(out["geom"][0, 3])
            obs = TargetObs(
                visible=vis_f >= 0.5,
                bearing_deg=math.degrees(math.atan2(sn, cs)),
                log_distance=log_d,
            )
            geom_np = np.asarray([vis_f, sn, cs, log_d], np.float32)
            mode_i = int(out["mode"][0].argmax().item())
            turn_i = int(out["turn"][0].argmax().item())
            fwd_i = int(out["forward"][0].argmax().item())
            p_stop = float(torch.sigmoid(out["stop"][0]).item())
        self._s = update(self._s, pos, yaw, self._prev_actions, obs, max_steps=self._s.max_steps)
        feat = self._s.features()
        info = {
            "mode": MODE_NAMES[mode_i] if 0 <= mode_i < len(MODE_NAMES) else str(mode_i),
            "waypoint": {"turn_deg": turn_center(turn_i), "forward": FORWARD_BINS[fwd_i] if 0 <= fwd_i < len(FORWARD_BINS) else 0},
            "p_stop": p_stop,
            "belief": {
                "bearing_sin": feat["bearing_sin"],
                "bearing_cos": feat["bearing_cos"],
                "log_distance": feat["log_distance"],
                "confidence": feat["confidence"],
            },
            "tau": self.tau,
        }
        self.last_info = info
        if p_stop > self.tau:
            return PolicyOutput(stop=True, info=info)
        dyaw = turn_center(turn_i)
        forward = FORWARD_BINS[fwd_i] if 0 <= fwd_i < len(FORWARD_BINS) else 0
        actions = invert_waypoint(turn_i, fwd_i)[: self.chunk_max]
        self._prev_actions = list(actions)
        info["action_chunk"] = actions
        if forward < 0:
            return PolicyOutput(action_chunk=actions, info=info)
        return PolicyOutput(waypoints=[(dyaw, forward)], info=info)
