"""Joint A (perception) + C2 (chunk policy) on frozen DINOv2 patch features."""

from __future__ import annotations

import numpy as np

from lfnav.aggregate import argmax_name
from lfnav.policy.base import Policy
from lfnav.policy.c2_data import DIRECTIONS, DISTANCES, FeatureSpec, encode_aux
from lfnav.policy.c2 import _visible
from lfnav.types import ObjectBelief, PolicyInput, PolicyOutput, SceneState

IGNORE = -100


def joint_spec(cfg: dict | None = None) -> FeatureSpec:
    spec = FeatureSpec.from_cfg(cfg)
    spec.recent_k = int(((cfg or {}).get("joint") or {}).get("recent_k") or 4)
    spec.horizon = int(((cfg or {}).get("joint") or {}).get("chunk") or spec.horizon)
    return spec


def scene_targets(scene: dict, spec: FeatureSpec) -> dict:
    present, direction, distance = [], [], []
    for name in spec.objects:
        raw = ((scene or {}).get("objects") or {}).get(name) or {}
        if not isinstance(raw, dict):
            raw = {}
        p = 1 if float(raw.get("present") or 0.0) >= 0.5 else 0
        present.append(p)
        if p:
            direction.append(_argmax_index(raw.get("direction"), DIRECTIONS))
            distance.append(_argmax_index(raw.get("distance"), DISTANCES))
        else:
            direction.append(IGNORE)
            distance.append(IGNORE)
    return {
        "present": np.asarray(present, dtype=np.int64),
        "direction": np.asarray(direction, dtype=np.int64),
        "distance": np.asarray(distance, dtype=np.int64),
        "biome": np.int64(_argmax_index((scene or {}).get("biome"), spec.biomes, empty=0)),
        "time": np.int64(_argmax_index((scene or {}).get("time"), spec.times, empty=0)),
        "weather": np.int64(_argmax_index((scene or {}).get("weather"), spec.weathers, empty=0)),
        "brightness": np.int64(_argmax_index((scene or {}).get("brightness"), spec.brightness, empty=0)),
    }


def _argmax_index(table, names, empty=IGNORE) -> int:
    if not names:
        return empty
    vals = [float((table or {}).get(n) or 0.0) for n in names]
    if max(vals) <= 0.0:
        return empty
    return int(np.argmax(vals))


def scene_from_heads(present_p, dir_p, dist_p, biome_p, time_p, weather_p, bright_p, spec: FeatureSpec) -> SceneState:
    objects = {}
    for i, name in enumerate(spec.objects):
        objects[name] = ObjectBelief(
            present=float(present_p[i]),
            direction={k: float(dir_p[i, j]) for j, k in enumerate(DIRECTIONS)},
            distance={k: float(dist_p[i, j]) for j, k in enumerate(DISTANCES)},
        )
    return SceneState(
        objects=objects,
        biome={k: float(biome_p[j]) for j, k in enumerate(spec.biomes)},
        time={k: float(time_p[j]) for j, k in enumerate(spec.times)},
        weather={k: float(weather_p[j]) for j, k in enumerate(spec.weathers)},
        brightness={k: float(bright_p[j]) for j, k in enumerate(spec.brightness)},
    )


def belief_aligned_distance(scene: SceneState, target: str, cfg: dict | None = None) -> float | None:
    """Numeric stand-in for B's nearest-ray stop, from A's distance classes only."""
    obj = scene.object(target)
    tau = float(((cfg or {}).get("match") or {}).get("tau_obj", 0.5))
    if float(obj.present) < tau:
        return None
    name = argmax_name(obj.distance)
    bins = (cfg or {}).get("distance_bins") or {}
    near = float(bins.get("near") or 3.0)
    mid = float(bins.get("mid") or 10.0)
    if name == "near":
        return near * 0.5
    if name == "mid":
        return (near + mid) / 2.0
    if name == "far":
        return mid + 5.0
    return None


def scene_vector_np(present_p, dir_p, dist_p, biome_p, time_p, weather_p, bright_p) -> np.ndarray:
    parts = [np.asarray(present_p, dtype=np.float32).reshape(-1)]
    parts.append(np.asarray(dir_p, dtype=np.float32).reshape(-1))
    parts.append(np.asarray(dist_p, dtype=np.float32).reshape(-1))
    for row in (biome_p, time_p, weather_p, bright_p):
        parts.append(np.asarray(row, dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)


def JointNet(dim: int, spec: FeatureSpec, hidden: int = 256, horizon: int | None = None):
    import torch
    from torch import nn

    n_obj = len(spec.objects)
    n_act = len(spec.actions)
    k = int(horizon or spec.horizon)
    scene_dim = spec.scene_dim
    in_dim = int(dim) + scene_dim + spec.aux_dim

    class _Joint(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = int(dim)
            self.hidden = int(hidden)
            self.horizon = k
            self.n_actions = n_act
            self.spec_objects = n_obj
            self.conv = nn.Conv2d(int(dim), int(hidden), kernel_size=3, padding=1)
            self.fc = nn.Linear(int(hidden), int(hidden))
            self.present = nn.Linear(int(hidden), n_obj * 2)
            self.direction = nn.Linear(int(hidden), n_obj * 3)
            self.distance = nn.Linear(int(hidden), n_obj * 3)
            self.biome = nn.Linear(int(hidden), len(spec.biomes))
            self.time = nn.Linear(int(hidden), len(spec.times))
            self.weather = nn.Linear(int(hidden), len(spec.weathers))
            self.brightness = nn.Linear(int(hidden), len(spec.brightness))
            self.p1 = nn.Linear(in_dim, int(hidden))
            self.p2 = nn.Linear(int(hidden), int(hidden))
            self.heads = nn.ModuleList(nn.Linear(int(hidden), n_act) for _ in range(k))

        def perception(self, tokens):
            # tokens: (B, H, W, D)
            x = tokens.permute(0, 3, 1, 2)
            h = nn.functional.relu(self.conv(x))
            h = h.mean(dim=(2, 3))
            h = nn.functional.relu(self.fc(h))
            b = tokens.shape[0]
            present = self.present(h).view(b, n_obj, 2)
            direction = self.direction(h).view(b, n_obj, 3)
            distance = self.distance(h).view(b, n_obj, 3)
            biome = self.biome(h)
            time = self.time(h)
            weather = self.weather(h)
            brightness = self.brightness(h)
            present_p = nn.functional.softmax(present, dim=-1)[..., 1]
            dir_p = nn.functional.softmax(direction, dim=-1)
            dist_p = nn.functional.softmax(distance, dim=-1)
            biome_p = nn.functional.softmax(biome, dim=-1)
            time_p = nn.functional.softmax(time, dim=-1)
            weather_p = nn.functional.softmax(weather, dim=-1)
            bright_p = nn.functional.softmax(brightness, dim=-1)
            scene_vec = torch.cat(
                [
                    present_p.reshape(b, -1),
                    dir_p.reshape(b, -1),
                    dist_p.reshape(b, -1),
                    biome_p,
                    time_p,
                    weather_p,
                    bright_p,
                ],
                dim=-1,
            )
            return {
                "present": present,
                "direction": direction,
                "distance": distance,
                "biome": biome,
                "time": time,
                "weather": weather,
                "brightness": brightness,
                "present_p": present_p,
                "dir_p": dir_p,
                "dist_p": dist_p,
                "biome_p": biome_p,
                "time_p": time_p,
                "weather_p": weather_p,
                "bright_p": bright_p,
                "scene_vec": scene_vec,
            }

        def policy(self, pooled, scene_vec, aux):
            x = torch.cat([pooled, scene_vec, aux], dim=-1)
            h = nn.functional.relu(self.p1(x))
            h = nn.functional.relu(self.p2(h))
            return [head(h) for head in self.heads]

        def forward(self, tokens, aux):
            a = self.perception(tokens)
            pooled = tokens.mean(dim=(1, 2))
            logits = self.policy(pooled, a["scene_vec"], aux)
            a["action"] = logits
            a["pooled"] = pooled
            return a

    return _Joint()


def joint_loss(out, batch, class_w, lam: float):
    import torch
    from torch.nn import functional as F

    present = out["present"].reshape(-1, 2)
    y_p = batch["present"].reshape(-1)
    loss_p = F.cross_entropy(present, y_p)
    loss_dir = F.cross_entropy(out["direction"].reshape(-1, 3), batch["direction"].reshape(-1), ignore_index=IGNORE)
    loss_dist = F.cross_entropy(out["distance"].reshape(-1, 3), batch["distance"].reshape(-1), ignore_index=IGNORE)
    loss_biome = F.cross_entropy(out["biome"], batch["biome"])
    loss_time = F.cross_entropy(out["time"], batch["time"])
    loss_weather = F.cross_entropy(out["weather"], batch["weather"])
    loss_bright = F.cross_entropy(out["brightness"], batch["brightness"])
    loss_a = loss_p + loss_dir + loss_dist + loss_biome + loss_time + loss_weather + loss_bright
    total = out["present"].new_zeros(out["present"].shape[0])
    for k, head in enumerate(out["action"]):
        total = total + F.cross_entropy(head, batch["y"][:, k], weight=class_w, reduction="none")
    loss_c2 = total.mean()
    return loss_a + float(lam) * loss_c2, {
        "a": float(loss_a.detach().item()),
        "c2": float(loss_c2.detach().item()),
        "present": loss_p,
        "direction": loss_dir,
        "distance": loss_dist,
    }


def joint_val_acc(out, batch) -> dict:
    import torch

    present_hat = out["present"].argmax(dim=-1)
    present_ok = (present_hat == batch["present"]).float().mean()
    dir_mask = batch["direction"] != IGNORE
    dist_mask = batch["distance"] != IGNORE
    dir_ok = ((out["direction"].argmax(dim=-1) == batch["direction"]) & dir_mask).float().sum() / dir_mask.float().sum().clamp_min(1.0)
    dist_ok = ((out["distance"].argmax(dim=-1) == batch["distance"]) & dist_mask).float().sum() / dist_mask.float().sum().clamp_min(1.0)
    act = []
    for k, head in enumerate(out["action"]):
        act.append(float((head.argmax(dim=-1) == batch["y"][:, k]).float().mean().item()))
    return {
        "present": float(present_ok.item()),
        "direction": float(dir_ok.item()),
        "distance": float(dist_ok.item()),
        "action": float(sum(act) / max(len(act), 1)),
    }


class JointAgent(Policy):
    name = "joint"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        joint = self.cfg.get("joint") or {}
        self.spec = joint_spec(self.cfg)
        self.hidden = int(joint.get("hidden") or 256)
        self.lambda_c2 = float(joint.get("lambda_c2", 1.0))
        self._torch = None
        self.model = None
        self.encoder = None
        self.last_seen = None
        self._tokens = None
        self._pooled = None
        self._scene = SceneState()
        try:
            import torch
        except ImportError:
            torch = None
        self._torch = torch
        from lfnav.perception.encoder import FakeEncoder, load_encoder

        enc_name = joint.get("encoder") or ((self.cfg.get("perception") or {}).get("encoder") or "fake")
        if self.cfg.get("dry_run") or enc_name in ("fake", "dry-run") or torch is None:
            dim = int(joint.get("fake_dim") or 32)
            grid = tuple(joint.get("fake_grid") or (2, 3))
            self.encoder = FakeEncoder(dim=dim, grid=grid)
        else:
            self.encoder = load_encoder(enc_name, self.cfg)
        hw = self.cfg.get("image_hw") or [224, 364]
        feats = self.encoder(np.zeros((int(hw[0]), int(hw[1]), 3), dtype=np.uint8))
        self.dim = int(np.asarray(feats["tokens"]).shape[-1])
        if torch is not None:
            self.model = JointNet(self.dim, self.spec, self.hidden, self.spec.horizon)
            ckpt = joint.get("checkpoint")
            if ckpt:
                payload = torch.load(ckpt, map_location="cpu")
                state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
                self.model.load_state_dict(state)
            self.model.eval()
        self.reset()

    def reset(self) -> None:
        self.last_seen = None
        self._tokens = None
        self._pooled = None
        self._scene = SceneState()

    def n_parameters(self) -> int:
        if self.model is None:
            return 0
        return int(sum(p.numel() for p in self.model.parameters()))

    def features(self, frame) -> dict:
        rgb = frame
        if isinstance(frame, dict):
            from lfnav.env.frames import rgb_hwc

            rgb = rgb_hwc(frame, self.cfg)
            if rgb is None:
                rgb = frame.get("rgb")
        encoded = self.encoder(rgb)
        tokens = np.asarray(encoded["tokens"], dtype=np.float32)
        pooled = np.asarray(encoded.get("pooled"), dtype=np.float32)
        if pooled.ndim == 0 or pooled.size == 0:
            pooled = tokens.mean(axis=(0, 1))
        self._tokens = tokens
        self._pooled = pooled
        return encoded

    def perceive(self, frame) -> SceneState:
        self.features(frame)
        if self.model is None or self._torch is None:
            self._scene = _untrained_scene(self._tokens, self.spec)
            return self._scene
        torch = self._torch
        with torch.no_grad():
            tokens = torch.from_numpy(self._tokens).unsqueeze(0)
            a = self.model.perception(tokens)
        self._scene = scene_from_heads(
            a["present_p"][0].cpu().numpy(),
            a["dir_p"][0].cpu().numpy(),
            a["dist_p"][0].cpu().numpy(),
            a["biome_p"][0].cpu().numpy(),
            a["time_p"][0].cpu().numpy(),
            a["weather_p"][0].cpu().numpy(),
            a["bright_p"][0].cpu().numpy(),
            self.spec,
        )
        return self._scene

    def act(self, policy_input: PolicyInput) -> PolicyOutput:
        match = policy_input.match or {}
        env_step = int((policy_input.memory or {}).get("env_steps") or 0)
        if _visible(match):
            self.last_seen = {
                "step": env_step,
                "direction": match.get("target_direction"),
                "distance": match.get("target_distance"),
            }
        mem = policy_input.memory or {}
        sectors = mem.get("sector8")
        if sectors is None:
            from lfnav.policy.c1 import sector8_counts

            sectors = sector8_counts(mem.get("local_grid") or [])
        aux = encode_aux(
            policy_input.task or {},
            sectors,
            policy_input.recent_actions or mem.get("recent_actions") or [],
            self.last_seen if self.last_seen is not None else mem.get("last_seen"),
            env_step,
            self.spec,
        )
        if self.model is None or self._torch is None or self._tokens is None:
            return PolicyOutput(["noop"] * self.spec.horizon)
        torch = self._torch
        with torch.no_grad():
            tokens = torch.from_numpy(np.asarray(self._tokens, dtype=np.float32)).unsqueeze(0)
            aux_t = torch.from_numpy(aux).unsqueeze(0)
            out = self.model(tokens, aux_t)
            idx = [int(row.argmax(dim=-1).item()) for row in out["action"]]
        chunk = [self.spec.actions[i] for i in idx]
        return PolicyOutput(chunk)


def _untrained_scene(tokens, spec: FeatureSpec) -> SceneState:
    raw = np.asarray(tokens)
    present = float(np.clip(raw[..., 0].mean() if raw.size else 0.0, 0.0, 1.0))
    objects = {
        name: ObjectBelief(
            present=present,
            direction={"left": 0.0, "center": 1.0, "right": 0.0},
            distance={"near": 0.0, "mid": 1.0, "far": 0.0},
        )
        for name in spec.objects
    }
    return SceneState(objects=objects)
