"""Joint A (perception) + C2 (chunk policy) on frozen DINOv2 patch features."""

from __future__ import annotations

import logging

import numpy as np

from lfnav.aggregate import argmax_name
from lfnav.policy.base import Policy
from lfnav.policy.c2_data import DIRECTIONS, DISTANCES, FeatureSpec, encode_aux
from lfnav.policy.c2 import _visible
from lfnav.types import ObjectBelief, PolicyInput, PolicyOutput, SceneState

IGNORE = -100
log = logging.getLogger("lfnav.policy.joint")


CKPT_RUNTIME_KEYS = ("recent_k", "memory_size", "token_pool")


def joint_runtime_from_cfg(cfg: dict | None = None) -> dict:
    """Eval/train runtime from cfg.joint only — never policy.recent_k / policy.memory_size."""
    joint = (cfg or {}).get("joint") or {}
    return {
        "recent_k": int(joint["recent_k"] if joint.get("recent_k") is not None else 4),
        "memory_size": int(joint["memory_size"] if joint.get("memory_size") is not None else 15),
        "token_pool": int(joint["token_pool"] if joint.get("token_pool") is not None else 1),
    }


def checkpoint_runtime(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("joint checkpoint must be a dict")
    missing = [key for key in CKPT_RUNTIME_KEYS if key not in payload]
    if missing:
        raise RuntimeError(
            "checkpoint missing " + ", ".join(missing) + "; eval will not fall back to policy.*"
        )
    return {key: int(payload[key]) for key in CKPT_RUNTIME_KEYS}


def joint_spec(cfg: dict | None = None, runtime: dict | None = None) -> FeatureSpec:
    spec = FeatureSpec.from_cfg(cfg)
    spec.horizon = int(((cfg or {}).get("joint") or {}).get("chunk") or spec.horizon)
    rt = runtime if runtime is not None else joint_runtime_from_cfg(cfg)
    spec.recent_k = int(rt["recent_k"])
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


def joint_policy_flags(cfg: dict | None = None) -> dict:
    joint = (cfg or {}).get("joint") or {}
    return {
        "policy_uses_pooled": bool(joint.get("policy_uses_pooled", False)),
        "detach_scene": bool(joint.get("detach_scene", True)),
    }


def c2_input_dim(spec: FeatureSpec, dim: int, policy_uses_pooled: bool) -> int:
    n = spec.scene_dim + spec.aux_dim
    if policy_uses_pooled:
        n += int(dim)
    return n


def _state_dict(payload: dict) -> dict:
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        return payload["model"]
    return payload if isinstance(payload, dict) else {}


def infer_policy_uses_pooled(payload: dict, spec: FeatureSpec, dim: int | None = None) -> bool:
    """Whether C2's first linear layer includes the pooled DINOv2 width."""
    state = _state_dict(payload)
    weight = state.get("p1.weight")
    if weight is None:
        raise RuntimeError("cannot infer policy_uses_pooled: missing p1.weight")
    in_dim = int(weight.shape[1])
    if dim is None:
        dim = payload.get("dim") if isinstance(payload, dict) else None
        conv = state.get("conv.weight")
        if dim is None and conv is not None:
            dim = int(conv.shape[1])
    if dim is None:
        raise RuntimeError("cannot infer policy_uses_pooled: missing feature dim")
    dim = int(dim)
    with_pool = c2_input_dim(spec, dim, True)
    without = c2_input_dim(spec, dim, False)
    if in_dim == with_pool:
        return True
    if in_dim == without:
        return False
    raise RuntimeError(
        f"cannot infer policy_uses_pooled from p1 in_dim={in_dim} (expected {without} or {with_pool})"
    )


def assert_checkpoint_flags(payload: dict, cfg: dict | None = None, spec: FeatureSpec | None = None, dim: int | None = None) -> None:
    flags = joint_policy_flags(cfg)
    if not isinstance(payload, dict):
        raise RuntimeError("joint checkpoint must be a dict")
    if "policy_uses_pooled" in payload:
        got_pooled = bool(payload["policy_uses_pooled"])
    else:
        if spec is None:
            spec = joint_spec(cfg)
        got_pooled = infer_policy_uses_pooled(payload, spec, dim=dim if dim is not None else payload.get("dim"))
        log.warning("checkpoint missing policy_uses_pooled; inferred %s from C2 p1 in_dim", got_pooled)
    if got_pooled != flags["policy_uses_pooled"]:
        raise RuntimeError(f"checkpoint policy_uses_pooled={got_pooled} != config {flags['policy_uses_pooled']}")
    if "detach_scene" in payload:
        got_detach = bool(payload["detach_scene"])
        if got_detach != flags["detach_scene"]:
            raise RuntimeError(f"checkpoint detach_scene={got_detach} != config {flags['detach_scene']}")
    else:
        log.warning("checkpoint missing detach_scene; ignored at eval (train-only flag)")


def scene_vector_np(present_p, dir_p, dist_p, biome_p, time_p, weather_p, bright_p) -> np.ndarray:
    parts = [np.asarray(present_p, dtype=np.float32).reshape(-1)]
    parts.append(np.asarray(dir_p, dtype=np.float32).reshape(-1))
    parts.append(np.asarray(dist_p, dtype=np.float32).reshape(-1))
    for row in (biome_p, time_p, weather_p, bright_p):
        parts.append(np.asarray(row, dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)


def JointNet(
    dim: int,
    spec: FeatureSpec,
    hidden: int = 256,
    horizon: int | None = None,
    policy_uses_pooled: bool = False,
    detach_scene: bool = True,
):
    import torch
    from torch import nn

    n_obj = len(spec.objects)
    n_act = len(spec.actions)
    k = int(horizon or spec.horizon)
    uses_pooled = bool(policy_uses_pooled)
    detach = bool(detach_scene)
    in_dim = c2_input_dim(spec, dim, uses_pooled)

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
            self.policy_uses_pooled = uses_pooled
            self.detach_scene = detach
            self.c2_in_dim = in_dim
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
            vec = scene_vec.detach() if self.detach_scene else scene_vec
            parts = [vec, aux]
            if self.policy_uses_pooled:
                parts = [pooled, vec, aux]
            x = torch.cat(parts, dim=-1)
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
        self.hidden = int(joint.get("hidden") or 256)
        self.lambda_c2 = float(joint.get("lambda_c2", 1.0))
        self.flags = joint_policy_flags(self.cfg)
        self._torch = None
        self.model = None
        self.encoder = None
        self.last_seen = None
        self._tokens = None
        self._pooled = None
        self._scene = SceneState()
        runtime = joint_runtime_from_cfg(self.cfg)
        payload = None
        try:
            import torch
        except ImportError:
            torch = None
        self._torch = torch
        ckpt = joint.get("checkpoint")
        if ckpt:
            if torch is None:
                raise RuntimeError("joint checkpoint requires torch")
            payload = torch.load(ckpt, map_location="cpu")
            if not isinstance(payload, dict) or "model" not in payload:
                raise RuntimeError("joint checkpoint must be a dict with a model state")
            runtime = checkpoint_runtime(payload)
        self.runtime = runtime
        self.spec = joint_spec(self.cfg, runtime=runtime)
        self.memory_size = int(runtime["memory_size"])
        self.token_pool = int(runtime["token_pool"])
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
            self.model = JointNet(
                self.dim,
                self.spec,
                self.hidden,
                self.spec.horizon,
                policy_uses_pooled=self.flags["policy_uses_pooled"],
                detach_scene=self.flags["detach_scene"],
            )
            if payload is not None:
                assert_checkpoint_flags(payload, self.cfg, spec=self.spec, dim=self.dim)
                self.model.load_state_dict(payload["model"])
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

    def features(self, frame, *, saved_jpeg: bool = False) -> dict:
        from lfnav.data.demo_format import _hwc_uint8, preprocess_frame
        from lfnav.perception.joint_features import encode_joint_tokens

        rgb = frame
        live = isinstance(frame, dict)
        if live:
            from lfnav.env.frames import rgb_hwc

            rgb = rgb_hwc(frame, self.cfg)
            if rgb is None:
                rgb = frame.get("rgb")
        if live or not saved_jpeg:
            rgb = preprocess_frame(rgb)
        else:
            rgb = _hwc_uint8(rgb)
        encoded = encode_joint_tokens(rgb, self.encoder, self.token_pool)
        self._tokens = encoded["tokens"]
        self._pooled = encoded["pooled"]
        return encoded

    def aux_from_input(self, policy_input: PolicyInput) -> np.ndarray:
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
        return encode_aux(
            policy_input.task or {},
            sectors,
            policy_input.recent_actions or mem.get("recent_actions") or [],
            self.last_seen if self.last_seen is not None else mem.get("last_seen"),
            env_step,
            self.spec,
        )

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
        aux = self.aux_from_input(policy_input)
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
