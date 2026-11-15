"""C2Nav: patch-wise A heads plus a mode/waypoint/stop policy C."""

from __future__ import annotations

import json
import math
import random
import time

import numpy as np

from lfnav.aggregate import argmax_name, column_direction, distance_bin
from lfnav.policy.base import Policy
from lfnav.policy.c2nav_labels import FORWARD_BINS, MODE_NAMES, N_TURN, invert_waypoint, turn_center
from lfnav.policy.state_tracker import LOG_DIST_UNSEEN, STEPS_NORM, TargetObs, TrackerState, update, wrap_deg

VERSION = "c2nav_v1"
CHANNELS = 64
S_DIM = 18
DIR_LCR_NAMES = ("left", "center", "right")
DIST_BIN_NAMES = ("near", "mid", "far")
GEOM_DIM = 4 + len(DIR_LCR_NAMES) + len(DIST_BIN_NAMES)
NATIVE_PATCH_HW = (16, 26)
OBS_PRED = "pred"
OBS_GT = "gt"
DECISION_ACTION_CAP = 12
STOP_GATE_STOP = "stop"
STOP_GATE_NOT_VISIBLE = "blocked_not_visible"
STOP_GATE_CONDITIONS = "blocked_conditions"
STOP_GATE_TOO_FAR = "blocked_too_far"
STOP_GATE_BELOW_TAU = "below_tau"
STOP_POLICY_RULE = "rule"
STOP_POLICY_HEAD = "head"
PRESENT_POS_CAP = 5.0
VIS_POS_CAP = 5.0
PATCH_PRESENT_MIN = 0.3
DIST_BIN_LOSS_CAP = 5.0
# No real per-patch distance label: scene.patches is only 0/1, and the filled
# target is a constant log 8. Keep the head for old checkpoints, but do not train it.
PATCH_LOG_DIST_LOSS_WEIGHT = 0.0
C_PARAM_PREFIX = ("c1", "c2", "mode", "turn", "fwd", "stop")
NEW_A_PREFIXES = ("dist_bin_head.", "target_dist_head.")
WIDE_RADAR_KEYS = ("wide_rays", "lfnav_wide_rays", "wide_lidar", "radar_360")
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
    if cfg is None:
        cfg = {}
    labels = cfg.get("labels")
    if labels is None:
        labels = {}
    nav = cfg.get("c2nav")
    if nav is None:
        nav = {}
    objects = labels.get("objects")
    if objects is None:
        objects = ["tree", "grass", "water", "pig", "cow", "sheep", "chicken", "sand"]
    objects = list(objects)
    biomes = labels.get("biomes")
    if biomes is None:
        biomes = ["forest", "plains", "desert", "savanna", "taiga", "jungle", "other"]
    times = labels.get("time")
    if times is None:
        times = ["day", "night"]
    weathers = labels.get("weather")
    if weathers is None:
        weathers = ["clear", "rain"]
    brightness = labels.get("brightness")
    if brightness is None:
        brightness = ["sufficient", "insufficient"]
    hidden = nav.get("hidden")
    channels = nav.get("channels")
    dim = nav.get("dim")
    grid = nav.get("grid")
    joint = cfg.get("joint")
    if joint is None:
        joint = {}
    token_pool = nav.get("token_pool")
    if token_pool is None:
        token_pool = joint.get("token_pool")
    if token_pool is None:
        token_pool = 2
    dropout = nav.get("dropout")
    chunk = nav.get("chunk_max_steps")
    obs_source = nav.get("obs_source")
    vis_tau = nav.get("vis_tau")
    torch_threads = nav.get("torch_threads")
    a_extra_per_epoch = nav.get("a_extra_per_epoch")
    version = nav.get("version")
    return {
        "objects": objects,
        "biomes": list(biomes),
        "times": list(times),
        "weathers": list(weathers),
        "brightness": list(brightness),
        "hidden": 256 if hidden is None else int(hidden),
        "channels": CHANNELS if channels is None else int(channels),
        "dim": 384 if dim is None else int(dim),
        "grid": (8, 13) if grid is None else tuple(grid),
        "token_pool": int(token_pool),
        "dropout": 0.3 if dropout is None else float(dropout),
        "chunk_max_steps": DECISION_ACTION_CAP if chunk is None else int(chunk),
        "obs_source": OBS_PRED if obs_source is None else str(obs_source),
        "vis_tau": 0.5 if vis_tau is None else float(vis_tau),
        "torch_threads": 2 if torch_threads is None else int(torch_threads),
        "a_extra_per_epoch": 30000 if a_extra_per_epoch is None else int(a_extra_per_epoch),
        "modes": list(MODE_NAMES),
        "n_turn": int(N_TURN),
        "forward_bins": list(FORWARD_BINS),
        "s_dim": S_DIM,
        "geom_dim": GEOM_DIM,
        "cond_dim": 1 + len(objects) + 4,
        "version": VERSION if version is None else str(version),
    }


def horizontal_fov_deg(cfg: dict | None = None) -> float:
    cfg = cfg or {}
    hw = list(cfg.get("image_hw") or [224, 364])
    height = float(hw[0])
    width = float(hw[1])
    vfov = math.radians(float(cfg.get("fov_deg") or 70))
    return math.degrees(2.0 * math.atan(math.tan(vfov / 2.0) * (width / height)))


def column_bearing_deg(col: float, n_w: int, hfov_deg: float) -> float:
    u = (float(col) + 0.5) / float(n_w) - 0.5
    return math.degrees(math.atan(2.0 * u * math.tan(math.radians(float(hfov_deg)) / 2.0)))


def patch_bearings(n_h: int, n_w: int, cfg: dict | None = None) -> np.ndarray:
    """Yaw at each patch, degrees. Image centre is 0, right is positive. HFOV from 224x364 / 70° VFOV."""
    hfov = horizontal_fov_deg(cfg)
    yaw = np.zeros((int(n_h), int(n_w)), dtype=np.float32)
    for j in range(int(n_w)):
        yaw[:, j] = column_bearing_deg(j, int(n_w), hfov)
    return yaw


def align_hw(arr, height: int, width: int, how: str = "max") -> np.ndarray:
    """Map a 2D grid onto (height, width). 16x26 ignore/present becomes 8x13 by 2x2 pooling."""
    gh, gw = int(height), int(width)
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        return np.zeros((gh, gw), np.float32)
    h, w = int(a.shape[0]), int(a.shape[1])
    if h == gh and w == gw:
        return a
    if gh < 1 or gw < 1:
        return np.zeros((gh, gw), np.float32)
    kh, kw = h // gh, w // gw
    if kh < 1 or kw < 1 or h != kh * gh or w != kw * gw:
        return np.zeros((gh, gw), np.float32)
    blocks = a.reshape(gh, kh, gw, kw)
    if how == "max":
        return np.asarray(blocks.max(axis=(1, 3)), dtype=np.float32)
    if how == "min":
        return np.asarray(blocks.min(axis=(1, 3)), dtype=np.float32)
    return np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float32)


def obs_source_of(cfg: dict | None = None) -> str:
    nav = (cfg or {}).get("c2nav") or {}
    src = nav.get("obs_source")
    if src is None:
        return OBS_PRED
    name = str(src)
    if name not in (OBS_PRED, OBS_GT):
        raise RuntimeError(f"c2nav.obs_source must be {OBS_PRED!r} or {OBS_GT!r}, got {name!r}")
    return name


def stop_requires_visible_of(cfg: dict | None = None) -> bool:
    nav = (cfg or {}).get("c2nav") or {}
    raw = nav.get("stop_requires_visible")
    if raw is None:
        return True
    return bool(raw)


def stop_policy_of(cfg: dict | None = None) -> str:
    nav = (cfg or {}).get("c2nav") or {}
    raw = nav.get("stop_policy")
    if raw is None:
        return STOP_POLICY_RULE
    name = str(raw)
    if name not in (STOP_POLICY_RULE, STOP_POLICY_HEAD):
        raise RuntimeError(f"c2nav.stop_policy must be {STOP_POLICY_RULE!r} or {STOP_POLICY_HEAD!r}, got {name!r}")
    return name


def forward_backward_weight_cap_of(cfg: dict | None = None) -> float:
    nav = (cfg or {}).get("c2nav")
    if nav is None:
        nav = {}
    raw = nav.get("forward_backward_weight_cap")
    if raw is None:
        return 10.0
    return float(raw)


def explore_sample_of(cfg: dict | None = None) -> bool:
    nav = (cfg or {}).get("c2nav") or {}
    raw = nav.get("explore_sample")
    if raw is None:
        return True
    return bool(raw)


def train_a_only_of(cfg: dict | None = None) -> bool:
    nav = (cfg or {}).get("c2nav")
    if nav is None:
        nav = {}
    raw = nav.get("train_a_only")
    if raw is None:
        return False
    return bool(raw)


def init_checkpoint_of(cfg: dict | None = None):
    nav = (cfg or {}).get("c2nav")
    if nav is None:
        nav = {}
    raw = nav.get("init_checkpoint")
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    return text


def stop_near_tau_of(cfg: dict | None = None) -> float:
    nav = (cfg or {}).get("c2nav")
    if nav is None:
        nav = {}
    raw = nav.get("stop_near_tau")
    if raw is None:
        return 0.5
    return float(raw)


def explore_temp_of(cfg: dict | None = None) -> float:
    nav = (cfg or {}).get("c2nav")
    if nav is None:
        nav = {}
    raw = nav.get("explore_temp")
    if raw is None:
        return 1.0
    return float(raw)


def softmax_turn(logits, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    if x.size < 1:
        return np.zeros((0,), dtype=np.float64)
    t = float(temp)
    if (not np.isfinite(t)) or t <= 0:
        p = np.zeros(x.shape[0], dtype=np.float64)
        p[int(np.argmax(x))] = 1.0
        return p
    z = x / t
    z = z - np.max(z)
    e = np.exp(z)
    s = float(e.sum())
    if (not np.isfinite(s)) or s <= 0:
        p = np.zeros(x.shape[0], dtype=np.float64)
        p[int(np.argmax(x))] = 1.0
        return p
    return e / s


def sample_turn_index(logits, rng, temp: float = 1.0) -> int:
    p = softmax_turn(logits, temp)
    n = int(p.shape[0])
    if n < 1:
        return 0
    return int(rng.choices(range(n), weights=[float(v) for v in p], k=1)[0])


def top_k_turn_probs(logits, k: int = 3, temp: float = 1.0) -> list:
    p = softmax_turn(logits, temp)
    n = int(p.shape[0])
    if n < 1:
        return []
    take = min(int(k), n)
    order = np.argsort(-p)[:take]
    out = []
    for i in order:
        idx = int(i)
        out.append({
            "index": idx,
            "turn_deg": float(turn_center(idx)),
            "p": float(p[idx]),
        })
    return out


def torch_threads_of(cfg: dict | None = None) -> int:
    nav = (cfg or {}).get("c2nav")
    if nav is None:
        nav = {}
    raw = nav.get("torch_threads")
    if raw is None:
        return 2
    return int(raw)


def a_extra_per_epoch_of(cfg: dict | None = None) -> int:
    nav = (cfg or {}).get("c2nav")
    if nav is None:
        nav = {}
    raw = nav.get("a_extra_per_epoch")
    if raw is None:
        return 30000
    return int(raw)


def a_extra_features_of(cfg: dict | None = None) -> list:
    nav = (cfg or {}).get("c2nav")
    if nav is None:
        nav = {}
    raw = nav.get("a_extra_features")
    if raw is None:
        return []
    return list(raw)


def a_extra_demos_of(cfg: dict | None = None) -> list:
    nav = (cfg or {}).get("c2nav")
    if nav is None:
        nav = {}
    raw = nav.get("a_extra_demos")
    if raw is None:
        return []
    return list(raw)


def ckpt_dir_of(cfg: dict | None = None) -> str:
    nav = (cfg or {}).get("c2nav")
    if nav is None:
        nav = {}
    raw = nav.get("ckpt_dir")
    if raw is None:
        return "c2nav_easy"
    return str(raw)


def stop_max_dist_of(cfg: dict | None = None) -> float:
    nav = (cfg or {}).get("c2nav")
    if nav is None:
        nav = {}
    raw = nav.get("stop_max_dist")
    if raw is not None:
        return float(raw)
    judge = (cfg or {}).get("judge")
    if judge is None:
        judge = {}
    radius = judge.get("radius")
    if radius is None:
        radius = 3.0
    return float(radius) - 0.3


def conditions_satisfied(match: dict | None) -> bool:
    """B reports every required condition holds (status found, nothing missing)."""
    data = {} if match is None else dict(match)
    missing = data.get("missing")
    if missing is not None and len(list(missing)) > 0:
        return False
    status = data.get("status")
    if status is not None:
        return str(status) == "found"
    sat = data.get("satisfied")
    if sat is not None:
        return all(bool(v) for v in sat.values())
    return False


def infer_stop_gate(
    p_stop,
    tau,
    visible,
    match,
    requires_visible: bool = True,
    dist=None,
    max_dist=None,
    policy=None,
    p_near=None,
    near_tau=None,
) -> str:
    """Inference stop: rule ignores p_stop; head requires p_stop >= tau, then the same gates."""
    name = STOP_POLICY_RULE if policy is None else str(policy)
    if name not in (STOP_POLICY_RULE, STOP_POLICY_HEAD):
        raise RuntimeError(f"c2nav.stop_policy must be {STOP_POLICY_RULE!r} or {STOP_POLICY_HEAD!r}, got {name!r}")
    if name == STOP_POLICY_HEAD and float(p_stop) < float(tau):
        return STOP_GATE_BELOW_TAU
    if requires_visible and not bool(visible):
        return STOP_GATE_NOT_VISIBLE
    if not conditions_satisfied(match):
        return STOP_GATE_CONDITIONS
    if dist is None or max_dist is None or float(dist) > float(max_dist):
        return STOP_GATE_TOO_FAR
    if near_tau is not None and float(near_tau) > 0.0:
        if p_near is None or float(p_near) < float(near_tau):
            return STOP_GATE_TOO_FAR
    return STOP_GATE_STOP


def pool2d(arr: np.ndarray, k: int, how: str) -> np.ndarray:
    a = np.asarray(arr)
    k = max(int(k), 1)
    if k <= 1:
        return a
    h, w = a.shape
    nh, nw = h // k, w // k
    if nh == 0 or nw == 0:
        return a
    blocks = a[: nh * k, : nw * k].reshape(nh, k, nw, k)
    if how == "max":
        return blocks.max(axis=(1, 3))
    if how == "min":
        return blocks.min(axis=(1, 3))
    return blocks.mean(axis=(1, 3))


def pool_to_grid(arr: np.ndarray, grid: tuple[int, int], how: str = "max") -> np.ndarray:
    """Pool or keep a 2D map so it matches grid. 16x26 patches become 8x13 with factor 2."""
    gh, gw = int(grid[0]), int(grid[1])
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        return np.zeros((gh, gw), np.float32)
    h, w = int(a.shape[0]), int(a.shape[1])
    if h == gh and w == gw:
        return a
    kh = h // gh
    kw = w // gw
    if kh < 1 or kw < 1:
        return np.zeros((gh, gw), np.float32)
    k = kh if kh == kw else kh
    pooled = pool2d(a, k, how)
    if pooled.shape != (gh, gw):
        return np.zeros((gh, gw), np.float32)
    return np.asarray(pooled, dtype=np.float32)


GT_POOL_PRESENT_HOW = "max"
GT_POOL_IGNORE_HOW = "min"
GT_POOL_DIST_HOW = "min"


def pool_gt_to_grid(
    present_hw,
    ignore_hw,
    dist_hw,
    grid: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool 16x26 GT maps onto the token grid (8x13 when pool=2).

    Present: max of the block (any positive sub-patch stays positive).
    Ignore: min of the block (ignored only if every sub-patch is ignored).
    Distance: min of present-positive sub-patches in the block.
    """
    gh, gw = int(grid[0]), int(grid[1])
    present_raw = np.asarray(present_hw, dtype=np.float32)
    if present_raw.ndim != 2:
        present_raw = np.zeros((gh, gw), np.float32)
    present = pool_to_grid(present_raw, grid, GT_POOL_PRESENT_HOW)
    if ignore_hw is None:
        ignore = np.zeros((gh, gw), np.float32)
    else:
        ig = np.asarray(ignore_hw, dtype=np.float32)
        if ig.ndim != 2:
            ignore = np.zeros((gh, gw), np.float32)
        else:
            ignore = pool_to_grid(ig, grid, GT_POOL_IGNORE_HOW)
    dist = np.zeros((gh, gw), np.float32)
    if dist_hw is not None:
        raw_d = np.asarray(dist_hw, dtype=np.float32)
        if raw_d.ndim == 2:
            if raw_d.shape != present_raw.shape:
                filled = np.full(present_raw.shape, np.inf, dtype=np.float32)
            else:
                filled = np.where(present_raw > 0.5, raw_d, np.inf).astype(np.float32)
                if not np.any(present_raw > 0.5):
                    filled = np.where(np.isfinite(raw_d), raw_d, np.inf).astype(np.float32)
            pooled_d = pool_to_grid(filled, grid, GT_POOL_DIST_HOW)
            dist = np.where(np.isfinite(pooled_d), pooled_d, 0.0).astype(np.float32)
    return present, ignore, dist


def patch_targets(scene: dict, objects: list[str], pool: int, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del pool
    gh, gw = int(grid[0]), int(grid[1])
    k = len(objects)
    present = np.zeros((k, gh, gw), np.float32)
    log_d = np.zeros((k, gh, gw), np.float32)
    ignore = np.zeros((gh, gw), np.float32)
    raw_ignore = (scene or {}).get("ignore") or []
    ig = None
    if raw_ignore:
        ig_arr = np.asarray(raw_ignore, dtype=np.float32)
        if ig_arr.ndim == 2:
            ig = ig_arr
            ignore = pool_to_grid(ig, grid, GT_POOL_IGNORE_HOW)
    patches = (scene or {}).get("patches") or {}
    dist_maps = (scene or {}).get("patch_dist")
    if dist_maps is None:
        dist_maps = {}
    for i, name in enumerate(objects):
        grid_p = np.asarray(patches.get(name) or [], dtype=np.float32)
        if grid_p.ndim != 2:
            continue
        dist_p = dist_maps.get(name) if isinstance(dist_maps, dict) else None
        pooled, ignore, dist = pool_gt_to_grid(grid_p, ig, dist_p, grid)
        present[i] = pooled
        if dist_p is None:
            log_d[i] = np.where(pooled > 0.5, math.log(8.0), 0.0)
        else:
            log_d[i] = np.where(pooled > 0.5, np.log(np.maximum(dist, 0.05)), 0.0)
    return present, log_d, ignore


def distance_bins_of(cfg: dict | None = None) -> tuple[float, float]:
    bins = (cfg or {}).get("distance_bins")
    if bins is None:
        bins = {}
    near = 3.0 if bins.get("near") is None else float(bins["near"])
    mid = 10.0 if bins.get("mid") is None else float(bins["mid"])
    return near, mid


def upsample_nearest(arr, out_hw) -> np.ndarray:
    """Nearest-neighbor upsample a 2D map (8x13 present_p → 16x26)."""
    a = np.asarray(arr, dtype=np.float32)
    oh, ow = int(out_hw[0]), int(out_hw[1])
    if a.ndim != 2:
        return np.zeros((oh, ow), np.float32)
    h, w = int(a.shape[0]), int(a.shape[1])
    if h == oh and w == ow:
        return a
    if h < 1 or w < 1:
        return np.zeros((oh, ow), np.float32)
    ys = np.minimum((np.arange(oh) * h / float(oh)).astype(np.int64), h - 1)
    xs = np.minimum((np.arange(ow) * w / float(ow)).astype(np.int64), w - 1)
    return np.asarray(a[np.ix_(ys, xs)], dtype=np.float32)


def dir_lcr_from_present(present_hw, ignore_hw=None, native_hw=NATIVE_PATCH_HW) -> str | None:
    """Mass per image third via column_direction. Upsample to native 16x26 if needed."""
    p = np.asarray(present_hw, dtype=np.float32)
    if p.ndim != 2:
        return None
    native = (int(native_hw[0]), int(native_hw[1]))
    if (int(p.shape[0]), int(p.shape[1])) != native:
        p = upsample_nearest(p, native)
    ig = None if ignore_hw is None else np.asarray(ignore_hw, dtype=np.float32)
    if ig is not None and ig.ndim == 2 and tuple(int(x) for x in ig.shape) != tuple(p.shape):
        ig = upsample_nearest(ig, p.shape)
    n_h, n_w = int(p.shape[0]), int(p.shape[1])
    mass = {"left": 0.0, "center": 0.0, "right": 0.0}
    for i in range(n_h):
        for j in range(n_w):
            if ig is not None and ig.ndim == 2 and float(ig[i, j]) >= 0.5:
                continue
            mass[column_direction(j, n_w)] += float(p[i, j])
    if sum(mass.values()) <= 0.0:
        return None
    return argmax_name(mass)


def dist_bin_from_num(dist_num, cfg: dict | None = None) -> str | None:
    if dist_num is None:
        return None
    near, mid = distance_bins_of(cfg)
    return distance_bin(float(dist_num), near, mid)


def dist_bin_from_mass(dist_mass) -> str | None:
    if not dist_mass:
        return None
    packed = {str(k): float(v) for k, v in dict(dist_mass).items()}
    name = argmax_name(packed)
    if name in DIST_BIN_NAMES:
        return name
    return None


def dist_bin_labels_from_objects(scene, objects) -> tuple[np.ndarray, np.ndarray]:
    """Per-class near/mid/far from scene.objects. Invisible classes (present not true) are masked out."""
    names = list(objects)
    bins = np.zeros((len(names),), np.int64)
    mask = np.zeros((len(names),), np.float32)
    raw = scene.get("objects") if isinstance(scene, dict) else None
    if not isinstance(raw, dict):
        return bins, mask
    for i, name in enumerate(names):
        bel = raw.get(name)
        if not isinstance(bel, dict):
            continue
        present = bel.get("present")
        if present is None or float(present) <= 0.5:
            continue
        mass = bel.get("distance") or {}
        if not isinstance(mass, dict) or sum(float(v) for v in mass.values()) <= 0.0:
            continue
        chosen = dist_bin_from_mass(mass)
        if chosen is None:
            continue
        bins[i] = DIST_BIN_NAMES.index(chosen)
        mask[i] = 1.0
    return bins, mask


def target_log_dist_label(dist_num) -> tuple[float, float]:
    """Smooth-L1 target. Empty target_distance_num is not supervised."""
    if dist_num is None:
        return 0.0, 0.0
    return math.log(max(float(dist_num), 0.05)), 1.0


def vis_is_positive(vis_logit, vis_tau: float) -> bool:
    prob = 1.0 / (1.0 + math.exp(-float(vis_logit)))
    return prob >= float(vis_tau)


def softmax_last(logits) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1]) if x.ndim >= 1 else x.reshape(1, -1)
    z = flat - np.max(flat, axis=-1, keepdims=True)
    e = np.exp(z)
    prob = e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)
    return prob.reshape(x.shape)


def freeze_c_params(model) -> int:
    """Freeze policy C. Returns the number of parameters that stay trainable."""
    n = 0
    for name, param in model.named_parameters():
        head = name.split(".", 1)[0]
        if head in C_PARAM_PREFIX:
            param.requires_grad_(False)
        else:
            n += int(param.numel())
    return n


def load_compatible_state(model, state: dict) -> tuple[list, list]:
    """Load with strict=False. New heads stay at their random initialization."""
    own = model.state_dict()
    filtered = {}
    for key, value in state.items():
        if key not in own or tuple(own[key].shape) != tuple(getattr(value, "shape", ())):
            continue
        filtered[key] = value
    missing, _unexpected = model.load_state_dict(filtered, strict=False)
    return sorted(filtered.keys()), list(missing)


def dist_bin_from_present_log(present_hw, log_dist_hw, ignore_hw=None, cfg: dict | None = None, native_hw=NATIVE_PATCH_HW) -> str | None:
    p = np.asarray(present_hw, dtype=np.float32)
    d = None if log_dist_hw is None else np.asarray(log_dist_hw, dtype=np.float32)
    if p.ndim != 2 or d is None or d.ndim != 2:
        return None
    native = (int(native_hw[0]), int(native_hw[1]))
    if (int(p.shape[0]), int(p.shape[1])) != native:
        p = upsample_nearest(p, native)
    if (int(d.shape[0]), int(d.shape[1])) != tuple(p.shape):
        d = upsample_nearest(d, p.shape)
    ig = None if ignore_hw is None else np.asarray(ignore_hw, dtype=np.float32)
    if ig is not None and ig.ndim == 2 and tuple(int(x) for x in ig.shape) != tuple(p.shape):
        ig = upsample_nearest(ig, p.shape)
    near, mid = distance_bins_of(cfg)
    mass = {name: 0.0 for name in DIST_BIN_NAMES}
    n_h, n_w = int(p.shape[0]), int(p.shape[1])
    for i in range(n_h):
        for j in range(n_w):
            if ig is not None and ig.ndim == 2 and float(ig[i, j]) >= 0.5:
                continue
            w = float(p[i, j])
            if w <= 0.0:
                continue
            mass[distance_bin(math.exp(float(d[i, j])), near, mid)] += w
    if sum(mass.values()) <= 0.0:
        return None
    return argmax_name(mass)


def dist_bin_from_log(log_d, cfg: dict | None = None) -> str | None:
    if log_d is None:
        return None
    return dist_bin_from_num(math.exp(float(log_d)), cfg)


def attach_lcr_dist(obs: TargetObs, present_hw, ignore_hw, dist_num, cfg: dict | None = None, dist_mass=None, log_dist_hw=None) -> TargetObs:
    if not obs.visible:
        obs.dir_lcr = None
        obs.dist_bin = None
        return obs
    obs.dir_lcr = dir_lcr_from_present(present_hw, ignore_hw)
    obs.dist_bin = dist_bin_from_mass(dist_mass)
    if obs.dist_bin is None and log_dist_hw is not None:
        obs.dist_bin = dist_bin_from_present_log(present_hw, log_dist_hw, ignore_hw, cfg)
    if obs.dist_bin is None and dist_num is not None:
        obs.dist_bin = dist_bin_from_num(dist_num, cfg)
    if obs.dist_bin is None:
        obs.dist_bin = dist_bin_from_log(obs.log_distance, cfg)
    return obs


def geom_from_obs(obs: TargetObs) -> np.ndarray:
    lcr = np.zeros(len(DIR_LCR_NAMES), np.float32)
    dbin = np.zeros(len(DIST_BIN_NAMES), np.float32)
    if not obs.visible:
        return np.concatenate([
            np.asarray([0.0, 0.0, 1.0, math.log(32.0)], np.float32), lcr, dbin,
        ])
    yaw = 0.0 if obs.bearing_deg is None else float(obs.bearing_deg)
    rad = math.radians(yaw)
    log_d = float(obs.log_distance if obs.log_distance is not None else math.log(32.0))
    if obs.dir_lcr in DIR_LCR_NAMES:
        lcr[DIR_LCR_NAMES.index(obs.dir_lcr)] = 1.0
    if obs.dist_bin in DIST_BIN_NAMES:
        dbin[DIST_BIN_NAMES.index(obs.dist_bin)] = 1.0
    return np.concatenate([
        np.asarray([1.0, math.sin(rad), math.cos(rad), log_d], np.float32), lcr, dbin,
    ])


def gt_target_obs(present: np.ndarray, ignore: np.ndarray, target_index: int, bearings: np.ndarray, dist_num) -> TargetObs:
    """Visible if the target has at least one non-ignore positive patch. Bearing is mean column yaw."""
    p = np.asarray(present[int(target_index)], np.float32)
    valid = np.asarray(ignore, np.float32) < 0.5
    hit = (p > 0.5) & valid
    if not np.any(hit):
        return TargetObs(visible=False)
    yaw_map = np.asarray(bearings, np.float32)
    yaw = float(yaw_map[hit].mean())
    if dist_num is None:
        log_d = float(LOG_DIST_UNSEEN)
    else:
        log_d = math.log(max(float(dist_num), 0.05))
    return TargetObs(visible=True, bearing_deg=yaw, log_distance=log_d)


def vis_labels(present: np.ndarray, ignore: np.ndarray, class_ignore=None) -> tuple[np.ndarray, np.ndarray]:
    """Frame-level vis: 1 if any non-HUD patch of the class is positive. class_ignore skips the loss."""
    p = np.asarray(present, np.float32)
    ig = np.asarray(ignore, np.float32)
    k = int(p.shape[0])
    vis = np.zeros((k,), np.float32)
    vis_ig = np.zeros((k,), np.float32)
    valid = ig < 0.5 if ig.ndim == 2 else np.ones(p.shape[1:], dtype=bool)
    ci = np.zeros((k,), np.float32) if class_ignore is None else np.asarray(class_ignore, np.float32).reshape(-1)
    for i in range(k):
        if i < ci.shape[0] and ci[i] >= 0.5:
            vis_ig[i] = 1.0
            continue
        if not np.any(valid):
            vis_ig[i] = 1.0
            continue
        vis[i] = 1.0 if np.any((p[i] > 0.5) & valid) else 0.0
    return vis, vis_ig


def wide_radar_field(step: dict | None) -> tuple[str | None, list | None]:
    """Return (field_name, rays) if a 360° radar list is stored on the step."""
    if not isinstance(step, dict):
        return None, None
    blobs = (step, step.get("scene") if isinstance(step.get("scene"), dict) else {}, step.get("memory") if isinstance(step.get("memory"), dict) else {})
    for blob in blobs:
        for key in WIDE_RADAR_KEYS:
            val = blob.get(key)
            if val:
                return str(key), list(val)
    return None, None


def far_class_ignore(step: dict, objects: list[str], hfov_deg: float, d0: float = 16.0, d1: float = 32.0) -> np.ndarray:
    """Ignore present/vis labels when wide radar sees the class at 16–32 blocks inside the camera HFOV."""
    ig = np.zeros((len(objects),), np.float32)
    _key, rays = wide_radar_field(step)
    if rays is None:
        return ig
    half = float(hfov_deg) / 2.0
    names = {str(n): i for i, n in enumerate(objects)}
    for ray in rays:
        if not isinstance(ray, dict):
            continue
        dist = ray.get("distance")
        if dist is None:
            continue
        d = float(dist)
        if d < float(d0) or d > float(d1):
            continue
        yaw = ray.get("yaw_offset")
        if yaw is None:
            yaw = ray.get("bearing")
        if yaw is None:
            continue
        if abs(wrap_deg(float(yaw))) > half:
            continue
        label = ray.get("label")
        if label is None:
            label = ray.get("name")
        if label in names:
            ig[names[str(label)]] = 1.0
    return ig


def obs_from_present_head(
    present_p,
    log_dist,
    target_index: int,
    bearings: np.ndarray,
    vis_logit,
    vis_tau: float,
    ignore=None,
    patch_min: float = PATCH_PRESENT_MIN,
) -> TargetObs:
    """A5: vis_head decides visible; bearing/dist from present patches (argmax if none ≥ 0.3)."""
    visible = vis_is_positive(vis_logit, vis_tau)
    p = np.asarray(present_p[int(target_index)], np.float32)
    dmap = np.asarray(log_dist[int(target_index)], np.float32)
    yaw_map = np.asarray(bearings, np.float32)
    valid = np.ones(p.shape, dtype=bool)
    if ignore is not None:
        ig = np.asarray(ignore, np.float32)
        if ig.shape == p.shape:
            valid = ig < 0.5
    strong = (p >= float(patch_min)) & valid
    if visible and not np.any(strong):
        masked = np.where(valid, p, -1.0)
        idx = int(np.argmax(masked))
        strong = np.zeros(p.shape, dtype=bool)
        strong.reshape(-1)[idx] = True
        strong = strong.reshape(p.shape)
        if float(masked.reshape(-1)[idx]) < 0.0:
            strong[:] = False
    if not visible or not np.any(strong):
        return TargetObs(visible=False)
    yaw = float(yaw_map[strong].mean())
    log_d = float(dmap[strong].mean())
    return TargetObs(visible=True, bearing_deg=yaw, log_distance=log_d)


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


def encode_s(state: TrackerState | dict | list[float], sector8=None) -> np.ndarray:
    if isinstance(state, TrackerState):
        vec = list(state.vector())
    elif isinstance(state, dict):
        vec = list(TrackerState.from_dict(state).vector())
    else:
        vec = list(np.asarray(state, dtype=np.float32).reshape(-1))
    if sector8 is not None:
        counts = [float(x) for x in list(sector8)[:8]]
        if len(counts) < 8:
            counts = counts + [0.0] * (8 - len(counts))
        total = sum(counts) or 1.0
        if len(vec) < 15:
            vec = vec + [0.0] * (15 - len(vec))
        vec[7:15] = [c / total for c in counts]
    return np.asarray(vec, dtype=np.float32)


FORWARD_KEYS = ("tokens", "s_vec", "cond", "target_index", "bearings")


def _yaw_for_grid(height: int, width: int, bearings, device, dtype, cfg: dict | None = None):
    import torch

    if bearings is not None:
        yaw = torch.as_tensor(bearings, device=device, dtype=dtype)
        if tuple(int(x) for x in yaw.shape[-2:]) == (int(height), int(width)):
            return yaw
    native = patch_bearings(int(height), int(width), cfg)
    return torch.as_tensor(native, device=device, dtype=dtype)


def _ignore_for_grid(ignore, height: int, width: int, device, dtype):
    import torch

    if ignore is None:
        return None
    if torch.is_tensor(ignore):
        arr = ignore.detach().cpu().numpy()
    else:
        arr = np.asarray(ignore, np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        aligned = pool_to_grid(arr, (int(height), int(width)), GT_POOL_IGNORE_HOW)
        return torch.as_tensor(aligned, device=device, dtype=dtype)
    if arr.ndim == 3:
        stacked = np.stack([
            pool_to_grid(row, (int(height), int(width)), GT_POOL_IGNORE_HOW) for row in arr
        ])
        return torch.as_tensor(stacked, device=device, dtype=dtype)
    return None


def target_geometry(present_p, log_dist, target_index, bearings=None, vis_tau: float = 0.5, ignore=None, cfg: dict | None = None):
    """present_p, log_dist: (B, K, H, W). Yaw uses the actual (H, W); ignore uses pool_gt_to_grid (min)."""
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
    hit = p >= float(vis_tau)
    ig = _ignore_for_grid(ignore, height, width, p.device, p.dtype)
    if ig is not None:
        if ig.ndim == 2:
            ig = ig.unsqueeze(0)
        hit = hit & (ig < 0.5)
    vis = hit.flatten(1).any(dim=1).to(dtype=p.dtype)
    mass = hit.to(dtype=p.dtype)
    denom = mass.flatten(1).sum(dim=1).clamp_min(1e-6)
    yaw = _yaw_for_grid(height, width, bearings, p.device, p.dtype, cfg)
    yaw_hat = (mass * yaw).flatten(1).sum(dim=1) / denom
    log_d = (mass * dist).flatten(1).sum(dim=1) / denom
    unseen = vis < 0.5
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
    geom_dim = int(spec["geom_dim"] if spec.get("geom_dim") is not None else GEOM_DIM)
    c_in = geom_dim + s_dim + cond_dim

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
            self.vis_fc1 = nn.Linear(pooled, pooled)
            self.vis_fc2 = nn.Linear(pooled, n_obj)
            self.dist_bin_head = nn.Sequential(
                nn.Linear(pooled, pooled),
                nn.ReLU(),
                nn.Linear(pooled, n_obj * 3),
            )
            self.target_dist_head = nn.Sequential(
                nn.Linear(pooled, pooled),
                nn.ReLU(),
                nn.Linear(pooled, n_obj),
            )
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
            vis_logit = self.vis_fc2(nn.functional.relu(self.vis_fc1(pooled)))
            b = int(pooled.shape[0])
            dist_bin_logit = self.dist_bin_head(pooled).view(b, n_obj, 3)
            target_log_dist = self.target_dist_head(pooled)
            return {
                "present_logit": present_logit,
                "present_p": torch.sigmoid(present_logit),
                "log_dist": log_d,
                "vis_logit": vis_logit,
                "vis_p": torch.sigmoid(vis_logit),
                "dist_bin_logit": dist_bin_logit,
                "target_log_dist": target_log_dist,
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
            vis_tau = float(self.spec["vis_tau"] if self.spec.get("vis_tau") is not None else 0.5)
            vis, sn, cs, log_d = target_geometry(
                a["present_p"], a["log_dist"], batch["target_index"], batch.get("bearings"),
                vis_tau=vis_tau, ignore=batch.get("ignore"), cfg=None,
            )
            geom_from_a = torch.stack([vis, sn, cs, log_d], dim=-1)
            gdim = int(self.spec["geom_dim"] if self.spec.get("geom_dim") is not None else GEOM_DIM)
            if geom_from_a.shape[-1] < gdim:
                pad = geom_from_a.new_zeros(geom_from_a.shape[0], gdim - geom_from_a.shape[-1])
                geom_from_a = torch.cat([geom_from_a, pad], dim=-1)
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
    loaded, missing = load_compatible_state(net, state)
    print("c2nav loaded_params " + json.dumps(loaded), flush=True)
    print("c2nav missing_params " + json.dumps(missing), flush=True)
    net.eval()
    return net, spec


def scene_from_perception(a, spec: dict, bearings: np.ndarray, ignore=None, cfg: dict | None = None):
    """B's object present is the same frame-level vis_head decision C uses, not max patch probability."""
    import torch

    from lfnav.types import ObjectBelief, SceneState

    present_p = a["present_p"][0].detach().cpu().numpy()
    log_dist = a["log_dist"][0].detach().cpu().numpy()
    vis_logit = a["vis_logit"][0].detach().cpu().numpy()
    dist_logit = a["dist_bin_logit"][0].detach().cpu().numpy()
    vis_tau = float(spec["vis_tau"] if spec.get("vis_tau") is not None else 0.5)
    t_p = torch.from_numpy(np.asarray(present_p, np.float32)).unsqueeze(0)
    t_d = torch.from_numpy(np.asarray(log_dist, np.float32)).unsqueeze(0)
    objects = {}
    for i, name in enumerate(spec["objects"]):
        visible = vis_is_positive(vis_logit[i], vis_tau)
        _patch_vis, sn, cs, _log_d = target_geometry(
            t_p, t_d, i, bearings, vis_tau=PATCH_PRESENT_MIN, ignore=ignore, cfg=cfg,
        )
        yaw = math.degrees(math.atan2(float(sn), float(cs))) if visible else 0.0
        if not visible:
            direction = {"left": 0.0, "center": 0.0, "right": 0.0}
            distance = {key: 0.0 for key in DIST_BIN_NAMES}
        else:
            if yaw < -20:
                direction = {"left": 1.0, "center": 0.0, "right": 0.0}
            elif yaw > 20:
                direction = {"left": 0.0, "center": 0.0, "right": 1.0}
            else:
                direction = {"left": 0.0, "center": 1.0, "right": 0.0}
            probs = softmax_last(dist_logit[i])
            distance = {DIST_BIN_NAMES[j]: float(probs[j]) for j in range(len(DIST_BIN_NAMES))}
        objects[name] = ObjectBelief(present=1.0 if visible else 0.0, direction=direction, distance=distance)
    return SceneState(objects=objects)


class C2NavAgent(Policy):
    name = "c2nav"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        nav = self.cfg.get("c2nav") or {}
        stop_tau = nav.get("stop_tau")
        self.tau = 0.5 if stop_tau is None else float(stop_tau)
        chunk = nav.get("chunk_max_steps")
        self.chunk_max = DECISION_ACTION_CAP if chunk is None else int(chunk)
        self.obs_source = obs_source_of(self.cfg)
        self.stop_requires_visible = stop_requires_visible_of(self.cfg)
        self.stop_max_dist = stop_max_dist_of(self.cfg)
        self.stop_policy = stop_policy_of(self.cfg)
        self.stop_near_tau = stop_near_tau_of(self.cfg)
        self.explore_sample = explore_sample_of(self.cfg)
        self.explore_temp = explore_temp_of(self.cfg)
        self._episode_seed = None
        self._rng = random.Random(0)
        self.encode_calls = 0
        self._tokens = None
        self._scene = None
        self._last_a = None
        self._last_frame = None
        self._gt_obs = None
        self._s = TrackerState(max_steps=STEPS_NORM)
        self._prev_actions: list[str] = []
        self.n_stop_blocked = 0
        self.n_stop_blocked_not_visible = 0
        self.n_stop_blocked_conditions = 0
        self.n_stop_blocked_too_far = 0
        self.last_info: dict = {}
        self.model = None
        self.encoder = None
        self.spec = spec_from_cfg(self.cfg)
        self.bearings = patch_bearings(self.spec["grid"][0], self.spec["grid"][1], self.cfg)
        self._ignore_hw = self._hud_ignore_map()
        try:
            import torch
        except ImportError:
            torch = None
        self._torch = torch
        n_th = torch_threads_of(self.cfg)
        self.torch_threads = n_th
        if torch is not None and n_th > 0:
            torch.set_num_threads(n_th)
        ckpt = nav.get("checkpoint")
        payload = None
        if ckpt is not None:
            if torch is None:
                raise RuntimeError("c2nav checkpoint requires torch")
            payload = torch.load(ckpt, map_location="cpu")
            self.model, self.spec = load_net(payload)
            meta = payload.get("meta") if payload.get("meta") is not None else {}
            if stop_tau is None:
                chosen_stop = meta.get("chosen_stop_tau")
                if chosen_stop is None:
                    chosen_stop = meta.get("chosen_tau")
                if chosen_stop is not None:
                    self.tau = float(chosen_stop)
            if nav.get("vis_tau") is None and meta.get("chosen_vis_tau") is not None:
                self.spec["vis_tau"] = float(meta["chosen_vis_tau"])
            self.bearings = patch_bearings(self.spec["grid"][0], self.spec["grid"][1], self.cfg)
            if self.spec.get("vis_tau") is None:
                self.spec["vis_tau"] = 0.5
            self.spec["obs_source"] = self.obs_source
            self._ignore_hw = self._hud_ignore_map()
        if self.obs_source != OBS_GT:
            from lfnav.perception.encoder import FakeEncoder, load_encoder

            enc_name = nav.get("encoder")
            if enc_name is None:
                enc_name = (self.cfg.get("perception") or {}).get("encoder")
            if enc_name is None:
                enc_name = "fake"
            if self.cfg.get("dry_run") or enc_name in ("fake", "dry-run") or torch is None:
                self.encoder = FakeEncoder(dim=int(self.spec["dim"]), grid=tuple(self.spec["grid"]))
            else:
                self.encoder = load_encoder(enc_name, self.cfg)
        if torch is not None and self.model is None:
            self.model = C2NavNet(self.spec)
            self.model.eval()
        self.reset()

    def _hud_ignore_map(self) -> np.ndarray:
        from lfnav.geom import patch_grid
        from lfnav.gt.camera import patch_ignore

        hw = list(self.cfg.get("image_hw") or [224, 364])
        patch = int(self.cfg.get("patch") or 14)
        n_h, n_w = patch_grid(int(hw[0]), int(hw[1]), patch)
        ignored = patch_ignore(int(hw[0]), int(hw[1]), n_h, n_w, self.cfg.get("camera_calib") or {})
        return np.asarray(ignored, dtype=np.float32)

    def reset(self) -> None:
        self._tokens = None
        self._scene = None
        self._last_a = None
        self._last_frame = None
        self._gt_obs = None
        self._s = TrackerState(max_steps=STEPS_NORM)
        self._prev_actions = []
        self.n_stop_blocked = 0
        self.n_stop_blocked_not_visible = 0
        self.n_stop_blocked_conditions = 0
        self.n_stop_blocked_too_far = 0
        self.encode_calls = 0
        self.time_encode = 0.0
        self.time_policy = 0.0
        self.last_info = {}
        self._episode_seed = None
        self._rng = random.Random(self._seed_from_cfg())

    def _seed_from_cfg(self) -> int:
        raw = self.cfg.get("seed")
        if raw is None:
            return 0
        return int(raw)

    def _episode_seed_from_input(self, policy_input: PolicyInput):
        mem = policy_input.memory if policy_input.memory is not None else {}
        task = policy_input.task if policy_input.task is not None else {}
        ep = task.get("episode") if isinstance(task.get("episode"), dict) else {}
        for raw in (mem.get("seed"), task.get("seed"), ep.get("seed"), self.cfg.get("seed")):
            if raw is not None:
                return int(raw)
        return None

    def _ensure_episode_rng(self, policy_input: PolicyInput) -> None:
        seed = self._episode_seed_from_input(policy_input)
        if seed is None:
            return
        if self._episode_seed != seed:
            self._episode_seed = seed
            self._rng = random.Random(seed)

    def gt_obs_from_scene(self, scene: dict, target: str, dist_num) -> TargetObs:
        objects = list(self.spec["objects"])
        t_i = objects.index(target) if target in objects else 0
        grid = tuple(self.spec["grid"])
        pool = int(self.spec["token_pool"] if self.spec.get("token_pool") is not None else 2)
        present, _log_d, ignore = patch_targets(scene or {}, objects, pool, grid)
        obs = gt_target_obs(present, ignore, t_i, self.bearings, dist_num)
        patches = (scene or {}).get("patches") or {}
        raw = np.asarray(patches.get(target) or present[t_i], np.float32)
        raw_ig = (scene or {}).get("ignore")
        bel = ((scene or {}).get("objects") or {}).get(target) if isinstance((scene or {}).get("objects"), dict) else None
        dist_mass = (bel or {}).get("distance") if isinstance(bel, dict) else None
        return attach_lcr_dist(
            obs, raw, raw_ig if raw_ig is not None else ignore, dist_num, self.cfg, dist_mass=dist_mass,
        )

    def features(self, frame, force: bool = False) -> dict:
        from lfnav.data.demo_format import preprocess_frame
        from lfnav.perception.joint_features import encode_joint_tokens

        if self.obs_source == OBS_GT:
            return {"tokens": None, "obs_source": OBS_GT}
        if self.encoder is None:
            raise RuntimeError("c2nav pred path requires an encoder")
        if not force and self._tokens is not None:
            return {"tokens": self._tokens}
        self.encode_calls += 1
        t0 = time.perf_counter()
        rgb = frame
        if isinstance(frame, dict):
            from lfnav.env.frames import rgb_hwc

            rgb = rgb_hwc(frame, self.cfg)
            if rgb is None:
                rgb = frame.get("rgb")
        rgb = preprocess_frame(rgb)
        pool = int(self.spec["token_pool"] if self.spec.get("token_pool") is not None else 1)
        encoded = encode_joint_tokens(rgb, self.encoder, pool)
        tokens = np.asarray(encoded["tokens"], dtype=np.float32)
        gh, gw = int(self.spec["grid"][0]), int(self.spec["grid"][1])
        if tokens.ndim == 3 and (int(tokens.shape[0]), int(tokens.shape[1])) != (gh, gw) and gh > 0:
            from lfnav.perception.encoder import pool_patch_tokens

            kh = int(tokens.shape[0]) // gh
            tokens = pool_patch_tokens(tokens, max(int(kh), 1))
        encoded["tokens"] = tokens
        self._tokens = tokens
        self.time_encode += time.perf_counter() - t0
        return encoded

    def numeric_distance(self, target: str):
        if self.obs_source == OBS_GT:
            obs = self._gt_obs
            if obs is None and self._scene is not None:
                dist = None
                if self._last_frame is not None:
                    from lfnav.gt.initial import privileged_ray_distance

                    dist = privileged_ray_distance(self._last_frame, self.cfg, target)
                obs = self.gt_obs_from_scene(self._scene.to_dict(), target, dist)
            if obs is None or not obs.visible or obs.log_distance is None:
                return None
            return math.exp(float(obs.log_distance))
        if self._last_a is None or self._torch is None:
            return None
        objects = list(self.spec["objects"])
        if target not in objects:
            return None
        vis_tau = float(self.spec["vis_tau"] if self.spec.get("vis_tau") is not None else 0.5)
        idx = objects.index(target)
        vis_logit = self._last_a.get("vis_logit")
        target_log = self._last_a.get("target_log_dist")
        if vis_logit is None or target_log is None:
            return None
        if not vis_is_positive(float(vis_logit[0, idx].detach().cpu()), vis_tau):
            return None
        return math.exp(float(target_log[0, idx].detach().cpu()))

    def perceive(self, frame):
        from lfnav.gt.scene_state import scene_state_from_obs
        from lfnav.types import SceneState

        if self.obs_source == OBS_GT:
            if hasattr(frame, "to_dict") and hasattr(frame, "patches"):
                self._last_frame = None
                self._scene = frame
            elif isinstance(frame, dict) and (frame.get("patches") is not None or frame.get("ignore") is not None):
                self._last_frame = None
                self._scene = SceneState.from_dict(frame)
            elif isinstance(frame, dict):
                self._last_frame = frame
                self._scene = scene_state_from_obs(frame, self.cfg)
            else:
                raise TypeError("c2nav obs_source=gt expects a full observation or a GT scene dict")
            return self._scene

        self.features(frame, force=True)
        if self.model is None or self._torch is None or self._tokens is None:
            self._scene = SceneState()
            return self._scene
        torch = self._torch
        with torch.no_grad():
            tokens = torch.from_numpy(np.asarray(self._tokens, np.float32)).unsqueeze(0)
            a = self.model.perception(tokens)
        self._last_a = a
        self._scene = scene_from_perception(
            a, self.spec, self.bearings, ignore=self._ignore_hw, cfg=self.cfg,
        )
        return self._scene

    def act(self, policy_input: PolicyInput) -> PolicyOutput:
        from lfnav.gt.initial import privileged_ray_distance
        from lfnav.types import PolicyOutput

        t_pol = time.perf_counter()
        self._ensure_episode_rng(policy_input)
        mem = policy_input.memory if policy_input.memory is not None else {}
        pos = [0.0, 0.0, 0.0] if mem.get("pos") is None else list(mem.get("pos"))
        yaw = 0.0 if mem.get("yaw") is None else float(mem.get("yaw"))
        match = policy_input.match or {}
        target = str((policy_input.task or {}).get("target") or "")
        objects = list(self.spec["objects"])
        t_i = objects.index(target) if target in objects else 0
        obs = TargetObs(visible=False)
        geom_np = geom_from_obs(obs)
        mode_i = turn_i = fwd_i = 0
        p_stop = 0.0
        p_near = None
        turn_logits = None
        vis_tau = float(self.spec["vis_tau"] if self.spec.get("vis_tau") is not None else 0.5)

        if self.obs_source == OBS_GT:
            scene_d = {} if self._scene is None else self._scene.to_dict()
            dist_num = None
            if self._last_frame is not None:
                dist_num = privileged_ray_distance(self._last_frame, self.cfg, target)
            elif (policy_input.scene or {}).get("target_distance_num") is not None:
                dist_num = policy_input.scene.get("target_distance_num")
            obs = self.gt_obs_from_scene(scene_d, target, dist_num)
            self._gt_obs = obs
            geom_np = geom_from_obs(obs)
        elif self.model is not None and self._torch is not None:
            torch = self._torch
            a = self._last_a
            if a is None and self._tokens is not None:
                device = next(self.model.parameters()).device
                with torch.no_grad():
                    tokens = torch.from_numpy(np.asarray(self._tokens, np.float32)).unsqueeze(0).to(device)
                    a = self.model.perception(tokens)
                self._last_a = a
            if a is not None:
                vis_logit = a["vis_logit"][0, t_i]
                present_np = a["present_p"][0].detach().cpu().numpy()
                dist_np = a["log_dist"][0].detach().cpu().numpy()
                obs = obs_from_present_head(
                    present_np, dist_np, t_i, self.bearings, float(vis_logit.detach().cpu()),
                    vis_tau, ignore=self._ignore_hw, patch_min=PATCH_PRESENT_MIN,
                )
                bin_prob = softmax_last(a["dist_bin_logit"][0, t_i].detach().cpu().numpy())
                p_near = float(bin_prob[DIST_BIN_NAMES.index("near")])
                dist_mass = {DIST_BIN_NAMES[j]: float(bin_prob[j]) for j in range(len(DIST_BIN_NAMES))}
                if obs.visible and a.get("target_log_dist") is not None:
                    obs.log_distance = float(a["target_log_dist"][0, t_i].detach().cpu())
                obs = attach_lcr_dist(
                    obs, present_np[t_i], self._ignore_hw,
                    None if obs.log_distance is None else math.exp(float(obs.log_distance)),
                    self.cfg, dist_mass=dist_mass,
                )
                geom_np = geom_from_obs(obs)

        self._s = update(self._s, pos, yaw, self._prev_actions, obs, max_steps=self._s.max_steps)

        if self.model is not None and self._torch is not None:
            torch = self._torch
            device = next(self.model.parameters()).device
            with torch.no_grad():
                geom_t = torch.from_numpy(np.asarray(geom_np, np.float32)).unsqueeze(0).to(device)
                s_t = torch.from_numpy(encode_s(self._s, sector8=mem.get("sector8"))).unsqueeze(0).to(device)
                cond_t = torch.from_numpy(encode_match(match, objects)).unsqueeze(0).to(device)
                out = self.model.policy(geom_t, s_t, cond_t)
            mode_i = int(out["mode"][0].argmax().item())
            turn_logits = out["turn"][0].detach().cpu().numpy()
            fwd_i = int(out["forward"][0].argmax().item())
            p_stop = float(torch.sigmoid(out["stop"][0]).item())

        feat = self._s.features()
        mode_name = MODE_NAMES[mode_i] if 0 <= mode_i < len(MODE_NAMES) else str(mode_i)
        is_explore = mode_name == "explore"
        turn_sampled = bool(is_explore and self.explore_sample)
        if turn_logits is not None:
            if turn_sampled:
                turn_i = sample_turn_index(turn_logits, self._rng, self.explore_temp)
            else:
                turn_i = int(np.argmax(np.asarray(turn_logits, dtype=np.float64).reshape(-1)))
            topk_temp = self.explore_temp if is_explore else 1.0
            turn_topk = top_k_turn_probs(turn_logits, 3, topk_temp)
        else:
            turn_topk = []
        turn_deg = turn_center(turn_i)
        forward = FORWARD_BINS[fwd_i] if 0 <= fwd_i < len(FORWARD_BINS) else 0
        dist = None if (not obs.visible or obs.log_distance is None) else math.exp(float(obs.log_distance))
        bearing = 0.0 if (not obs.visible or obs.bearing_deg is None) else float(obs.bearing_deg)
        info = {
            "mode": mode_name,
            "waypoint": {"turn_deg": turn_deg, "forward": forward},
            "turn_index": turn_i,
            "turn_deg": turn_deg,
            "turn_sampled": turn_sampled,
            "turn_topk": turn_topk,
            "explore_sample": self.explore_sample,
            "explore_temp": self.explore_temp,
            "p_stop": p_stop,
            "p_near": p_near,
            "stop_near_tau": None if self.obs_source == OBS_GT else self.stop_near_tau,
            "visible": bool(obs.visible),
            "bearing": bearing,
            "dist": dist,
            "obs_source": self.obs_source,
            "pos": list(pos),
            "yaw": float(yaw),
            "belief": {
                "bearing_sin": feat["bearing_sin"],
                "bearing_cos": feat["bearing_cos"],
                "log_distance": feat["log_distance"],
                "confidence": feat["confidence"],
            },
            "tau": self.tau,
        }
        cap = DECISION_ACTION_CAP if self.chunk_max is None else min(int(self.chunk_max), DECISION_ACTION_CAP)
        if cap < 1:
            cap = DECISION_ACTION_CAP
        actions = invert_waypoint(turn_i, fwd_i)[:cap] or ["noop"]
        self._prev_actions = list(actions)
        info["action_chunk"] = actions
        gate = infer_stop_gate(
            p_stop,
            self.tau,
            obs.visible,
            match,
            requires_visible=self.stop_requires_visible,
            dist=dist,
            max_dist=self.stop_max_dist,
            policy=self.stop_policy,
            p_near=None if self.obs_source == OBS_GT else p_near,
            near_tau=None if self.obs_source == OBS_GT else self.stop_near_tau,
        )
        info["stop_gate"] = gate
        info["stop_policy"] = self.stop_policy
        info["stop_max_dist"] = self.stop_max_dist
        if gate == STOP_GATE_NOT_VISIBLE:
            self.n_stop_blocked_not_visible += 1
            self.n_stop_blocked += 1
        elif gate == STOP_GATE_CONDITIONS:
            self.n_stop_blocked_conditions += 1
            self.n_stop_blocked += 1
        elif gate == STOP_GATE_TOO_FAR:
            self.n_stop_blocked_too_far += 1
            self.n_stop_blocked += 1
        self.last_info = info
        self.time_policy += time.perf_counter() - t_pol
        if gate == STOP_GATE_STOP:
            return PolicyOutput(stop=True, info=info)
        return PolicyOutput(action_chunk=actions, info=info)
