"""C2+GT samples from expert demos. One row per decision; split by episode seed."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lfnav.data.demo_format import read_episode
from lfnav.policy.c1 import sector8_counts
from lfnav.types import PolicyInput

DIRECTIONS = ("left", "center", "right")
DISTANCES = ("near", "mid", "far")
STATUSES = ("found", "not_found")
PAD_ACTION = "noop"
AGE_CAP = 100.0
RECENT_K = 4
VAL_FRAC = 0.15


def action_names(cfg: dict | None) -> list[str]:
    actions = (cfg or {}).get("actions") or {}
    names = [str(n) for n in (actions.get("names") or [])]
    for key in (actions.get("vectors") or {}):
        if key not in names:
            names.append(str(key))
    if PAD_ACTION not in names:
        names.append(PAD_ACTION)
    return names


def _one_hot(names: list[str], key) -> list[float]:
    row = [0.0] * len(names)
    if key is None:
        return row
    try:
        row[names.index(str(key))] = 1.0
    except ValueError:
        pass
    return row


def _belief(scene: dict, name: str) -> dict:
    raw = ((scene or {}).get("objects") or {}).get(name) or {}
    if not isinstance(raw, dict):
        return {}
    return raw


@dataclass
class FeatureSpec:
    objects: list[str]
    biomes: list[str]
    times: list[str]
    weathers: list[str]
    brightness: list[str]
    actions: list[str]
    horizon: int = 4
    recent_k: int = RECENT_K

    @classmethod
    def from_cfg(cls, cfg: dict | None = None) -> "FeatureSpec":
        cfg = cfg or {}
        labels = cfg.get("labels") or {}
        pol = cfg.get("policy") or {}
        c2 = cfg.get("c2") or {}
        return cls(
            objects=list(labels.get("objects") or ["tree", "grass", "water", "pig", "cow", "sheep", "chicken", "sand"]),
            biomes=list(labels.get("biomes") or ["forest", "plains", "desert"]),
            times=list(labels.get("time") or ["day", "night"]),
            weathers=list(labels.get("weather") or ["clear", "rain"]),
            brightness=list(labels.get("brightness") or ["sufficient", "insufficient"]),
            actions=action_names(cfg),
            horizon=int(c2.get("chunk") or pol.get("chunk") or 4),
            recent_k=int(pol.get("c2_recent_k") or RECENT_K),
        )

    @property
    def scene_dim(self) -> int:
        per_obj = 1 + len(DIRECTIONS) + len(DISTANCES)
        return (
            len(self.objects) * per_obj
            + len(self.biomes)
            + len(self.times)
            + len(self.weathers)
            + len(self.brightness)
        )

    @property
    def aux_dim(self) -> int:
        return (
            len(self.objects)
            + len(self.objects)
            + len(self.biomes)
            + len(self.times)
            + len(self.weathers)
            + len(self.brightness)
            + 8
            + self.recent_k * len(self.actions)
            + 1
            + len(DIRECTIONS)
            + len(DISTANCES)
        )

    @property
    def dim(self) -> int:
        per_obj = 1 + len(DIRECTIONS) + len(DISTANCES)
        return (
            len(self.objects) * per_obj
            + len(self.objects)
            + len(self.objects)
            + len(self.biomes)
            + len(self.times)
            + len(self.weathers)
            + len(self.brightness)
            + len(STATUSES)
            + len(DIRECTIONS)
            + len(DISTANCES)
            + 8
            + self.recent_k * len(self.actions)
            + 1
            + len(DIRECTIONS)
            + len(DISTANCES)
        )


def encode_object(belief: dict) -> list[float]:
    direction = belief.get("direction") or {}
    distance = belief.get("distance") or {}
    return (
        [float(belief.get("present") or 0.0)]
        + [float(direction.get(name) or 0.0) for name in DIRECTIONS]
        + [float(distance.get(name) or 0.0) for name in DISTANCES]
    )


def encode_last_seen(last: dict | None, env_step: int) -> list[float]:
    if not last:
        return [1.0] + [0.0] * (len(DIRECTIONS) + len(DISTANCES))
    age = max(0.0, float(env_step) - float(last.get("step") or 0))
    return (
        [min(age, AGE_CAP) / AGE_CAP]
        + _one_hot(list(DIRECTIONS), last.get("direction"))
        + _one_hot(list(DISTANCES), last.get("distance"))
    )


def encode_aux(
    task: dict,
    sector8,
    recent_actions,
    last_seen: dict | None,
    env_step: int,
    spec: FeatureSpec,
) -> np.ndarray:
    """C2 side features without the GT scene or matcher output."""
    cond = (task or {}).get("conditions") or {}
    parts: list[float] = []
    parts.extend(_one_hot(spec.objects, (task or {}).get("target")))
    near = [0.0] * len(spec.objects)
    for name in list(cond.get("near") or []):
        if name in spec.objects:
            near[spec.objects.index(name)] = 1.0
    parts.extend(near)
    parts.extend(_one_hot(spec.biomes, cond.get("biome")))
    parts.extend(_one_hot(spec.times, cond.get("time")))
    parts.extend(_one_hot(spec.weathers, cond.get("weather")))
    parts.extend(_one_hot(spec.brightness, cond.get("brightness")))
    sectors = [float(x) for x in list(sector8 or [])[:8]]
    sectors += [0.0] * (8 - len(sectors))
    total = sum(sectors)
    if total > 0:
        sectors = [x / total for x in sectors]
    parts.extend(sectors)
    recent = list(recent_actions or [])[-spec.recent_k :]
    recent = [None] * (spec.recent_k - len(recent)) + recent
    for name in recent:
        parts.extend(_one_hot(spec.actions, name))
    parts.extend(encode_last_seen(last_seen, env_step))
    vector = np.asarray(parts, dtype=np.float32)
    if vector.shape[0] != spec.aux_dim:
        raise RuntimeError(f"aux dim {vector.shape[0]} != spec {spec.aux_dim}")
    return vector


def encode_features(
    scene: dict,
    task: dict,
    match: dict,
    sector8,
    recent_actions,
    last_seen: dict | None,
    env_step: int,
    spec: FeatureSpec,
) -> np.ndarray:
    cond = (task or {}).get("conditions") or {}
    parts: list[float] = []
    for name in spec.objects:
        parts.extend(encode_object(_belief(scene, name)))
    parts.extend(_one_hot(spec.objects, (task or {}).get("target")))
    near = [0.0] * len(spec.objects)
    for name in list(cond.get("near") or []):
        if name in spec.objects:
            near[spec.objects.index(name)] = 1.0
    parts.extend(near)
    parts.extend(_one_hot(spec.biomes, cond.get("biome")))
    parts.extend(_one_hot(spec.times, cond.get("time")))
    parts.extend(_one_hot(spec.weathers, cond.get("weather")))
    parts.extend(_one_hot(spec.brightness, cond.get("brightness")))
    parts.extend(_one_hot(list(STATUSES), (match or {}).get("status") or "not_found"))
    parts.extend(_one_hot(list(DIRECTIONS), (match or {}).get("target_direction")))
    parts.extend(_one_hot(list(DISTANCES), (match or {}).get("target_distance")))
    sectors = [float(x) for x in list(sector8 or [])[:8]]
    sectors += [0.0] * (8 - len(sectors))
    total = sum(sectors)
    if total > 0:
        sectors = [x / total for x in sectors]
    parts.extend(sectors)
    recent = list(recent_actions or [])[-spec.recent_k:]
    recent = [None] * (spec.recent_k - len(recent)) + recent
    for name in recent:
        parts.extend(_one_hot(spec.actions, name))
    parts.extend(encode_last_seen(last_seen, env_step))
    vector = np.asarray(parts, dtype=np.float32)
    if vector.shape[0] != spec.dim:
        raise RuntimeError(f"feature dim {vector.shape[0]} != spec {spec.dim}")
    return vector


def encode_policy_input(policy_input: PolicyInput, spec: FeatureSpec, last_seen: dict | None = None) -> np.ndarray:
    mem = policy_input.memory or {}
    recent = policy_input.recent_actions or mem.get("recent_actions") or []
    env_step = int(mem.get("env_steps") or 0)
    seen = last_seen if last_seen is not None else mem.get("last_seen")
    sectors = mem.get("sector8")
    if sectors is None:
        sectors = sector8_counts(mem.get("local_grid") or [])
    return encode_features(
        policy_input.scene or {},
        policy_input.task or {},
        policy_input.match or {},
        sectors,
        recent,
        seen,
        env_step,
        spec,
    )


def pad_chunk(chunk, spec: FeatureSpec) -> list[int]:
    names = spec.actions
    pad = names.index(PAD_ACTION) if PAD_ACTION in names else len(names) - 1
    out = []
    for name in list(chunk or [])[: spec.horizon]:
        try:
            out.append(names.index(str(name)))
        except ValueError:
            out.append(pad)
    while len(out) < spec.horizon:
        out.append(pad)
    return out


def _visible(match: dict) -> bool:
    if (match or {}).get("target_direction") is not None:
        return True
    return (match or {}).get("status") == "found"


def success_root(demos) -> Path:
    path = Path(demos)
    if (path / "success").is_dir():
        return path / "success"
    return path


def iter_success_episodes(demos) -> list[Path]:
    root = success_root(demos)
    return sorted(p.parent for p in root.glob("**/record.json"))


def _config_stem(name) -> str | None:
    if not name:
        return None
    return Path(str(name)).stem


def _episode_key(record: dict, folder: Path) -> tuple[str, str, int]:
    task_id = str(record.get("task_name") or record.get("task") or folder.parts[-3])
    difficulty = str(record.get("difficulty") or folder.parts[-2])
    seed = int(record.get("seed") if record.get("seed") is not None else folder.name)
    return task_id, difficulty, seed


def _is_main_source(record: dict, folder: Path) -> bool:
    stem = _config_stem(record.get("config_name"))
    if stem == "main":
        return True
    return any(str(p) == "demos_main" or str(p).startswith("demos_main_") for p in Path(folder).parts)


def _is_legacy_source(record: dict) -> bool:
    stem = _config_stem(record.get("config_name"))
    return stem is None or stem == "demo_dynamic"


def demo_roots(demos, cfg: dict | None = None) -> list[Path]:
    if demos is None:
        return [Path(p) for p in ((cfg or {}).get("c2") or {}).get("demo_dirs") or []]
    if isinstance(demos, (list, tuple)):
        return [Path(p) for p in demos]
    return [Path(demos)]


def select_training_episodes(folders: list[Path], cfg: dict | None = None) -> tuple[list[Path], dict]:
    """Keep demos that match the C2 training filter. Does not delete files."""
    c2 = (cfg or {}).get("c2") or {}
    exclude = {str(t) for t in (c2.get("exclude_tasks") or ["4-3"])}
    legacy_tasks = {str(t) for t in (c2.get("legacy_tasks") or [])}
    stats = {
        "seen": 0,
        "not_success": 0,
        "difficulty_unmet": 0,
        "exclude_tasks": 0,
        "main_not_peaceful": 0,
        "main_hostile": 0,
        "legacy_not_allowed": 0,
        "duplicate": 0,
        "kept_main": 0,
        "kept_legacy": 0,
        "kept": 0,
    }
    ranked: dict[tuple[str, str, int], tuple[int, Path]] = {}
    for folder in folders:
        packed = read_episode(folder)
        record = packed["record"]
        stats["seen"] += 1
        task_id, difficulty, seed = _episode_key(record, Path(folder))
        if not record.get("success"):
            stats["not_success"] += 1
            continue
        if record.get("difficulty_unmet"):
            stats["difficulty_unmet"] += 1
            continue
        if task_id in exclude:
            stats["exclude_tasks"] += 1
            continue
        main = _is_main_source(record, Path(folder))
        if main:
            if not record.get("peaceful_sent", record.get("peaceful_ok", False)):
                stats["main_not_peaceful"] += 1
            if int(record.get("hostile_seen") or 0) != 0:
                stats["main_hostile"] += 1
                continue
            rank = 1
        elif _is_legacy_source(record):
            if task_id not in legacy_tasks:
                stats["legacy_not_allowed"] += 1
                continue
            rank = 0
        else:
            stats["legacy_not_allowed"] += 1
            continue
        key = (task_id, difficulty, seed)
        prev = ranked.get(key)
        if prev is not None:
            stats["duplicate"] += 1
            if prev[0] >= rank:
                continue
        ranked[key] = (rank, Path(folder))
    kept = []
    for rank, folder in ranked.values():
        kept.append(folder)
        if rank == 1:
            stats["kept_main"] += 1
        else:
            stats["kept_legacy"] += 1
    stats["kept"] = len(kept)
    for key, n in stats.items():
        print(f"c2 filter {key}={n}")
    return kept, stats


def load_task(record: dict, folder: Path, cfg: dict | None = None) -> dict:
    if record.get("task") and isinstance(record["task"], dict) and record["task"].get("target"):
        return dict(record["task"])
    task_id = record.get("task_name") or record.get("task") or folder.parts[-3]
    root = Path(__file__).resolve().parents[2]
    spec_path = root / "lfnav" / "tasks" / "specs" / f"{task_id}.json"
    if spec_path.is_file():
        return json.loads(spec_path.read_text())
    return {"id": str(task_id), "target": record.get("target") or "pig", "conditions": {}}


@dataclass
class Sample:
    x: np.ndarray
    y: np.ndarray
    used_wide: bool
    weight: float
    seed: int
    episode_id: str
    task: str


@dataclass
class C2Split:
    train: list[Sample]
    val: list[Sample]
    spec: FeatureSpec
    used_wide_frac: float
    n_episodes: int
    val_seeds: list[int]
    filter_stats: dict = None


def samples_from_episode(folder, spec: FeatureSpec, wide_weight: float = 1.0) -> list[Sample]:
    packed = read_episode(folder)
    record = packed["record"]
    task = load_task(record, Path(folder))
    seed = int(record.get("seed") if record.get("seed") is not None else Path(folder).name)
    task_id = str(record.get("task_name") or task.get("id") or Path(folder).parts[-3])
    difficulty = str(record.get("difficulty") or Path(folder).parts[-2])
    episode_id = f"{task_id}/{difficulty}/{seed}"
    last = None
    rows = []
    for step in packed["steps"]:
        chunk = step.get("action_chunk")
        if not chunk:
            continue
        match = step.get("match") or {}
        env_step = int(step.get("env_step") or 0)
        if _visible(match):
            last = {
                "step": env_step,
                "direction": match.get("target_direction"),
                "distance": match.get("target_distance"),
            }
        mem = step.get("memory") or {}
        used = bool(step.get("used_wide"))
        x = encode_features(
            step.get("scene") or {},
            task,
            match,
            mem.get("sector8"),
            mem.get("recent_actions"),
            last,
            env_step,
            spec,
        )
        y = np.asarray(pad_chunk(chunk, spec), dtype=np.int64)
        weight = float(wide_weight) if used else 1.0
        rows.append(Sample(x, y, used, weight, seed, episode_id, task_id))
    return rows


def split_by_seed(episodes: list[Path], val_frac: float = VAL_FRAC, rng_seed: int = 0) -> tuple[list[Path], list[Path], list[int]]:
    by_seed: dict[int, list[Path]] = {}
    for folder in episodes:
        packed = read_episode(folder)
        seed = int(packed["record"].get("seed") if packed["record"].get("seed") is not None else Path(folder).name)
        by_seed.setdefault(seed, []).append(folder)
    seeds = sorted(by_seed)
    rng = random.Random(int(rng_seed))
    rng.shuffle(seeds)
    n_val = int(math.floor(len(seeds) * float(val_frac) + 1e-9))
    if len(seeds) >= 2:
        n_val = max(1, n_val)
        n_val = min(n_val, len(seeds) - 1)
    else:
        n_val = 0
    val_seeds = sorted(seeds[:n_val])
    val_set = set(val_seeds)
    train, val = [], []
    for seed, folders in by_seed.items():
        dest = val if seed in val_set else train
        dest.extend(folders)
    return train, val, val_seeds


def load_c2_split(demos, cfg: dict | None = None, val_frac: float = VAL_FRAC, rng_seed: int = 0) -> C2Split:
    cfg = cfg or {}
    spec = FeatureSpec.from_cfg(cfg)
    wide_weight = float((cfg.get("c2") or {}).get("wide_weight", 1.0))
    folders = []
    for root in demo_roots(demos, cfg):
        folders.extend(iter_success_episodes(root))
    episodes, filter_stats = select_training_episodes(folders, cfg)
    train_eps, val_eps, val_seeds = split_by_seed(episodes, val_frac=val_frac, rng_seed=rng_seed)
    train, val = [], []
    for folder in train_eps:
        train.extend(samples_from_episode(folder, spec, wide_weight=wide_weight))
    for folder in val_eps:
        val.extend(samples_from_episode(folder, spec, wide_weight=wide_weight))
    all_rows = train + val
    n_wide = sum(1 for row in all_rows if row.used_wide)
    frac = n_wide / len(all_rows) if all_rows else 0.0
    print(f"used_wide {n_wide}/{len(all_rows)} = {frac:.4f}  episodes={len(episodes)} train={len(train)} val={len(val)}")
    return C2Split(train, val, spec, frac, len(episodes), val_seeds, filter_stats)


def class_weights(samples: list[Sample], n_actions: int) -> np.ndarray:
    counts = np.zeros(n_actions, dtype=np.float64)
    for row in samples:
        for idx in row.y.tolist():
            counts[int(idx)] += 1.0
    inv = 1.0 / np.maximum(counts, 1.0)
    inv *= n_actions / inv.sum()
    return inv.astype(np.float32)
