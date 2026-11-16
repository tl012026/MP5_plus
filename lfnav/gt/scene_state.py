"""Build a ground-truth SceneState from aligned rays and configured global fields.

Block and mob name fragments come from lfnav/labels.yaml. Biome ids are mapped
through obs_keys.biome_id_to_name. Time uses sun_brightness vs time_day_min;
brightness uses light_level vs brightness_min_light. Unset thresholds leave the
field unknown instead of a silent miss.
"""

from __future__ import annotations

import logging

from lfnav.aggregate import aggregate_object, column_direction, distance_bin
from lfnav.geom import patch_grid
from lfnav.gt.camera import patch_ignore
from lfnav.obsutil import lookup, scalar
from lfnav.types import SceneState

log = logging.getLogger("lfnav.gt.scene_state")

# Minecraft 1.11 ids used as candidates until calibrate_env_state.py measures them.
CANDIDATE_BIOME_IDS = {
    "plains": 1,
    "desert": 2,
    "forest": 4,
    "taiga": 5,
    "swampland": 6,
    "jungle": 21,
    "birch_forest": 27,
    "roofed_forest": 29,
    "savanna": 35,
}


OCCLUDER_NAMES = ("tallgrass", "tall_grass", "double_plant")


def _occluder_key(raw) -> str:
    text = str(raw).strip().lower()
    if text.startswith("minecraft:"):
        text = text[len("minecraft:"):]
    return text.replace(" ", "_")


def occluder_names_of(cfg: dict | None) -> tuple:
    labels = (cfg or {}).get("labels") or {}
    raw = labels.get("occluder_names")
    if raw is None:
        return OCCLUDER_NAMES
    return tuple(_occluder_key(name) for name in raw)


def is_occluder_name(raw, names=None) -> bool:
    """Exact match after lowercasing, stripping minecraft:, and turning spaces into underscores."""
    if raw is None:
        return False
    allowed = OCCLUDER_NAMES if names is None else names
    return _occluder_key(raw) in {_occluder_key(name) for name in allowed}


def split_ray_hits(hits, names=None) -> tuple[list, list]:
    """Sort one subray. The closest hit is visible. An entity behind only occluders is occluded."""
    cleaned = []
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        raw = hit.get("raw")
        if raw is None:
            raw = hit.get("label")
        dist = hit.get("distance")
        try:
            dist_f = None if dist is None else float(dist)
        except (TypeError, ValueError):
            dist_f = None
        if raw is None and dist_f is None:
            continue
        item = {"raw": raw, "distance": dist_f}
        if hit.get("kind") is not None:
            item["kind"] = hit.get("kind")
        cleaned.append(item)
    cleaned.sort(key=lambda item: (item["distance"] is None, float(item["distance"] or 0.0)))
    if not cleaned:
        return [], []
    visible = [dict(cleaned[0])]
    occluded = []
    for idx in range(1, len(cleaned)):
        hit = cleaned[idx]
        if str(hit.get("kind") or "") != "entity":
            continue
        if hit["distance"] is None:
            continue
        front = [
            prev for prev in cleaned[:idx]
            if prev["distance"] is not None and float(prev["distance"]) < float(hit["distance"])
        ]
        if front and all(is_occluder_name(prev.get("raw"), names) for prev in front):
            occluded.append(dict(hit))
    return visible, occluded


def _ray_visibility(ray: dict, names=None) -> tuple[list, list]:
    """Use stored per-subray classes when convert_rays wrote them. Otherwise split hits."""
    if isinstance(ray, dict) and ("visible_hits" in ray or "occluded_hits" in ray):
        return list(ray.get("visible_hits") or []), list(ray.get("occluded_hits") or [])
    raw_hits = ray.get("hits") if isinstance(ray, dict) else None
    if isinstance(raw_hits, list) and raw_hits:
        return split_ray_hits(raw_hits, names)
    raw = ray.get("raw") if isinstance(ray, dict) else None
    if raw is None and isinstance(ray, dict):
        raw = ray.get("label")
    dist = ray.get("distance") if isinstance(ray, dict) else None
    return split_ray_hits([{"raw": raw, "distance": dist}], names)


def _hits_of(ray: dict) -> list[tuple]:
    raw_hits = ray.get("hits") if isinstance(ray, dict) else None
    if isinstance(raw_hits, list) and raw_hits:
        out = []
        for hit in raw_hits:
            if not isinstance(hit, dict):
                continue
            raw = hit.get("raw")
            if raw is None:
                raw = hit.get("label")
            dist = hit.get("distance")
            try:
                dist = None if dist is None else float(dist)
            except (TypeError, ValueError):
                dist = None
            if raw is None and dist is None:
                continue
            out.append((raw, dist))
        if out:
            return out
    raw = ray.get("raw")
    if raw is None:
        raw = ray.get("label")
    dist = ray.get("distance")
    try:
        dist = None if dist is None else float(dist)
    except (TypeError, ValueError):
        dist = None
    if raw is None and dist is None:
        return []
    return [(raw, dist)]


def _closer(prev, dist):
    if dist is None:
        return prev
    if prev is None or float(dist) < float(prev):
        return float(dist)
    return prev


def occluder_at_stop(scene, max_dist: float = 3.0) -> bool:
    """True when a non-HUD center-third cell hits tallgrass or double_plant within max_dist."""
    if hasattr(scene, "occluder"):
        occ = scene.occluder
        dist = scene.patch_dist
        ignore = scene.ignore
    else:
        data = scene or {}
        occ = data.get("occluder")
        dist = data.get("patch_dist")
        ignore = data.get("ignore")
    if not occ or not occ[0]:
        return False
    n_h = len(occ)
    n_w = len(occ[0])
    for i in range(n_h):
        for j in range(n_w):
            if column_direction(j, n_w) != "center":
                continue
            if ignore and i < len(ignore) and j < len(ignore[i]) and ignore[i][j]:
                continue
            try:
                marked = float(occ[i][j]) >= 0.5
            except (TypeError, ValueError):
                marked = False
            if not marked:
                continue
            cell = None if dist is None or i >= len(dist) or j >= len(dist[i]) else dist[i][j]
            try:
                cell = None if cell is None else float(cell)
            except (TypeError, ValueError):
                cell = None
            if cell is not None and cell <= float(max_dist):
                return True
    return False


def resolve_label(raw, aliases: dict, objects: list[str]) -> str | None:
    if raw is None:
        return None
    text = str(raw).lower()
    if text in objects:
        return text
    for name, fragments in (aliases or {}).items():
        for fragment in fragments:
            if fragment and fragment.lower() in text:
                return name
    return None


def _one_hot(value, vocab: list[str]) -> dict:
    if value is None:
        return {}
    text = str(value).lower()
    if text not in vocab:
        if "other" in vocab:
            text = "other"
        else:
            return {}
    return {name: 1.0 if name == text else 0.0 for name in vocab}


def mapped_biome(biome_id, mapping: dict | None) -> str | None:
    if biome_id is None or not mapping:
        return None
    if biome_id in mapping:
        return str(mapping[biome_id])
    key = str(biome_id)
    if key in mapping:
        return str(mapping[key])
    try:
        numbered = int(biome_id)
    except (TypeError, ValueError):
        return None
    if numbered in mapping:
        return str(mapping[numbered])
    if str(numbered) in mapping:
        return str(mapping[str(numbered)])
    return None


def biome_id_for_name(name: str, mapping: dict | None = None) -> int | None:
    text = str(name).lower()
    for key, value in (mapping or {}).items():
        if str(value).lower() == text:
            try:
                return int(key)
            except (TypeError, ValueError):
                continue
    return CANDIDATE_BIOME_IDS.get(text)


def _mark_unknown(unknown: list[str], field: str, message: str) -> None:
    if field not in unknown:
        unknown.append(field)
    log.debug(message)


def decode_biome(obs: dict, cfg: dict, unknown: list[str]) -> str | None:
    keys = cfg.get("obs_keys") or {}
    named = scalar(lookup(obs, keys.get("biome_name")))
    if named is not None:
        return str(named).lower()
    biome_id = scalar(lookup(obs, keys.get("biome_id")))
    if biome_id is None:
        if keys.get("biome_id") is not None or keys.get("biome_name") is not None:
            _mark_unknown(unknown, "biome", "biome id is missing from the observation; biome is unknown")
        return None
    mapping = keys.get("biome_id_to_name") or {}
    name = mapped_biome(biome_id, mapping)
    if name is None:
        _mark_unknown(
            unknown, "biome",
            f"biome_id {biome_id!r} is not in obs_keys.biome_id_to_name; biome is unknown",
        )
        return None
    return name.lower()


def decode_time(obs: dict, cfg: dict, unknown: list[str]) -> str | None:
    keys = cfg.get("obs_keys") or {}
    named = scalar(lookup(obs, keys.get("time_name")))
    if named is not None:
        return str(named).lower()
    raw = scalar(lookup(obs, keys.get("time_of_day")))
    thresh = cfg.get("time_day_min")
    if thresh is None:
        thresh = cfg.get("time_day_below")
    if raw is None:
        if keys.get("time_of_day") is not None:
            _mark_unknown(unknown, "time", "sun_brightness is missing from the observation; time is unknown")
        return None
    if thresh is None:
        _mark_unknown(
            unknown, "time",
            "time_day_min is unset; run lfnav/gt/calibrate_env_state.py and fill the threshold. time is unknown",
        )
        return None
    return "day" if float(raw) >= float(thresh) else "night"


def decode_weather(obs: dict, cfg: dict, unknown: list[str]) -> str | None:
    keys = cfg.get("obs_keys") or {}
    named = scalar(lookup(obs, keys.get("weather_name")))
    if named is not None:
        return str(named).lower()
    raw = scalar(lookup(obs, keys.get("weather_rain")))
    if raw is None:
        if keys.get("weather_rain") is not None:
            _mark_unknown(unknown, "weather", "is_raining is missing from the observation; weather is unknown")
        return None
    return "rain" if bool(raw) else "clear"


def decode_brightness(obs: dict, cfg: dict, unknown: list[str]) -> str | None:
    keys = cfg.get("obs_keys") or {}
    named = scalar(lookup(obs, keys.get("brightness_name")))
    if named is not None:
        return str(named).lower()
    raw = scalar(lookup(obs, keys.get("light_level")))
    thresh = cfg.get("brightness_min_light")
    if raw is None:
        if keys.get("light_level") is not None:
            _mark_unknown(unknown, "brightness", "light_level is missing from the observation; brightness is unknown")
        return None
    if thresh is None:
        _mark_unknown(
            unknown, "brightness",
            "brightness_min_light is unset; run lfnav/gt/calibrate_env_state.py and fill the threshold. brightness is unknown",
        )
        return None
    return "sufficient" if float(raw) >= float(thresh) else "insufficient"


def scene_state_from_obs(obs: dict, cfg: dict) -> SceneState:
    keys = cfg.get("obs_keys") or {}
    labels = cfg.get("labels") or {}
    objects = list(labels.get("objects") or [])
    aliases = labels.get("aliases") or {}
    names = occluder_names_of(cfg)
    height, width = cfg.get("image_hw", [224, 364])
    patch = int(cfg.get("patch", 14))
    n_h, n_w = patch_grid(int(height), int(width), patch)
    near = float((cfg.get("distance_bins") or {}).get("near", 3))
    mid = float((cfg.get("distance_bins") or {}).get("mid", 10))
    present = {name: [[0.0 for _ in range(n_w)] for _ in range(n_h)] for name in objects}
    dist = {name: [[None for _ in range(n_w)] for _ in range(n_h)] for name in objects}
    occluder = [[0.0 for _ in range(n_w)] for _ in range(n_h)]
    patch_dist = [[None for _ in range(n_w)] for _ in range(n_h)]
    by_class = {name: [[None for _ in range(n_w)] for _ in range(n_h)] for name in objects}
    occluded = {name: [[0.0 for _ in range(n_w)] for _ in range(n_h)] for name in objects}
    hit_names: dict[str, int] = {}
    ignored = patch_ignore(int(height), int(width), n_h, n_w, cfg.get("camera_calib") or {})
    rays = lookup(obs, keys.get("aligned_rays"), []) or []
    from lfnav.env.minedojo_env import ray_raw

    cells_with_hits = set()
    lone = []
    for ray in rays:
        if isinstance(ray.get("hits"), list):
            cells_with_hits.add((int(ray["row"]), int(ray["col"])))
            for hit in ray["hits"]:
                if not isinstance(hit, dict):
                    continue
                raw = hit.get("raw")
                if raw is None:
                    raw = hit.get("label")
                name = ray_raw(raw)
                if name:
                    hit_names[name] = hit_names.get(name, 0) + 1
        else:
            lone.append(ray)
    for ray in lone:
        if (int(ray["row"]), int(ray["col"])) in cells_with_hits:
            continue
        raw = ray.get("raw")
        if raw is None:
            raw = ray.get("label")
        name = ray_raw(raw)
        if name:
            hit_names[name] = hit_names.get(name, 0) + 1
    for ray in rays:
        i, j = int(ray["row"]), int(ray["col"])
        if not (0 <= i < n_h and 0 <= j < n_w):
            continue
        for raw, hit_dist in _hits_of(ray):
            if is_occluder_name(raw, names):
                occluder[i][j] = 1.0
            if hit_dist is None:
                continue
            patch_dist[i][j] = _closer(patch_dist[i][j], hit_dist)
        visible_hits, occluded_hits = _ray_visibility(ray, names)
        for hit in visible_hits:
            hit_dist = hit.get("distance")
            if hit_dist is None:
                continue
            hit_name = resolve_label(hit.get("raw"), aliases, objects)
            if hit_name is not None:
                by_class[hit_name][i][j] = _closer(by_class[hit_name][i][j], hit_dist)
        for hit in occluded_hits:
            hit_name = resolve_label(hit.get("raw"), aliases, objects)
            if hit_name is not None:
                occluded[hit_name][i][j] = 1.0
        if ignored[i][j]:
            continue
        name = resolve_label(ray.get("label"), aliases, objects)
        if name is None:
            continue
        present[name][i][j] = 1.0
        dist[name][i][j] = distance_bin(float(ray.get("distance", mid + 1)), near, mid)
    beliefs = {name: aggregate_object(present[name], dist[name], ignored) for name in objects}

    unknown: list[str] = []
    biome_value = decode_biome(obs, cfg, unknown)
    time_value = decode_time(obs, cfg, unknown)
    weather_value = decode_weather(obs, cfg, unknown)
    bright = decode_brightness(obs, cfg, unknown)
    return SceneState(
        objects=beliefs,
        biome=_one_hot(biome_value, list(labels.get("biomes") or [])),
        time=_one_hot(time_value, list(labels.get("time") or ["day", "night"])),
        weather=_one_hot(weather_value, list(labels.get("weather") or ["clear", "rain"])),
        brightness=_one_hot(bright, list(labels.get("brightness") or [])),
        unknown=unknown,
        ignore=ignored,
        patches=present,
        occluder=occluder,
        patch_dist=patch_dist,
        patch_dist_by_class=by_class,
        occluded_by_class=occluded,
        hit_names=hit_names,
    )
