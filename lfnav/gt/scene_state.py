"""Build a ground-truth SceneState from aligned rays and configured global fields.

Block and mob name fragments come from lfnav/labels.yaml. Biome ids are mapped
through obs_keys.biome_id_to_name. Time uses sun_brightness vs time_day_min;
brightness uses light_level vs brightness_min_light. Unset thresholds leave the
field unknown instead of a silent miss.
"""

from __future__ import annotations

import logging

from lfnav.aggregate import aggregate_object, distance_bin
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
    height, width = cfg.get("image_hw", [224, 364])
    patch = int(cfg.get("patch", 14))
    n_h, n_w = patch_grid(int(height), int(width), patch)
    near = float((cfg.get("distance_bins") or {}).get("near", 3))
    mid = float((cfg.get("distance_bins") or {}).get("mid", 10))
    present = {name: [[0.0 for _ in range(n_w)] for _ in range(n_h)] for name in objects}
    dist = {name: [[None for _ in range(n_w)] for _ in range(n_h)] for name in objects}
    ignored = patch_ignore(int(height), int(width), n_h, n_w, cfg.get("camera_calib") or {})
    rays = lookup(obs, keys.get("aligned_rays"), []) or []
    for ray in rays:
        name = resolve_label(ray.get("label"), aliases, objects)
        if name is None:
            continue
        i, j = int(ray["row"]), int(ray["col"])
        if not (0 <= i < n_h and 0 <= j < n_w):
            continue
        if ignored[i][j]:
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
    )
