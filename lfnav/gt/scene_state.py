"""Build a ground-truth SceneState from aligned rays and configured global fields.

Block and mob name fragments come from lfnav/labels.yaml. Biome id numbers are
read only through obs_keys.biome_id_to_name, which probe_obs.py is meant to fill.
"""

from __future__ import annotations

from lfnav.aggregate import aggregate_object, distance_bin
from lfnav.geom import patch_grid
from lfnav.obsutil import lookup
from lfnav.types import SceneState


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
    rays = lookup(obs, keys.get("aligned_rays"), []) or []
    for ray in rays:
        name = resolve_label(ray.get("label"), aliases, objects)
        if name is None:
            continue
        i, j = int(ray["row"]), int(ray["col"])
        if not (0 <= i < n_h and 0 <= j < n_w):
            continue
        present[name][i][j] = 1.0
        dist[name][i][j] = distance_bin(float(ray.get("distance", mid + 1)), near, mid)
    beliefs = {name: aggregate_object(present[name], dist[name]) for name in objects}

    biome_value = lookup(obs, keys.get("biome_name"))
    if biome_value is None and keys.get("biome_id") is not None:
        biome_id = lookup(obs, keys.get("biome_id"))
        biome_value = (keys.get("biome_id_to_name") or {}).get(str(biome_id), biome_id)
    time_value = lookup(obs, keys.get("time_name"))
    # time_day_below is unset until probe_obs shows the clock units.
    if time_value is None and cfg.get("time_day_below") is not None and keys.get("time_of_day") is not None:
        raw = lookup(obs, keys.get("time_of_day"))
        if raw is not None:
            time_value = "day" if float(raw) < float(cfg["time_day_below"]) else "night"
    weather_value = lookup(obs, keys.get("weather_name"))
    if weather_value is None and keys.get("weather_rain") is not None:
        raw = lookup(obs, keys.get("weather_rain"))
        if raw is not None:
            weather_value = "rain" if bool(raw) else "clear"
    bright = lookup(obs, keys.get("brightness_name"))
    if bright is None and cfg.get("brightness_min_light") is not None and keys.get("light_level") is not None:
        raw = lookup(obs, keys.get("light_level"))
        if raw is not None:
            bright = "sufficient" if float(raw) >= float(cfg["brightness_min_light"]) else "insufficient"
    return SceneState(
        objects=beliefs,
        biome=_one_hot(biome_value, list(labels.get("biomes") or [])),
        time=_one_hot(time_value, list(labels.get("time") or ["day", "night"])),
        weather=_one_hot(weather_value, list(labels.get("weather") or ["clear", "rain"])),
        brightness=_one_hot(bright, list(labels.get("brightness") or [])),
    )
