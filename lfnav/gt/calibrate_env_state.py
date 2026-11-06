"""Measure time, light, rain, and biome ids from a real MineDojo environment.

Creates one env for day/night x clear/rain, then one env per labels.yaml biome.
Prints sun_brightness, light_level, sky_light_level, is_raining, biome_id.
Does not write thresholds; copy the printed values into default.yaml and
obs_keys.biome_id_to_name after you have run this inside the MineDojo image.

    python lfnav/gt/calibrate_env_state.py --out /scratch/$USER/lfnav/calib/env_state.json
"""

from __future__ import annotations

import argparse
import json

from lfnav.config import load_config
from lfnav.gt.scene_state import CANDIDATE_BIOME_IDS
from lfnav.io import atomic_write_json
from lfnav.obsutil import lookup


STAT_FIELDS = ("sun_brightness", "light_level", "sky_light_level", "is_raining", "biome_id")


def location_stats(obs: dict, cfg: dict) -> dict:
    keys = cfg.get("obs_keys") or {}
    loc = obs.get("location_stats") if isinstance(obs, dict) else None
    raw = dict(loc) if isinstance(loc, dict) else {}
    out = {name: raw.get(name) for name in STAT_FIELDS}
    out["sun_brightness"] = lookup(obs, keys.get("time_of_day"), out.get("sun_brightness"))
    out["light_level"] = lookup(obs, keys.get("light_level"), out.get("light_level"))
    out["sky_light_level"] = lookup(obs, keys.get("sky_light_level"), out.get("sky_light_level"))
    out["is_raining"] = lookup(obs, keys.get("weather_rain"), out.get("is_raining"))
    out["biome_id"] = lookup(obs, keys.get("biome_id"), out.get("biome_id"))
    return out


def mapping_from_measurements(rows: list[dict]) -> dict:
    """biome_id string -> specified_biome. Later rows do not overwrite an earlier id."""
    mapping = {}
    for row in rows:
        biome = row.get("specified_biome")
        biome_id = row.get("biome_id")
        if biome is None or biome_id is None or biome == "other":
            continue
        key = str(int(biome_id)) if str(biome_id).lstrip("-").isdigit() else str(biome_id)
        if key not in mapping:
            mapping[key] = str(biome)
    return mapping


def suggested_threshold(low_group: list[float], high_group: list[float]) -> float | None:
    """Midpoint between the highest low value and the lowest high value."""
    lows = [float(v) for v in low_group if v is not None]
    highs = [float(v) for v in high_group if v is not None]
    if not lows or not highs:
        return None
    return (max(lows) + min(highs)) / 2.0


def _reset_stats(cfg: dict, seed: int, **scene) -> dict:
    import minedojo

    from lfnav.env.minedojo_env import _kwargs

    probe = dict(cfg)
    env_cfg = dict(probe.get("minedojo") or {})
    env_cfg["use_lidar"] = False
    probe["minedojo"] = env_cfg
    kwargs = _kwargs(probe, seed)
    kwargs.update(scene)
    env = minedojo.make(**kwargs)
    try:
        return location_stats(env.reset(), cfg)
    finally:
        env.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure MineDojo time, light, rain, and biome ids.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-biomes", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    names = (cfg.get("minedojo") or {}).get("param_names") or {}
    time_key = names.get("time", "start_at_night")
    weather_key = names.get("weather", "initial_weather")
    biome_key = names.get("biome", "specified_biome")
    time_values = (cfg.get("minedojo") or {}).get("time_values") or {"day": False, "night": True}

    clock_rows = []
    for time_name, weather in (("day", "clear"), ("night", "clear"), ("day", "rain"), ("night", "rain")):
        stats = _reset_stats(cfg, seed=0, **{time_key: time_values[time_name], weather_key: weather})
        row = {"time": time_name, "weather": weather, **stats}
        clock_rows.append(row)
        print(
            f"{time_name:5} {weather:5}  sun_brightness={stats.get('sun_brightness')!r}  "
            f"light_level={stats.get('light_level')!r}  sky_light_level={stats.get('sky_light_level')!r}  "
            f"is_raining={stats.get('is_raining')!r}",
            flush=True,
        )

    biome_rows = []
    if not args.skip_biomes:
        labels = list((cfg.get("labels") or {}).get("biomes") or [])
        print("candidate Minecraft 1.11 ids:", json.dumps(CANDIDATE_BIOME_IDS), flush=True)
        for biome in labels:
            if biome == "other":
                continue
            stats = _reset_stats(cfg, seed=1, **{biome_key: biome})
            row = {"specified_biome": biome, "candidate_id": CANDIDATE_BIOME_IDS.get(biome), **stats}
            biome_rows.append(row)
            print(
                f"biome {biome}: measured_id={stats.get('biome_id')!r} candidate_id={CANDIDATE_BIOME_IDS.get(biome)}",
                flush=True,
            )

    mapping = mapping_from_measurements(biome_rows)
    day_sun = [r.get("sun_brightness") for r in clock_rows if r["time"] == "day"]
    night_sun = [r.get("sun_brightness") for r in clock_rows if r["time"] == "night"]
    high_light = [r.get("light_level") for r in clock_rows if r["time"] == "day"]
    low_light = [r.get("light_level") for r in clock_rows if r["time"] == "night"]
    suggested = {
        "time_day_min": suggested_threshold(night_sun, day_sun),
        "brightness_min_light": suggested_threshold(low_light, high_light),
        "biome_id_to_name": mapping,
    }
    payload = {
        "clock": clock_rows,
        "biomes": biome_rows,
        "candidate_biome_ids": CANDIDATE_BIOME_IDS,
        "suggested": suggested,
        "note": "Copy suggested.time_day_min and brightness_min_light into configs/lfnav/default.yaml. "
                "Copy suggested.biome_id_to_name into configs/lfnav/obs_keys.yaml. Measured ids win over candidates.",
    }
    atomic_write_json(args.out, payload)
    print(json.dumps({"out": args.out, "suggested": suggested}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
