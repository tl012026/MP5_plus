"""Sum per-patch object labels into present / direction / distance. No torch."""

from __future__ import annotations

from lfnav.types import ObjectBelief, SceneState

DIRS = ("left", "center", "right")
DISTS = ("near", "mid", "far")


def _blank_dist():
    return {k: 0.0 for k in DISTS}


def _blank_dir():
    return {k: 0.0 for k in DIRS}


def column_direction(col: int, n_w: int) -> str:
    """Third of the image that contains the patch center. A single column is center."""
    if n_w <= 0:
        raise ValueError("n_w must be positive")
    u = (col + 0.5) / n_w
    if u < 1.0 / 3.0:
        return "left"
    if u < 2.0 / 3.0:
        return "center"
    return "right"


def distance_bin(blocks: float, near: float = 3.0, mid: float = 10.0) -> str:
    if blocks <= near:
        return "near"
    if blocks <= mid:
        return "mid"
    return "far"


def aggregate_object(present_hw, dist_hw=None, ignore_hw=None) -> ObjectBelief:
    """present_hw[row][col] in [0, 1]. dist_hw[row][col] is near/mid/far or a {bin: prob} map.

    present is the max over patches. Direction mass is the sum of present in each horizontal
    third, normalized. Distance mass is present-weighted over patches. Ignored HUD/hand
    patches are skipped.
    """
    if not present_hw or not present_hw[0]:
        return ObjectBelief()
    n_h = len(present_hw)
    n_w = len(present_hw[0])
    present = 0.0
    dir_mass = _blank_dir()
    dist_mass = _blank_dist()
    for i in range(n_h):
        for j in range(n_w):
            if ignore_hw and i < len(ignore_hw) and j < len(ignore_hw[i]) and ignore_hw[i][j]:
                continue
            p = float(present_hw[i][j])
            present = max(present, p)
            dir_mass[column_direction(j, n_w)] += p
            if dist_hw is None or p <= 0:
                continue
            cell = dist_hw[i][j]
            if isinstance(cell, str):
                dist_mass[cell] = dist_mass.get(cell, 0.0) + p
            else:
                for name, val in cell.items():
                    dist_mass[name] = dist_mass.get(name, 0.0) + p * float(val)
    dir_sum = sum(dir_mass.values())
    dist_sum = sum(dist_mass.values())
    if dir_sum > 0:
        dir_mass = {k: v / dir_sum for k, v in dir_mass.items()}
    if dist_sum > 0:
        dist_mass = {k: v / dist_sum for k, v in dist_mass.items()}
    return ObjectBelief(present=present, direction=dir_mass, distance=dist_mass)


def scene_from_patches(per_object, globals_probs=None) -> SceneState:
    """per_object: name -> (present_hw, dist_hw). globals_probs holds biome/time/weather/brightness."""
    objects = {name: aggregate_object(pres, dist) for name, (pres, dist) in per_object.items()}
    g = globals_probs or {}
    return SceneState(
        objects=objects,
        biome=dict(g.get("biome") or {}),
        time=dict(g.get("time") or {}),
        weather=dict(g.get("weather") or {}),
        brightness=dict(g.get("brightness") or {}),
    )


def argmax_name(probs: dict) -> str | None:
    if not probs:
        return None
    return max(probs.items(), key=lambda kv: kv[1])[0]
