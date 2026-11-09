"""Rule matcher. No learning and no torch."""

from __future__ import annotations

import logging

from lfnav.aggregate import argmax_name
from lfnav.types import MatchResult, SceneState, TaskSpec

log = logging.getLogger("lfnav.matcher")


def _prob(table: dict, key) -> float:
    if key is None:
        return 1.0
    return float(table.get(key, 0.0))


class Matcher:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        match = cfg.get("match", cfg)
        self.tau_obj = float(match.get("tau_obj", 0.5))
        self.tau_cond = float(match.get("tau_cond", 0.5))
        self.tau_near = float(match.get("tau_near", 0.5))
        self.stop_mode = str(match.get("stop_mode", "sr_b"))
        self.stop_rule = str(match.get("stop_rule", "nearest"))
        judge = cfg.get("judge") or {}
        self.judge_radius = float(match.get("judge_radius", judge.get("radius", 3.0)))
        self.smooth_n = int(match.get("smooth_n", 1))
        self._found_hist: list[bool] = []

    def reset(self) -> None:
        self._found_hist.clear()

    def match(self, task: TaskSpec, scene: SceneState, aligned_distance=None) -> MatchResult:
        target = scene.object(task.target)
        satisfied = {}
        missing = []
        cond = task.conditions or {}

        target_ok = target.present >= self.tau_obj
        satisfied["target"] = target_ok
        if not target_ok:
            missing.append("target")

        for name in list(cond.get("near") or []):
            ok = scene.object(name).present >= self.tau_cond
            satisfied[f"near:{name}"] = ok
            if not ok:
                missing.append(f"near:{name}")

        unknown = set(scene.unknown or [])
        for field, label in (
            ("biome", "biome"),
            ("time", "time"),
            ("weather", "weather"),
            ("brightness", "brightness"),
        ):
            required = cond.get(field)
            if not required:
                continue
            key = f"{label}:{required}"
            if field in unknown:
                log.warning("%s condition %s is unknown, not a silent miss", field, required)
                satisfied[key] = False
                missing.append(f"{key}:unknown")
                continue
            table = getattr(scene, field)
            ok = _prob(table, required) >= self.tau_cond
            satisfied[key] = ok
            if not ok:
                missing.append(key)

        raw_found = not missing
        self._found_hist.append(raw_found)
        window = self._found_hist[-max(self.smooth_n, 1):]
        if self.smooth_n > 1:
            found = sum(window) * 2 >= len(window)
        else:
            found = raw_found

        if target_ok:
            direction = argmax_name(target.direction)
            distance = argmax_name(target.distance)
        else:
            direction = None
            distance = None
        near_p = float(target.distance.get("near", 0.0))
        if self.stop_mode == "sr_a":
            stop = found
        elif self.stop_rule == "fraction":
            stop = found and near_p >= self.tau_near
        else:
            # nearest (default): same as SR-b — found and closest aligned target ray ≤ judge.radius.
            stop = (
                found
                and aligned_distance is not None
                and float(aligned_distance) <= self.judge_radius
            )

        conf = target.present
        for key, ok in satisfied.items():
            if key == "target":
                continue
            conf *= 1.0 if ok else 0.0
        return MatchResult(
            status="found" if found else "not_found",
            stop=bool(stop),
            satisfied=satisfied,
            missing=[] if found else missing,
            target_direction=direction,
            target_distance=distance,
            confidence=float(conf if found else target.present),
        )
