"""Wilson intervals and a paired McNemar test for two runs on the same seeds."""

from __future__ import annotations

import math


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict:
    if total <= 0:
        return {"rate": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return {"rate": p, "low": max(0.0, center - margin), "high": min(1.0, center + margin), "n": total}


def mcnemar(reference: list[bool], other: list[bool]) -> dict:
    if len(reference) != len(other):
        raise ValueError("paired lists must have the same length")
    only_reference = sum(bool(a) and not bool(b) for a, b in zip(reference, other))
    only_other = sum(bool(b) and not bool(a) for a, b in zip(reference, other))
    discordant = only_reference + only_other
    if discordant == 0:
        chi2 = 0.0
    else:
        chi2 = (abs(only_reference - only_other) - 1) ** 2 / discordant
    return {"only_reference": only_reference, "only_other": only_other, "chi2_continuity": chi2}


def group_rates(rows: list[dict], key: str = "success") -> dict:
    groups = {}
    for row in rows:
        name = row.get("group", "all")
        groups.setdefault(name, []).append(bool(row.get(key)))
    return {name: wilson_interval(sum(vals), len(vals)) for name, vals in groups.items()}
