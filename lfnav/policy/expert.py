"""Expert used only to record demonstrations.

It is C1 with ground-truth perception plus the wide lidar. Evaluation must not
construct this class. The controller's local voxels are a separate privilege.
"""

from __future__ import annotations

from lfnav.obsutil import lookup
from lfnav.policy.c1 import RuleBased
from lfnav.types import PolicyInput, PolicyOutput


def wide_target(obs: dict, target: str, cfg: dict) -> dict:
    rays = lookup(obs, (cfg.get("obs_keys") or {}).get("wide_rays"), []) or []
    hits = [ray for ray in rays if ray.get("label") == target]
    if not hits:
        return {"visible": False}
    best = min(hits, key=lambda ray: float(ray.get("distance", 1e9)))
    offset = float(best.get("yaw_offset", 0.0))
    if offset > 180.0:
        offset -= 360.0
    # yaw_offset is lfnav degrees (+ is right). MineDojo sign flip happens only in to_minedojo_lidar.
    if offset < -20.0:
        direction = "left"
    elif offset > 20.0:
        direction = "right"
    else:
        direction = "center"
    return {"visible": True, "direction": direction, "distance": float(best["distance"])}


def aligned_target_missing(match: dict | None) -> bool:
    """True when aligned rays do not see the target (match.satisfied.target is false)."""
    satisfied = (match or {}).get("satisfied") or {}
    return not bool(satisfied.get("target"))


def uses_wide_hint(wide: dict | None, match: dict | None) -> bool:
    """Wide lidar is only a search cue. If the target is already in the camera, C1 handles it."""
    return bool((wide or {}).get("visible") and aligned_target_missing(match))


class Expert(RuleBased):
    name = "expert"

    def act(self, policy_input: PolicyInput) -> PolicyOutput:
        wide = (policy_input.memory or {}).get("wide_target") or {}
        # Old gate was status != "found", which also fired when the target was in
        # view but a near-X condition failed, so the expert spun on the wide hit.
        if uses_wide_hint(wide, policy_input.match):
            self._scan_left = 0
            return PolicyOutput([self._steer(wide.get("direction"))])
        return super().act(policy_input)
