"""Probe /difficulty peaceful, /summon, and resistance NBT inside the MineDojo image.

Exits non-zero if any check fails. Host Python is not enough; run via probe_commands.sbatch.
Does not change ``_send_chat``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

from lfnav.config import load_config
from lfnav.env.frames import rgb_hwc
from lfnav.env.minedojo_env import (
    RESISTANCE_NBT,
    MineDojoEnv,
    _chat_actor,
    _send_chat,
    count_hostile_hits,
    ray_raw,
)
from lfnav.obsutil import lookup, position_of, yaw_of
from lfnav.types import TaskSpec

PIG_CMD = "/summon pig ~ ~1 ~3"
ZOMBIE_CMD = "/summon zombie ~ ~ ~4"


def probe_dir(root=None) -> Path:
    if root is not None:
        return Path(root)
    scr = os.environ.get("SCR") or f"/scratch/{os.environ.get('USER', 'user')}"
    return Path(scr) / "lfnav" / "probe"


def loc_pos(obs: dict):
    loc = obs.get("location_stats") if isinstance(obs, dict) else None
    if not isinstance(loc, dict):
        return None
    pos = loc.get("pos")
    if pos is None:
        return None
    try:
        return [float(v) for v in pos]
    except (TypeError, ValueError):
        return pos


def note_obs(obs: dict, cfg: dict, source: str) -> None:
    print(f"obs_source={source} location_stats.pos={loc_pos(obs)} pos={position_of(obs, cfg)}")


def raw_name_counts(obs: dict, cfg: dict | None = None) -> Counter:
    keys = (cfg or {}).get("obs_keys") or {}
    counts: Counter = Counter()
    for key in (keys.get("aligned_rays"), keys.get("wide_rays")):
        for ray in lookup(obs, key, []) or []:
            name = ray_raw(ray.get("raw"))
            if name:
                counts[name] += 1
    return counts


def format_raw_counts(counts: Counter) -> str:
    if not counts:
        return "(none)"
    return " ".join(f"{name}={n}" for name, n in sorted(counts.items()))


def raw_hits(counts: Counter, name: str) -> int:
    target = ray_raw(name) or str(name).lower()
    return sum(n for key, n in counts.items() if key == target or key.endswith(target))


def save_rgb(obs: dict, cfg: dict, dest: Path) -> Path | None:
    rgb = rgb_hwc(obs, cfg)
    if rgb is None:
        print(f"rgb missing, skip {dest}")
        return None
    try:
        from PIL import Image

        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(dest, quality=90)
        print(f"saved {dest}")
        return dest
    except Exception as exc:
        print(f"rgb save failed {dest}: {exc}")
        return None


def dump_view(obs: dict, cfg: dict, tag: str, tick: int, out: Path) -> Counter:
    counts = raw_name_counts(obs, cfg)
    pos = position_of(obs, cfg)
    yaw = yaw_of(obs, cfg)
    print(f"dump tag={tag} tick={tick} pos={pos} yaw={yaw} raw={format_raw_counts(counts)}")
    save_rgb(obs, cfg, out / f"{tag}_{tick}.jpg")
    return counts


def step_obs(env, cfg: dict, action: str):
    obs, _r, _d, _i = env.step(action)
    note_obs(obs, cfg, "env.step")
    return obs


def wait_ticks(env, cfg: dict, tag: str, out: Path, n: int = 20, every: int = 5) -> dict:
    obs = env.observation() if hasattr(env, "observation") else None
    for tick in range(1, int(n) + 1):
        obs = step_obs(env, cfg, "noop")
        if tick % int(every) == 0:
            dump_view(obs, cfg, tag, tick, out)
    return obs


def spin_scan(env, cfg: dict, name: str) -> tuple[bool, dict]:
    """Eight 45° turns. Any heading that hits ``name`` on aligned or wide rays counts."""
    obs = env.observation() if hasattr(env, "observation") else None
    if obs is None:
        obs = step_obs(env, cfg, "noop")
    seen = raw_hits(raw_name_counts(obs, cfg), name) > 0
    for heading in range(8):
        obs = step_obs(env, cfg, "turn_right_45")
        counts = raw_name_counts(obs, cfg)
        hits = raw_hits(counts, name)
        print(f"spin heading={heading} yaw={yaw_of(obs, cfg)} hits={hits} raw={format_raw_counts(counts)}")
        if hits > 0:
            seen = True
    return seen, obs


def after_summon(env, cfg: dict, name: str, tag: str, out: Path) -> bool:
    wait_ticks(env, cfg, tag, out, n=20, every=5)
    seen, _obs = spin_scan(env, cfg, name)
    print(f"seen tag={tag} name={name} -> {seen}")
    return seen


def summon_execute_cmd(env, cfg: dict, cmd: str) -> dict:
    sim = _chat_actor(env)
    sim.execute_cmd(cmd)
    print(f"execute_cmd {cmd!r} (no obs from execute_cmd)")
    return step_obs(env, cfg, "noop")


def summon_send_chat(env, cfg: dict, cmd: str) -> dict:
    obs = _send_chat(env, cmd)
    note_obs(obs, cfg, "_send_chat")
    return obs


def summon_trial(env, cfg: dict, mob: str, method: str, cmd: str, tag: str, out: Path) -> dict:
    """Run one summon, wait 20 ticks, spin. method is execute_cmd or send_chat."""
    row = {"method": method, "mob": mob, "cmd": cmd, "tag": tag, "seen": False, "error": None}
    try:
        if method == "execute_cmd":
            summon_execute_cmd(env, cfg, cmd)
        elif method == "send_chat":
            summon_send_chat(env, cfg, cmd)
        else:
            raise ValueError(f"unknown summon method {method}")
        row["seen"] = after_summon(env, cfg, mob, tag, out)
    except Exception as exc:
        row["error"] = str(exc)
        print(f"ERROR tag={tag}: {exc}")
    return row


def format_table(rows: list[dict]) -> str:
    lines = ["summon_method  mob     seen  error", "-" * 48]
    for row in rows:
        seen = "error" if row.get("error") else ("seen" if row.get("seen") else "not_seen")
        err = (row.get("error") or "")[:40]
        lines.append(f"{row['method']:<14} {row['mob']:<7} {seen:<9} {err}")
    return "\n".join(lines)


def _tick(env, n: int):
    obs = env.observation()
    for _ in range(int(n)):
        obs, _r, _d, _i = env.step("noop")
    return obs


def _fail(name: str, detail: str) -> str:
    print(f"FAIL {name}: {detail}")
    return name


def check_peaceful(env, cfg: dict, out=None) -> str | None:
    """Positive control (easy + zombie on rays), then peaceful must despawn it.

    Returns a fail detail, or None on PASS. Prints ``detector cannot see zombie``
    when the control never hits a zombie ray.
    """
    dest = probe_dir(out)
    obs = _send_chat(env, "/difficulty easy")
    note_obs(obs, cfg, "_send_chat")
    obs = _send_chat(env, ZOMBIE_CMD)
    note_obs(obs, cfg, "_send_chat")
    wait_ticks(env, cfg, "peaceful_zombie", dest, n=20, every=5)
    seen, obs = spin_scan(env, cfg, "zombie")
    if not seen and count_hostile_hits(obs, cfg) <= 0:
        print("detector cannot see zombie")
        return "detector cannot see zombie"
    obs = _send_chat(env, "/difficulty peaceful")
    note_obs(obs, cfg, "_send_chat")
    wait_ticks(env, cfg, "peaceful_after", dest, n=20, every=5)
    gone_scan, obs = spin_scan(env, cfg, "zombie")
    hits = count_hostile_hits(obs, cfg)
    if gone_scan or hits > 0:
        return f"zombie still on rays after 20 ticks ({hits})"
    return None


def run_probes(cfg: dict, out=None) -> list[str]:
    failed = []
    dest = probe_dir(out)
    dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    task = TaskSpec(id="probe", source="", description="probe", target="pig", conditions={}, episode={
        "specified_biome": "plains",
        "start_time": "day",
        "initial_weather": "clear",
        "spawn": [],
    })
    env = MineDojoEnv(cfg, task, seed=3)
    try:
        env.reset(task=task, seed=3)
        t0 = time.perf_counter()
        try:
            detail = check_peaceful(env, cfg, out=dest)
            if detail:
                failed.append(_fail("peaceful", detail))
            else:
                print("PASS peaceful")
        except Exception as exc:
            failed.append(_fail("peaceful", str(exc)))

        rows.append(summon_trial(env, cfg, "pig", "execute_cmd", PIG_CMD, "pig_execute_cmd", dest))
        rows.append(summon_trial(env, cfg, "pig", "send_chat", PIG_CMD, "pig_send_chat", dest))
        rows.append(summon_trial(env, cfg, "zombie", "send_chat", ZOMBIE_CMD, "zombie_send_chat", dest))
        rows.append(summon_trial(env, cfg, "pig", "send_chat", PIG_CMD, "pig_send_chat_vs_exec", dest))

        for row in rows:
            if row["method"] == "execute_cmd" and row["mob"] == "pig" and not row["seen"]:
                failed.append(_fail("summon_execute_cmd", row.get("error") or "pig not seen"))
            if row["tag"] == "pig_send_chat" and not row["seen"]:
                failed.append(_fail("summon_send_chat", row.get("error") or "pig not seen"))

        try:
            obs = _send_chat(env, f"/summon pig ~ ~1 ~6 {RESISTANCE_NBT}")
            note_obs(obs, cfg, "_send_chat")
            print("PASS resistance_nbt")
        except Exception as exc:
            failed.append(_fail("resistance_nbt", str(exc)))
        print("SUMMON TABLE")
        print(format_table(rows))
        print(f"probe_s={time.perf_counter() - t0:.1f}")
    finally:
        env.close()
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lfnav/main.yaml")
    parser.add_argument("--out", default=None, help="Screenshot directory (default $SCR/lfnav/probe)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["backend"] = "minedojo"
    failed = run_probes(cfg, out=args.out)
    if failed:
        print("FAILED:", ",".join(failed))
        sys.exit(1)
    print("ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
