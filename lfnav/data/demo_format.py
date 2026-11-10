"""On-disk layout for expert demonstrations. Collect and train both use this."""

from __future__ import annotations

import json
from pathlib import Path

from lfnav.io import atomic_write_json, json_default
from lfnav.policy.c1 import sector8_counts

BUCKETS = ("success", "failure")
FRAME_HW = (224, 364)  # height, width
FRAME_QUALITY = 90


def episode_dir(root, bucket: str, task_id: str, difficulty: str, seed: int) -> Path:
    return Path(root) / bucket / task_id / str(difficulty) / str(int(seed))


def find_episode(root, task_id: str, difficulty: str, seed: int) -> Path | None:
    for bucket in BUCKETS:
        rec = episode_dir(root, bucket, task_id, difficulty, seed) / "record.json"
        if rec.is_file():
            return rec.parent
    return None


def config_name(cfg: dict | None) -> str | None:
    if not cfg:
        return None
    path = cfg.get("_path")
    return Path(path).name if path else None


def step_row(event: dict, env_step: int, memory: dict | None, used_wide: bool) -> dict:
    mem = memory or {}
    grid = mem.get("local_grid") or []
    return {
        "decision": event.get("decision"),
        "env_step": int(env_step),
        "match": event.get("match"),
        "scene": event.get("gt") or event.get("scene"),
        "memory": {
            "sector8": sector8_counts(grid),
            "recent_actions": list(mem.get("recent_actions") or []),
        },
        "action_chunk": None if not event.get("policy") else list((event.get("policy") or {}).get("action_chunk") or []),
        "used_wide": bool(used_wide),
    }


def write_record(folder: Path, record: dict) -> None:
    atomic_write_json(Path(folder) / "record.json", record)


def write_steps(folder: Path, rows: list[dict]) -> None:
    path = Path(folder) / "steps.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, default=json_default) + "\n" for row in rows)
    path.write_text(text)


def save_decision_frame(folder: Path, decision: int, obs: dict, cfg: dict | None = None) -> Path | None:
    from lfnav.env.frames import rgb_hwc

    rgb = rgb_hwc(obs, cfg or {})
    if rgb is None:
        return None
    from PIL import Image

    dest = Path(folder) / "frames" / f"{int(decision):05d}.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb)
    h, w = FRAME_HW
    if image.size != (w, h):
        image = image.resize((w, h), Image.BILINEAR)
    image.save(dest, quality=FRAME_QUALITY)
    return dest


def write_episode(root, bucket: str, task_id: str, difficulty: str, seed: int, record: dict, rows: list[dict]) -> Path:
    dest = episode_dir(root, bucket, task_id, difficulty, seed)
    dest.mkdir(parents=True, exist_ok=True)
    write_record(dest, record)
    write_steps(dest, rows)
    return dest


def read_episode(folder) -> dict:
    folder = Path(folder)
    record = json.loads((folder / "record.json").read_text())
    steps_path = folder / "steps.jsonl"
    steps = []
    if steps_path.is_file():
        for line in steps_path.read_text().splitlines():
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    frames_dir = folder / "frames"
    frames = sorted(frames_dir.glob("*.jpg")) if frames_dir.is_dir() else []
    return {"record": record, "steps": steps, "frames": frames, "folder": folder}
