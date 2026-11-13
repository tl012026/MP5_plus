"""On-disk layout for expert demonstrations. Collect and train both use this."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from lfnav.io import atomic_write_json, json_default
from lfnav.policy.c1 import sector8_counts

BUCKETS = ("success", "failure")
FRAME_HW = (224, 364)  # height, width
FRAME_QUALITY = 90


def _hwc_uint8(rgb) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected an RGB image, got {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        mx = float(arr.max()) if arr.size else 0.0
        if mx <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def preprocess_frame(rgb, write_to=None) -> np.ndarray:
    """MineDojo RGB → uint8 (224, 364, 3) after the same JPEG-90 resize used for demos."""
    from PIL import Image

    image = Image.fromarray(_hwc_uint8(rgb))
    height, width = FRAME_HW
    if image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=FRAME_QUALITY)
    data = buf.getvalue()
    if write_to is not None:
        dest = Path(write_to)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    decoded = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(decoded, dtype=np.uint8)


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
    return Path(path).stem if path else None


def step_row(
    event: dict,
    env_step: int,
    memory: dict | None,
    used_wide: bool,
    pos=None,
    yaw=None,
    pitch=None,
    executed_actions=None,
    expert_branch=None,
    target_distance_num=None,
) -> dict:
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
        "pos": None if pos is None else [float(v) for v in pos],
        "yaw": None if yaw is None else float(yaw),
        "pitch": None if pitch is None else float(pitch),
        "executed_actions": list(executed_actions or []),
        "expert_branch": expert_branch,
        "target_distance_num": None if target_distance_num is None else float(target_distance_num),
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
    if rgb is None and isinstance(obs, dict):
        rgb = obs.get("rgb")
    if rgb is None:
        return None
    dest = Path(folder) / "frames" / f"{int(decision):05d}.jpg"
    preprocess_frame(rgb, write_to=dest)
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
