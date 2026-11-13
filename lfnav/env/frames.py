"""Episode screenshots. Failures log a warning and do not stop the episode."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import numpy as np

from lfnav.obsutil import lookup

log = logging.getLogger("lfnav.frames")


def frames_cfg(cfg: dict | None) -> dict:
    raw = dict((cfg or {}).get("frames") or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "every": max(int(raw.get("every", 20)), 1),
        "format": str(raw.get("format", "jpg")).lower().lstrip("."),
        "quality": int(raw.get("quality", 80)),
        "scale": float(raw.get("scale", 0.5)),
        "max_per_episode": int(raw.get("max_per_episode", 400)),
        "annotate": bool(raw.get("annotate", True)),
    }


def rgb_hwc(obs: dict, cfg: dict | None) -> np.ndarray | None:
    keys = (cfg or {}).get("obs_keys") or {}
    raw = lookup(obs, keys.get("rgb", "rgb"))
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    if arr.shape[0] in (3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        mx = float(arr.max()) if arr.size else 0.0
        if mx <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _draw_caption(image, lines: list[str]):
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    y = 4
    for line in lines:
        text = str(line)
        draw.text((5, y + 1), text, fill=(0, 0, 0), font=font)
        draw.text((4, y), text, fill=(255, 255, 255), font=font)
        y += 12


class FrameRecorder:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.opt = frames_cfg(cfg)
        self.dir: Path | None = None
        self.task = ""
        self.seed = 0
        self.step = 0
        self.jpg_count = 0
        self.elapsed_s = 0.0
        self.status = None
        self.direction = None
        self.distance = None
        self.action = None
        self.extra = {}

    def begin_episode(self, folder, task_id, seed: int) -> None:
        self.opt = frames_cfg(self.cfg)
        self.dir = Path(folder)
        self.task = str(task_id)
        self.seed = int(seed)
        self.step = 0
        self.jpg_count = 0
        self.elapsed_s = 0.0
        self.status = None
        self.direction = None
        self.distance = None
        self.action = None
        self.extra = {}
        if not self.opt["enabled"]:
            return
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def set_info(self, match=None, action=None, extra=None) -> None:
        if isinstance(match, dict):
            self.status = match.get("status")
            self.direction = match.get("target_direction")
            self.distance = match.get("target_distance")
        if action is not None:
            self.action = action
        if extra:
            self.extra = dict(extra)

    def after_step(self, obs: dict) -> None:
        self.step += 1
        if not self.opt["enabled"] or self.dir is None:
            return
        if self.step % self.opt["every"] == 0:
            self.capture(obs, tag="step", keyframe=False)

    def capture(self, obs: dict, tag: str, keyframe: bool = False) -> None:
        if not self.opt["enabled"] or self.dir is None:
            return
        t0 = time.perf_counter()
        try:
            rgb = rgb_hwc(obs, self.cfg)
            if rgb is None:
                return
            stem = f"{self.step:05d}_{tag}"
            if keyframe:
                self._save_png(self.dir / f"{stem}.png", rgb)
            self._save_jpg(self.dir / f"{stem}.{self.opt['format']}", rgb)
        except Exception as exc:
            log.warning("frame write failed tag=%s step=%s: %s", tag, self.step, exc)
        self.elapsed_s += time.perf_counter() - t0

    def _save_png(self, path: Path, rgb: np.ndarray) -> None:
        from PIL import Image

        Image.fromarray(rgb).save(path)

    def _save_jpg(self, path: Path, rgb: np.ndarray) -> None:
        if self.jpg_count >= self.opt["max_per_episode"]:
            return
        from PIL import Image

        image = Image.fromarray(rgb)
        scale = self.opt["scale"]
        if scale and scale != 1.0:
            w, h = image.size
            image = image.resize((max(int(w * scale), 1), max(int(h * scale), 1)), Image.BILINEAR)
        if self.opt["annotate"]:
            extra = self.extra or {}
            belief = extra.get("belief") or {}
            wp = extra.get("waypoint") or {}
            lines = [
                f"task={self.task} seed={self.seed} step={self.step}",
                f"status={self.status} dir={self.direction} dist={self.distance}",
                f"action={self.action}",
            ]
            if extra:
                lines.extend([
                    f"mode={extra.get('mode')} wp={wp.get('turn_deg')}/{wp.get('forward')} p_stop={extra.get('p_stop')}",
                    f"belief yaw_sin={belief.get('bearing_sin')} dist={belief.get('log_distance')} conf={belief.get('confidence')}",
                ])
            _draw_caption(image, lines)
        extra = {}
        if self.opt["format"] in ("jpg", "jpeg"):
            extra["quality"] = self.opt["quality"]
        image.save(path, **extra)
        self.jpg_count += 1
