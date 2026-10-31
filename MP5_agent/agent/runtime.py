"""Per-episode output paths and flags. Structured actions and Memory read this."""

from __future__ import annotations

import os
from pathlib import Path


class Runtime:
    def __init__(self):
        self.out_dir = None
        self.images_dir = None
        self.video_dir = None
        self.logs_dir = None
        self.memory_dir = None
        self.save_frames = False
        self.save_perception_images = True
        self.event_logger = None

    def bind(self, out_dir, eval_cfg, event_logger=None):
        self.out_dir = Path(out_dir)
        self.images_dir = self.out_dir / "images"
        self.video_dir = self.out_dir / "video"
        self.logs_dir = self.out_dir / "logs"
        self.memory_dir = self.out_dir / "memory"
        self.save_frames = bool(eval_cfg.get("save_frames", False))
        self.save_perception_images = bool(eval_cfg.get("save_perception_images", True))
        self.event_logger = event_logger
        for d in (self.images_dir, self.video_dir, self.logs_dir, self.memory_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def perception_image_path(self, name):
        self.images_dir.mkdir(parents=True, exist_ok=True)
        return str(self.images_dir / f"{name}.jpg")

    def video_frame_path(self, save_count):
        self.video_dir.mkdir(parents=True, exist_ok=True)
        return str(self.video_dir / f"frame_{save_count}.jpg")

    def reset(self):
        self.__init__()


runtime = Runtime()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
