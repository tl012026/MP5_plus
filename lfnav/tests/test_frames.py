"""Episode screenshots on the fake env, without MineDojo."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lfnav.config import load_config
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.env.frames import FrameRecorder, rgb_hwc
from lfnav.eval.make_contact_sheet import write_sheet
from lfnav.eval.run import run_batch

ROOT = Path(__file__).resolve().parents[2]


def _cfg():
    cfg = load_config()
    cfg["image_hw"] = [14, 14]
    cfg["perception_backend"] = "gt"
    cfg["policy_name"] = "c1"
    cfg["backend"] = "fake"
    cfg["judge"] = dict(cfg.get("judge") or {})
    cfg["judge"]["max_steps"] = 24
    cfg["frames"] = {
        "enabled": True, "every": 5, "format": "jpg", "quality": 80,
        "scale": 0.5, "max_per_episode": 400, "annotate": True,
    }
    return cfg


class FramesTest(unittest.TestCase):
    def test_rgb_and_none_caption(self):
        cfg = _cfg()
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": 6.0}]})
        rgb = rgb_hwc(env.reset(), cfg)
        self.assertEqual(rgb.shape[-1], 3)
        rec = FrameRecorder(cfg)
        rec.set_info({"status": "not_found", "target_direction": None, "target_distance": None}, action=None)
        self.assertIsNone(rec.direction)
        self.assertIsNone(rec.distance)

    def test_batch_writes_start_step_end_and_clears_on_rerun(self):
        cfg = _cfg()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tasks = [str(ROOT / "lfnav/tasks/specs/1-4.json")]
            run_batch(cfg, tasks, [0], out, skip_done=False)
            frames = out / "1-4" / "0" / "frames"
            jpgs = sorted(p.name for p in frames.glob("*.jpg"))
            pngs = sorted(p.name for p in frames.glob("*.png"))
            self.assertIn("00000_start.jpg", jpgs)
            self.assertIn("00000_start.png", pngs)
            self.assertTrue(any(name.endswith("_end.jpg") for name in jpgs))
            self.assertTrue(any(name.endswith("_end.png") for name in pngs))
            rec = json.loads((out / "1-4" / "0.json").read_text())
            self.assertIn("frames_s", rec)
            self.assertIn("frames_overhead", rec)
            junk = frames / "zzzz_old.jpg"
            junk.write_bytes(b"x")
            dest = out / "1-4" / "0.json"
            dest.write_text(json.dumps({"task_name": "1-4", "seed": 0, "success": True, "marker": "keep"}))
            run_batch(cfg, tasks, [0], out, skip_done=True)
            self.assertTrue(junk.is_file())
            dest.unlink()
            run_batch(cfg, tasks, [0], out, skip_done=False)
            self.assertFalse(junk.is_file())
            sheet = write_sheet(frames)
            self.assertTrue(sheet.is_file())

    def test_write_failure_does_not_abort(self):
        cfg = _cfg()
        env = FakeMineDojoEnv(cfg, {})
        rec = env.frames
        with tempfile.TemporaryDirectory() as tmp:
            rec.begin_episode(Path(tmp) / "frames", "1-4", 0)
            with patch.object(rec, "_save_jpg", side_effect=OSError("disk")):
                rec.capture(env.reset(), tag="start", keyframe=False)
            rec.set_info({"status": "not_found", "target_direction": None, "target_distance": None}, action="forward")


if __name__ == "__main__":
    unittest.main()
