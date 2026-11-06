"""Ground-truth helpers and perception collection without MineDojo."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfnav.config import load_config
from lfnav.data.collect_perception import capture, frames_done, make_fake
from lfnav.data.vis_labels import tint, write_ppm
from lfnav.gt.fov import fov_deg_from_pixel
from lfnav.gt.scene_state import resolve_label


class FovTest(unittest.TestCase):
    def test_ninety_degree_edge(self):
        fov = fov_deg_from_pixel(99.5, 100, 45.0)
        self.assertAlmostEqual(fov, 90.0, places=3)

    def test_alias(self):
        aliases = {"tree": ["log", "leaves"]}
        self.assertEqual(resolve_label("oak_log", aliases, ["tree", "pig"]), "tree")
        self.assertEqual(resolve_label("pig", aliases, ["tree", "pig"]), "pig")


class CollectTest(unittest.TestCase):
    def test_shards_resume(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        cfg["shard_size"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            env = make_fake(cfg, 0)
            capture(env, cfg, 3, out, 0, yaw_step=30)
            self.assertGreaterEqual(frames_done(out), 3)
            again = capture(make_fake(cfg, 0), cfg, 3, out, 0)
            self.assertEqual(again, frames_done(out))
            with np.load(sorted(out.glob("shard_*.npz"))[0]) as data:
                self.assertEqual(data["rgb"].shape[-1], 3)

    def test_tint_writes_ppm(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        present = {"pig": [[0.0, 1.0], [0.0, 0.0]]}
        painted = tint(rgb, present, ["pig"])
        self.assertGreater(int(painted[0, 4, 0]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.ppm"
            write_ppm(path, painted)
            self.assertTrue(path.read_bytes().startswith(b"P6"))


if __name__ == "__main__":
    unittest.main()
