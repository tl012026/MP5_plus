"""Perception dry-run, feature cache, MineLLM parsing, and offline metrics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfnav.eval.eval_perception import score_scenes
from lfnav.perception.encoder import FakeEncoder
from lfnav.perception.extract_features import extract_shard
from lfnav.perception.minellm import parse_choice, parse_yes, scene_from_answers
from lfnav.perception.preprocess import target_hw
from lfnav.perception.train_perception import dry_run, split_name


class PerceptionTest(unittest.TestCase):
    def test_resize_and_split(self):
        self.assertEqual(target_hw(100, 200, 224, 14)[0], 224)
        self.assertEqual(target_hw(100, 200, 224, 14)[1] % 14, 0)
        self.assertEqual(split_name(0), "train")
        self.assertEqual(split_name(8), "val")
        self.assertEqual(split_name(9), "test")

    def test_dry_run_and_cache(self):
        dry_run()
        rgb = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "shard_00000.npz"
            np.savez_compressed(src, rgb=rgb)
            dst = Path(tmp) / "feat.npz"
            extract_shard(src, dst, FakeEncoder())
            with np.load(dst) as data:
                self.assertEqual(data["cls"].dtype, np.float16)
                self.assertEqual(data["cls"].shape[0], 2)

    def test_minellm_parse(self):
        self.assertEqual(parse_yes("Yes, there is."), 1.0)
        self.assertEqual(parse_yes("No."), 0.0)
        labels = {"biomes": ["forest", "plains", "other"], "time": ["day", "night"], "weather": ["clear", "rain"], "brightness": ["sufficient", "insufficient"]}
        scene = scene_from_answers(
            {"present:pig": 1.0, "biome": "plains", "time": "night", "weather": "rain", "brightness": "sufficient"},
            ["pig"],
            labels,
        )
        self.assertEqual(scene.object("pig").present, 1.0)
        self.assertEqual(parse_choice("plains", labels["biomes"])["plains"], 1.0)
        self.assertEqual(scene.time["night"], 1.0)

    def test_metrics(self):
        rows = [{
            "pred": {"objects": {"pig": {"present": 0.9, "direction": {"center": 1}, "distance": {"near": 1}}}},
            "gt": {"objects": {"pig": {"present": 1, "direction": {"center": 1}, "distance": {"near": 1}}}},
        }, {
            "pred": {"objects": {"pig": {"present": 0.8, "direction": {"left": 1}, "distance": {"far": 1}}}},
            "gt": {"objects": {"pig": {"present": 0, "direction": {"center": 0}, "distance": {"far": 0}}}},
        }]
        report = score_scenes(rows)
        self.assertGreater(report["present"]["f1"], 0.0)
        self.assertGreater(report["present"]["hallucination"], 0.0)
        self.assertEqual(report["direction_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
