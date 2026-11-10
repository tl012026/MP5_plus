"""Expert demo format, seed split, block-task skip, used_wide, dynamic initial distance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lfnav.config import load_config
from lfnav.data.collect_demos import assert_train_seed, skip_demo_cell, used_wide_for_decision
from lfnav.data.demo_format import find_episode, read_episode, save_decision_frame, step_row, write_episode
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.gt.initial import apply_spawn_offset_distance, initial_scene_metrics
from lfnav.gt.judge import aligned_target_distance
from lfnav.policy.expert import Expert, wide_target
from lfnav.types import PolicyInput, TaskSpec

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "lfnav" / "tasks" / "specs"


class DemoFormatTest(unittest.TestCase):
    def test_roundtrip(self):
        record = {"task_name": "1-4", "success": True, "steps": 4, "config_name": "demo_dynamic.yaml"}
        rows = [step_row(
            {"decision": 0, "match": {"status": "not_found"}, "gt": {"objects": {}}, "policy": {"action_chunk": ["forward"]}},
            0, {"local_grid": [[0.0] * 15] * 15, "recent_actions": ["turn_right_15"]}, True,
        )]
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_episode(tmp, "success", "1-4", "easy", 1, record, rows)
            again = read_episode(folder)
            self.assertEqual(again["record"], record)
            self.assertEqual(again["steps"][0]["used_wide"], True)
            self.assertEqual(again["steps"][0]["action_chunk"], ["forward"])
            self.assertEqual(len(again["steps"][0]["memory"]["sector8"]), 8)
            self.assertEqual(find_episode(tmp, "1-4", "easy", 1), folder)


class SeedAndSkipTest(unittest.TestCase):
    def test_seed_ending_in_nine_is_rejected(self):
        with self.assertRaises(ValueError):
            assert_train_seed(19)
        assert_train_seed(11)

    def test_block_medium_hard_skipped(self):
        tree = TaskSpec.from_dict(json.loads((SPECS / "1-1.json").read_text()))
        pig = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        self.assertTrue(skip_demo_cell(tree, "medium"))
        self.assertTrue(skip_demo_cell(tree, "hard"))
        self.assertFalse(skip_demo_cell(tree, "easy"))
        self.assertFalse(skip_demo_cell(pig, "hard"))
        grass = TaskSpec.from_dict(json.loads((SPECS / "2-2.json").read_text()))
        near_grass = TaskSpec.from_dict(json.loads((SPECS / "4-4.json").read_text()))
        self.assertTrue(skip_demo_cell(grass, "medium"))
        self.assertFalse(skip_demo_cell(near_grass, "hard"))


class UsedWideTest(unittest.TestCase):
    def test_wide_gate_matches_expert(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 4.0, "z": 0.3}]})
        obs = env.reset()
        wide = wide_target(obs, "pig", cfg)
        self.assertTrue(wide.get("visible"))
        missed = {"status": "not_found", "satisfied": {}, "missing": []}
        found = {"status": "found", "satisfied": {"target": True}}
        self.assertTrue(used_wide_for_decision(wide, missed))
        self.assertFalse(used_wide_for_decision(wide, found))
        out = Expert(cfg).act(PolicyInput(match=missed, memory={"wide_target": wide, "local_grid": [[0.0]]}))
        self.assertEqual(out.action_chunk[0], "turn_right_15")


class DynamicDistanceTest(unittest.TestCase):
    def _nearest_ray_metrics(self, cfg):
        cfg = dict(cfg)
        cfg["image_hw"] = [14, 14]
        cfg["aligned_ray_range"] = 32
        cfg["wide_ray_range"] = 32
        cfg["minedojo"] = dict(cfg.get("minedojo") or {})
        env = FakeMineDojoEnv(cfg, {"entities": [
            {"name": "pig", "x": 0.0, "z": 3.0},
            {"name": "pig", "x": 0.0, "z": 16.0},
        ]})
        obs = env.reset()
        env.spawn_meta = {"target_spawn_offset": {"dx": 0, "dz": 16}}
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        metrics = apply_spawn_offset_distance(initial_scene_metrics(obs, env, cfg, task), env.spawn_meta, cfg)
        ray = aligned_target_distance(obs, cfg, "pig")
        return metrics, ray

    def test_clear_entities_false_uses_nearest_ray(self):
        cfg = load_config()
        cfg["minedojo"] = dict(cfg.get("minedojo") or {})
        cfg["minedojo"]["clear_entities"] = False
        metrics, ray = self._nearest_ray_metrics(cfg)
        self.assertIsNotNone(ray)
        self.assertAlmostEqual(float(metrics["initial_target_distance"]), float(ray), delta=0.4)
        self.assertAlmostEqual(float(metrics["spawn_target_distance"]), 16.0)
        self.assertTrue(metrics["natural_target_closer"])
        self.assertLess(float(metrics["initial_target_distance"]), 8.0)

    def test_eval_dynamic_uses_the_same_ray_distance(self):
        cfg = load_config(ROOT / "configs/lfnav/eval_dynamic.yaml")
        metrics, ray = self._nearest_ray_metrics(cfg)
        self.assertFalse(cfg["minedojo"]["clear_entities"])
        self.assertIsNotNone(ray)
        self.assertAlmostEqual(float(metrics["initial_target_distance"]), float(ray), delta=0.4)


class DemoSeedsFileTest(unittest.TestCase):
    def test_forty_four_seeds(self):
        from lfnav.io import read_seeds
        seeds = read_seeds(ROOT / "configs" / "lfnav" / "train_seeds_demo.txt")
        self.assertEqual(len(seeds), 44)
        self.assertTrue(all(1 <= s <= 48 and s % 10 != 9 for s in seeds))


class DecisionFrameTest(unittest.TestCase):
    def test_frame_is_written(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        env = FakeMineDojoEnv(cfg, {})
        with tempfile.TemporaryDirectory() as tmp:
            path = save_decision_frame(Path(tmp), 3, env.reset(), cfg)
            self.assertTrue(path.is_file())
            self.assertEqual(path.name, "00003.jpg")


if __name__ == "__main__":
    unittest.main()
