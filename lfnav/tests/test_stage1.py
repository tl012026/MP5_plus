"""Stage 1 tests. No MineDojo and no torch."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from lfnav.aggregate import aggregate_object, column_direction, distance_bin
from lfnav.config import load_config
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.env.minedojo_env import convert_rays
from lfnav.geom import center_ray, patch_grid, ray_directions
from lfnav.gt.judge import aligned_target_distance, judge_episode, spl_approx
from lfnav.gt.scene_state import scene_state_from_obs
from lfnav.matcher import Matcher
from lfnav.sweep_thresholds import sweep
from lfnav.types import MatchResult, ObjectBelief, PolicyInput, PolicyOutput, SceneState, TaskSpec

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "lfnav" / "tasks" / "specs"


def scene_pig(grass=1.0, near=1.0, biome="plains", time="day", weather="clear", brightness="sufficient"):
    return SceneState(
        objects={
            "pig": ObjectBelief(0.95, {"left": 0.0, "center": 1.0, "right": 0.0},
                                {"near": near, "mid": 1.0 - near, "far": 0.0}),
            "grass": ObjectBelief(grass, {"left": 0.0, "center": 1.0, "right": 0.0},
                                  {"near": 1.0, "mid": 0.0, "far": 0.0}),
        },
        biome={biome: 1.0},
        time={time: 1.0},
        weather={weather: 1.0},
        brightness={brightness: 1.0},
    )


def task_4_4():
    return TaskSpec.from_dict(json.loads((SPECS / "4-4.json").read_text()))


class TypesTest(unittest.TestCase):
    def test_roundtrip(self):
        task = task_4_4()
        scene = scene_pig()
        match = MatchResult("found", True, {"target": True}, [], "center", "near", 0.9)
        pin = PolicyInput(None, task.to_dict(), match.to_dict(), scene.to_dict(), {"grid": [1]}, ["forward"])
        pout = PolicyOutput(["forward", "turn_left_15"], [])
        for obj in (task, scene, match, pin, pout):
            again = type(obj).from_dict(json.loads(json.dumps(obj.to_dict())))
            self.assertEqual(again.to_dict(), obj.to_dict())

    def test_policy_output_requires_one(self):
        with self.assertRaises(ValueError):
            PolicyOutput([], [])

    def test_sixteen_specs(self):
        files = sorted(SPECS.glob("*.json"))
        self.assertEqual(len(files), 16)
        ids = [TaskSpec.from_dict(json.loads(p.read_text())).id for p in files]
        self.assertEqual(ids, [p.stem for p in files])
        grass = TaskSpec.from_dict(json.loads((SPECS / "2-2.json").read_text()))
        self.assertEqual(grass.target, "grass")
        self.assertEqual(grass.conditions["near"], ["pig"])


class MatcherTest(unittest.TestCase):
    def test_found_and_stop_sr_b(self):
        result = Matcher({"match": {"stop_mode": "sr_b"}}).match(task_4_4(), scene_pig())
        self.assertEqual(result.status, "found")
        self.assertTrue(result.stop)
        self.assertEqual(result.target_direction, "center")
        self.assertEqual(result.target_distance, "near")

    def test_missing_condition(self):
        result = Matcher().match(task_4_4(), scene_pig(grass=0.1))
        self.assertEqual(result.status, "not_found")
        self.assertFalse(result.stop)
        self.assertIn("near:grass", result.missing)
        self.assertEqual(result.target_direction, "center")
        self.assertEqual(result.target_distance, "near")

    def test_unseen_target_has_no_direction(self):
        scene = scene_pig()
        scene.objects["pig"] = ObjectBelief(0.1, {"left": 1.0, "center": 0.0, "right": 0.0},
                                           {"near": 1.0, "mid": 0.0, "far": 0.0})
        result = Matcher().match(task_4_4(), scene)
        self.assertEqual(result.status, "not_found")
        self.assertIsNone(result.target_direction)
        self.assertIsNone(result.target_distance)

    def test_sr_a_stops_when_far(self):
        far = scene_pig(near=0.1)
        sr_b = Matcher({"match": {"stop_mode": "sr_b", "tau_near": 0.5}}).match(task_4_4(), far)
        sr_a = Matcher({"match": {"stop_mode": "sr_a"}}).match(task_4_4(), far)
        self.assertEqual(sr_b.status, "found")
        self.assertFalse(sr_b.stop)
        self.assertTrue(sr_a.stop)

    def test_majority_vote(self):
        matcher = Matcher({"match": {"smooth_n": 3, "stop_mode": "sr_a"}})
        bad = scene_pig(grass=0.0)
        good = scene_pig()
        self.assertFalse(matcher.match(task_4_4(), bad).status == "found")
        self.assertFalse(matcher.match(task_4_4(), bad).status == "found")
        voted = matcher.match(task_4_4(), good)
        self.assertEqual(voted.status, "not_found")
        matcher.reset()
        matcher.match(task_4_4(), good)
        matcher.match(task_4_4(), good)
        self.assertEqual(matcher.match(task_4_4(), bad).status, "found")


class JudgeTest(unittest.TestCase):
    def test_sr_b_and_spl(self):
        verdict = judge_episode(
            task_4_4(), True, 10, scene_pig(), 2.0, 12.0, [0, 64, 0], [0, 64, 6],
            initial_target_distance=9.0, radius=3.0,
        )
        self.assertTrue(verdict["sr_b"])
        self.assertTrue(verdict["sr_a"])
        self.assertFalse(verdict["false_stop"])
        self.assertAlmostEqual(verdict["spl"], 6.0 / 12.0)

    def test_too_far_is_not_false_stop(self):
        verdict = judge_episode(task_4_4(), True, 10, scene_pig(), 8.0, 8.0, [0, 0, 0], [0, 0, 8], radius=3)
        self.assertTrue(verdict["sr_a"])
        self.assertFalse(verdict["sr_b"])
        self.assertFalse(verdict["false_stop"])
        self.assertEqual(verdict["outcome"], "fail_too_far")

    def test_false_stop_and_timeout(self):
        bad = judge_episode(task_4_4(), True, 4, scene_pig(grass=0.0), 1.0, 1.0, [0], [1])
        self.assertEqual(bad["outcome"], "false_stop")
        self.assertTrue(bad["false_stop"])
        late = judge_episode(task_4_4(), False, 3000, None, None, 10, [0], [1], max_steps=3000, initial_target_distance=10)
        self.assertTrue(late["timeout"])
        self.assertEqual(late["spl"], 0.0)
        beyond = judge_episode(
            task_4_4(), True, 10, scene_pig(), 2.0, 12.0, [0, 64, 0], [0, 64, 6],
            initial_target_distance="> 32",
        )
        self.assertIsNone(beyond["spl"])

    def test_spl_zero_length(self):
        self.assertEqual(spl_approx(True, 0.0, 0.0), 1.0)
        self.assertEqual(spl_approx(False, 3.0, 3.0), 0.0)
        self.assertIsNone(spl_approx(True, None, 12.0))


class StopDistanceTest(unittest.TestCase):
    """SR-b distance comes from converted aligned rays, not env.distance_to."""

    def _task(self):
        return TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))

    def _cfg(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 28]
        return cfg

    def _obs(self, entity_distance, entity="pig", block_name="air", block_distance=12.0, extra_wide_pig=None):
        cfg = self._cfg()
        plan = {
            "aligned": (0, 1),
            "wide": (1, 2),
            "rows_cols": [(0, 0)],
            "wide_angles_deg": [(0.0, 90.0)],
        }
        wide_entity = "pig" if extra_wide_pig is not None else ""
        wide_dist = extra_wide_pig if extra_wide_pig is not None else 0.0
        rays = {
            "block_name": [block_name, "air"],
            "block_distance": [block_distance, 12.0],
            "entity_name": [entity, wide_entity],
            "entity_distance": [entity_distance, wide_dist],
        }
        converted = convert_rays(rays, plan, cfg)
        key = (cfg.get("obs_keys") or {}).get("aligned_rays") or "lfnav_rays"
        wide_key = (cfg.get("obs_keys") or {}).get("wide_rays") or "lfnav_wide_rays"
        return {key: converted["aligned"], wide_key: converted["wide"]}, cfg

    def _verdict(self, obs, cfg):
        task = self._task()
        scene = scene_state_from_obs(obs, cfg)
        dist = aligned_target_distance(obs, cfg, task.target)
        return judge_episode(task, True, 8, scene, dist, 8.0, [0, 64, 0], [0, 64, 2])

    def test_two_blocks_is_sr_b(self):
        rec = self._verdict(*self._obs(2.0))
        self.assertAlmostEqual(rec["distance_blocks"], 2.0)
        self.assertTrue(rec["sr_a"])
        self.assertTrue(rec["sr_b"])
        self.assertTrue(rec["success"])

    def test_five_blocks_is_sr_a_only(self):
        rec = self._verdict(*self._obs(5.0))
        self.assertAlmostEqual(rec["distance_blocks"], 5.0)
        self.assertTrue(rec["sr_a"])
        self.assertFalse(rec["sr_b"])
        self.assertEqual(rec["outcome"], "fail_too_far")

    def test_out_of_view_fails_both(self):
        rec = self._verdict(*self._obs(2.0, entity="", extra_wide_pig=1.0))
        self.assertIsNone(rec["distance_blocks"])
        self.assertFalse(rec["sr_a"])
        self.assertFalse(rec["sr_b"])
        self.assertEqual(rec["outcome"], "false_stop")

    def test_fake_in_view_matches_euclidean(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": 2.0}]})
        obs = env.reset()
        euclid = env.distance_to("pig")
        ray = aligned_target_distance(obs, cfg, "pig")
        self.assertAlmostEqual(euclid, 2.0)
        self.assertIsNotNone(ray)
        self.assertAlmostEqual(euclid, ray, delta=0.2)

    def test_fake_behind_agent_is_not_in_view(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": -2.0}]})
        obs = env.reset()
        self.assertAlmostEqual(env.distance_to("pig"), 2.0)
        self.assertIsNone(aligned_target_distance(obs, cfg, "pig"))


class GeomTest(unittest.TestCase):
    def test_grid_and_center_ray(self):
        self.assertEqual(patch_grid(224, 364, 14), (16, 26))
        n_h, n_w = 4, 6
        rays = ray_directions(n_h, n_w, 70, "vertical")
        center = center_ray(n_h, n_w, 70, "vertical", n_h // 2, n_w // 2)
        self.assertGreater(center[2], abs(center[0]))
        self.assertGreater(center[2], abs(center[1]))
        left = rays[n_h // 2][0]
        right = rays[n_h // 2][-1]
        self.assertLess(left[0], 0)
        self.assertGreater(right[0], 0)
        self.assertTrue(math.isclose(sum(c * c for c in center), 1.0, abs_tol=1e-6))

    def test_horizontal_fov_changes_vertical(self):
        vert = ray_directions(4, 8, 90, "horizontal")[0][0]
        self.assertLess(abs(vert[1]), 1.0)


class AggregateTest(unittest.TestCase):
    def test_present_direction_distance(self):
        present = [
            [0.0, 0.0, 0.2],
            [0.0, 0.9, 0.0],
        ]
        dist = [
            ["far", "far", "far"],
            ["mid", "near", "mid"],
        ]
        belief = aggregate_object(present, dist)
        self.assertAlmostEqual(belief.present, 0.9)
        self.assertEqual(column_direction(0, 3), "left")
        self.assertEqual(column_direction(1, 3), "center")
        self.assertEqual(column_direction(2, 3), "right")
        self.assertGreater(belief.direction["center"], belief.direction["right"])
        self.assertEqual(max(belief.distance, key=belief.distance.get), "near")
        self.assertEqual(distance_bin(3), "near")
        self.assertEqual(distance_bin(3.1), "mid")
        self.assertEqual(distance_bin(11), "far")

    def test_ignore_skips_present_and_direction(self):
        present = [[1.0, 0.0], [0.0, 1.0]]
        ignore = [[True, False], [False, False]]
        belief = aggregate_object(present, ignore_hw=ignore)
        self.assertEqual(belief.present, 1.0)
        self.assertEqual(max(belief.direction, key=belief.direction.get), "right")


class SweepAndEnvTest(unittest.TestCase):
    def test_sweep_prefers_separating_threshold(self):
        good = scene_pig().to_dict()
        bad = scene_pig(grass=0.0).to_dict()
        task = task_4_4().to_dict()
        rows = [
            {"task": task, "scene": good, "found": True, "stop": True},
            {"task": task, "scene": bad, "found": False, "stop": False},
        ]
        ranked = sweep(rows, [0.5], "sr_b")
        self.assertEqual(ranked[0]["found_acc"], 1.0)

    def test_fake_env_sees_target_ahead(self):
        cfg = {
            "image_hw": [14, 14],
            "patch": 14,
            "fov_deg": 70,
            "fov_axis": "vertical",
            "obs_keys": {
                "rgb": "rgb",
                "position": ["location_stats", "pos"],
                "yaw": ["location_stats", "yaw"],
                "pitch": ["location_stats", "pitch"],
                "voxels": ["voxels", "block_name"],
                "aligned_rays": "lfnav_rays",
                "wide_rays": "lfnav_wide_rays",
                "biome_name": "biome_name",
                "time_name": "time_name",
                "weather_name": "weather_name",
                "brightness_name": "brightness_name",
            },
            "actions": {"vectors": {"forward": [1, 0, 0, 12, 12, 0, 0, 0]}},
        }
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": 2.5}]})
        obs = env.reset()
        self.assertEqual(len(obs["rgb"]), 14)
        self.assertIn("pig", {hit["label"] for hit in obs["lfnav_rays"]})
        obs2, _r, _d, info = env.step([1, 0, 0, 12, 12, 0, 0, 0])
        self.assertGreater(obs2["location_stats"]["pos"][2], 0.5)
        self.assertFalse(info["blocked"])

    def test_load_default_config(self):
        cfg = load_config()
        self.assertIn("tree", cfg["labels"]["objects"])
        self.assertEqual(cfg["difficulty"], "easy")
        self.assertIn("easy", cfg["difficulty_table"])
        self.assertTrue(cfg["obs_keys"]["confirmed"])


if __name__ == "__main__":
    unittest.main()
