"""Ground-truth helpers and perception collection without MineDojo."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import json

from lfnav.config import load_config
from lfnav.data.collect_perception import capture, frames_done, make_fake
from lfnav.env.minedojo_env import _kwargs, convert_rays, episode_kwargs, lidar_plan
from lfnav.gt.calibrate_env_state import mapping_from_measurements, suggested_threshold
from lfnav.gt.judge import judge_episode
from lfnav.gt.scene_state import scene_state_from_obs
from lfnav.matcher import Matcher
from lfnav.types import ObjectBelief, TaskSpec

SPECS = Path(__file__).resolve().parents[2] / "lfnav" / "tasks" / "specs"
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


class MinedojoKwargsTest(unittest.TestCase):
    def test_default_config_builds_make_kwargs(self):
        kw = _kwargs(load_config(), 7)
        self.assertEqual(kw["task_id"], "harvest")
        self.assertEqual(kw["target_names"], "diamond")
        self.assertEqual(kw["target_quantities"], 100)
        self.assertEqual(kw["image_size"], (512, 820))
        self.assertEqual(kw["world_seed"], 7)
        self.assertEqual(kw["voxel_size"], dict(xmin=-5, ymin=-5, zmin=-5, xmax=5, ymax=5, zmax=5))
        self.assertTrue(kw["use_lidar"])
        self.assertEqual(kw["break_speed_multiplier"], 100.0)
        plan = lidar_plan(load_config())
        self.assertEqual(len(kw["lidar_rays"]), len(plan["lidar_rays"]))
        self.assertEqual(plan["aligned"], (0, len(plan["rows_cols"])))
        self.assertEqual(plan["wide"][0], plan["aligned"][1])

    def test_missing_task_id(self):
        with self.assertRaises(RuntimeError) as caught:
            _kwargs({"minedojo": {"image_size": [64, 64], "voxel_radius": 1}}, 0)
        self.assertIn("minedojo.task_id", str(caught.exception))

    def test_absent_section(self):
        with self.assertRaises(RuntimeError) as caught:
            _kwargs({}, 0)
        self.assertIn("minedojo.task_id", str(caught.exception))


def _load_specs():
    tasks = []
    for path in sorted(SPECS.glob("*.json")):
        tasks.append(TaskSpec.from_dict(json.loads(path.read_text())))
    return tasks


class RayConvertTest(unittest.TestCase):
    def _cfg(self):
        cfg = load_config()
        return cfg

    def _plan(self):
        return {
            "aligned": (0, 2),
            "wide": (2, 4),
            "rows_cols": [(0, 0), (0, 1)],
            "wide_angles_deg": [(0.0, -20.0), (10.0, 30.0)],
        }

    def _rays(self, entity_distance, block_distance):
        return {
            "block_name": ["log", "air", "tallgrass", "dirt"],
            "block_distance": [block_distance, 12.0, 6.0, 3.0],
            "entity_name": ["pig", "null", "cow", "sheep"],
            "entity_distance": [entity_distance, 0.0, 8.0, 1.0],
        }

    def test_entity_closer_than_block(self):
        converted = convert_rays(self._rays(2.0, 4.0), self._plan(), self._cfg())
        first, second = converted["aligned"]
        self.assertEqual((first["row"], first["col"], first["label"], first["distance"]), (0, 0, "pig", 2.0))
        self.assertEqual((second["row"], second["col"]), (0, 1))
        self.assertIsNone(second["label"])
        self.assertIsNone(second["distance"])
        grass, sheep = converted["wide"]
        self.assertEqual(grass["label"], "grass")
        self.assertEqual(grass["distance"], 6.0)
        self.assertEqual((grass["pitch"], grass["yaw"]), (0.0, -20.0))
        self.assertEqual(sheep["label"], "sheep")
        self.assertEqual(sheep["distance"], 1.0)
        self.assertEqual((sheep["pitch"], sheep["yaw"]), (10.0, 30.0))

    def test_entity_farther_than_block(self):
        converted = convert_rays(self._rays(5.0, 2.0), self._plan(), self._cfg())
        first = converted["aligned"][0]
        self.assertEqual(first["label"], "tree")
        self.assertEqual(first["distance"], 2.0)
        self.assertEqual(converted["wide"][0]["label"], "grass")

    def test_flowing_water_and_index_span(self):
        cfg = self._cfg()
        cfg["image_hw"] = [14, 28]
        cfg["wide_lidar"] = True
        cfg["wide_yaw_step"] = 180
        plan = lidar_plan(cfg)
        self.assertEqual(plan["rows_cols"], [(0, 0), (0, 1)])
        self.assertEqual(plan["aligned"], (0, 2))
        self.assertEqual(plan["wide"][0], 2)
        rays = {
            "block_name": ["flowing_water", "leaves", "air"],
            "block_distance": [1.5, 3.0, 9.0],
            "entity_name": ["", "null", ""],
            "entity_distance": [0.0, 0.0, 0.0],
        }
        converted = convert_rays(rays, plan, cfg)
        self.assertEqual(converted["aligned"][0]["label"], "water")
        self.assertEqual(converted["aligned"][1]["label"], "tree")
        self.assertIsNone(converted["wide"][0]["label"])


class EpisodeKwargsTest(unittest.TestCase):
    def test_sixteen_specs_match_scene(self):
        cfg = load_config()
        names = cfg["minedojo"]["param_names"]
        easy = cfg["difficulty_table"]["easy"]
        tasks = _load_specs()
        self.assertEqual(len(tasks), 16)
        for task in tasks:
            kw = episode_kwargs(task, cfg, 0)
            episode = task.episode
            self.assertEqual(kw[names["biome"]], episode["specified_biome"], task.id)
            self.assertEqual(kw[names["time"]], cfg["minedojo"]["time_values"][episode["start_time"]], task.id)
            self.assertEqual(kw[names["weather"]], episode["initial_weather"], task.id)
            mobs = [item["name"] for item in episode["spawn"] if item["kind"] == "mob"]
            blocks = [item["name"] for item in episode["spawn"] if item["kind"] == "block"]
            self.assertEqual(kw[names["mobs"]], mobs, task.id)
            for block in blocks:
                self.assertNotIn(block, kw[names["mobs"]], task.id)
            self.assertIs(kw[names["allow_mob_spawn"]], False, task.id)
            self.assertEqual(kw[names["mob_range_low"]], easy["spawn_distance_min"], task.id)
            self.assertEqual(kw[names["mob_range_high"]], easy["spawn_distance_max"], task.id)

    def test_make_kwargs_use_the_task_scene(self):
        cfg = load_config()
        names = cfg["minedojo"]["param_names"]
        night = next(task for task in _load_specs() if task.id == "3-4")
        kw = _kwargs(cfg, 11, night)
        self.assertEqual(kw["task_id"], "harvest")
        self.assertEqual(kw["world_seed"], 11)
        self.assertEqual(kw[names["biome"]], "plains")
        self.assertIs(kw[names["time"]], True)
        self.assertEqual(kw[names["weather"]], "rain")
        self.assertEqual(kw[names["mobs"]], ["pig"])

    def test_distractors_are_extra_mobs(self):
        cfg = load_config()
        cfg["difficulty"] = "medium"
        names = cfg["minedojo"]["param_names"]
        pig = next(task for task in _load_specs() if task.id == "1-4")
        kw = episode_kwargs(pig, cfg, 1)
        mobs = kw[names["mobs"]]
        self.assertEqual(mobs[0], "pig")
        self.assertEqual(len(mobs), 1 + cfg["difficulty_table"]["medium"]["distractors"])
        self.assertTrue(set(mobs[1:]).isdisjoint({"pig"}))
        self.assertEqual(kw[names["mob_range_low"]], cfg["difficulty_table"]["medium"]["spawn_distance_min"])


class EnvStateTest(unittest.TestCase):
    def _cfg(self, **overrides):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        cfg.update(overrides)
        return cfg

    def _obs(self, **loc):
        stats = {"pos": [0, 64, 0], "yaw": 0.0, "pitch": 0.0}
        stats.update(loc)
        return {"location_stats": stats, "lfnav_rays": []}

    def test_thresholds_decode_time_light_biome(self):
        cfg = self._cfg(time_day_min=0.5, brightness_min_light=8)
        day = scene_state_from_obs(self._obs(sun_brightness=0.9, light_level=12, biome_id=4, is_raining=False), cfg)
        self.assertEqual(day.time.get("day"), 1.0)
        self.assertEqual(day.brightness.get("sufficient"), 1.0)
        self.assertEqual(day.biome.get("forest"), 1.0)
        self.assertEqual(day.weather.get("clear"), 1.0)
        self.assertEqual(day.unknown, [])
        night = scene_state_from_obs(self._obs(sun_brightness=0.1, light_level=3, biome_id=35, is_raining=True), cfg)
        self.assertEqual(night.time.get("night"), 1.0)
        self.assertEqual(night.brightness.get("insufficient"), 1.0)
        self.assertEqual(night.biome.get("savanna"), 1.0)
        self.assertEqual(night.weather.get("rain"), 1.0)

    def test_unset_thresholds_are_unknown(self):
        cfg = self._cfg(time_day_min=None, brightness_min_light=None)
        cfg["obs_keys"] = dict(cfg["obs_keys"])
        cfg["obs_keys"]["biome_id_to_name"] = {}
        task = TaskSpec.from_dict(json.loads((SPECS / "3-4.json").read_text()))
        scene = scene_state_from_obs(
            self._obs(sun_brightness=0.8, light_level=14, biome_id=1, is_raining=True), cfg,
        )
        self.assertIn("time", scene.unknown)
        self.assertIn("brightness", scene.unknown)
        self.assertIn("biome", scene.unknown)
        self.assertEqual(scene.time, {})
        self.assertEqual(scene.brightness, {})
        self.assertEqual(scene.biome, {})
        self.assertEqual(scene.weather.get("rain"), 1.0)
        scene.objects["pig"] = ObjectBelief(1.0, {"center": 1.0}, {"near": 1.0})
        with self.assertLogs("lfnav.matcher", level="WARNING"):
            match = Matcher().match(task, scene)
        self.assertIn("time:night:unknown", match.missing)
        self.assertNotIn("time:night", match.missing)
        self.assertEqual(match.status, "not_found")
        with self.assertLogs("lfnav.gt.judge", level="WARNING"):
            verdict = judge_episode(task, True, 4, scene, 2.0, 4.0, [0], [2])
        self.assertFalse(verdict["sr_a"])
        self.assertIn("time:night:unknown", verdict["missing"])
        self.assertNotIn("time:night", verdict["missing"])

    def test_calibration_helpers(self):
        mapping = mapping_from_measurements([
            {"specified_biome": "forest", "biome_id": 4},
            {"specified_biome": "plains", "biome_id": 1},
            {"specified_biome": "forest", "biome_id": 4},
        ])
        self.assertEqual(mapping, {"4": "forest", "1": "plains"})
        self.assertEqual(suggested_threshold([0.0, 0.1], [0.8, 1.0]), 0.45)


if __name__ == "__main__":
    unittest.main()
