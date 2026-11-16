"""Occluders in front of an animal, and old scenes without that field."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

torch.set_num_threads(4)

from lfnav.config import load_config
from lfnav.data.collect_demos import apply_difficulty_max_steps
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.gt.judge import judge_episode
from lfnav.gt.scene_state import is_occluder_name, occluder_at_stop, scene_state_from_obs
from lfnav.policy.c2nav import patch_targets, vis_labels
from lfnav.policy.train_c2nav import animal_occluder_supervision, nav_loss, occluder_frac_of
from lfnav.types import TaskSpec


class OccluderRayTest(unittest.TestCase):
    def test_tallgrass_in_front_of_cow_is_ignored_for_animals(self):
        cfg = load_config()
        cfg["aligned_subrays"] = 1
        self.assertIn("tallgrass", cfg["labels"]["aliases"]["grass"])
        self.assertNotIn("double_plant", cfg["labels"]["aliases"]["grass"])
        env = FakeMineDojoEnv(cfg, {
            "blocks": {(0, 2): "tallgrass", (3, 3): "grass block", (-3, 4): "double_plant"},
            "entities": [{"name": "cow", "x": 0.0, "z": 5.0, "radius": 0.35}],
        })
        obs = env.reset()
        scene = scene_state_from_obs(obs, cfg)
        both = None
        grass_block = None
        plant = None
        n_h, n_w = len(scene.occluder), len(scene.occluder[0])
        for i in range(n_h):
            for j in range(n_w):
                cow_d = scene.patch_dist_by_class["cow"][i][j]
                hud = bool(scene.ignore[i][j])
                if both is None and scene.occluder[i][j] >= 0.5 and cow_d is not None and not hud:
                    both = (i, j, cow_d)
                if grass_block is None and scene.patch_dist_by_class["grass"][i][j] is not None and scene.occluder[i][j] < 0.5:
                    grass_block = (i, j)
                if plant is None and scene.occluder[i][j] >= 0.5 and scene.patch_dist_by_class["grass"][i][j] is None:
                    plant = (i, j)
        self.assertIsNotNone(both)
        i, j, cow_d = both
        self.assertLess(scene.patch_dist[i][j], cow_d)
        self.assertGreaterEqual(scene.patches["grass"][i][j], 0.5)
        self.assertLess(scene.patches["cow"][i][j], 0.5)
        self.assertIsNotNone(grass_block)
        self.assertIsNotNone(plant)
        self.assertFalse(is_occluder_name("grass block"))
        self.assertTrue(is_occluder_name("double_plant"))
        data = scene.to_dict()
        objects = list(cfg["labels"]["objects"])
        present, log_d, ignore = patch_targets(data, objects, 1, (n_h, n_w))
        vis, vis_ig = vis_labels(present, ignore)
        mask, vis2, ig2 = animal_occluder_supervision(
            data, objects, present, ignore, vis, vis_ig, occluder_frac_of(cfg),
        )
        cow = objects.index("cow")
        grass = objects.index("grass")
        self.assertGreaterEqual(float(mask[cow, i, j]), 0.5)
        self.assertLess(float(mask[grass, i, j]), 0.5)
        self.assertAlmostEqual(float(log_d[cow, i, j]), math.log(cow_d), places=4)
        self.assertAlmostEqual(float(log_d[grass, i, j]), math.log(scene.patch_dist[i][j]), places=4)
        self.assertEqual(float(vis2[cow]), 1.0)
        self.assertEqual(float(ig2[cow]), 0.0)
        clear = np.argwhere(mask[cow] < 0.5)
        self.assertGreater(len(clear), 0)
        self.assertTrue(occluder_at_stop(scene, max_dist=3.0))
        verdict = judge_episode(TaskSpec(id="t", source="", description="", target="cow"), True, 4, scene, 5.0, 4.0, [0], [1])
        self.assertNotIn("occluder_at_stop", verdict)

    def test_unseen_animal_with_near_occluders_skips_frame_vis(self):
        objects = ["pig", "cow", "sheep", "chicken", "grass"]
        occ = [[0.0, 1.0], [0.0, 0.0]]
        dist = [[None, 2.0], [None, None]]
        scene = {
            "occluder": occ,
            "patch_dist": dist,
            "patch_dist_by_class": {name: [[None, None], [None, None]] for name in objects},
            "ignore": [[0.0, 0.0], [0.0, 0.0]],
            "patches": {name: [[0.0, 0.0], [0.0, 0.0]] for name in objects},
        }
        present, log_d, ignore = patch_targets(scene, objects, 1, (2, 2))
        vis, vis_ig = vis_labels(present, ignore)
        _mask, vis2, ig2 = animal_occluder_supervision(scene, objects, present, ignore, vis, vis_ig, 0.1)
        cow = objects.index("cow")
        grass = objects.index("grass")
        self.assertEqual(float(ig2[cow]), 1.0)
        self.assertEqual(float(ig2[grass]), 0.0)
        self.assertEqual(float(vis2[cow]), 0.0)
        self.assertTrue(np.all(log_d == 0))
        _mask, _vis, ig_low = animal_occluder_supervision(scene, objects, present, ignore, vis, vis_ig, 0.5)
        self.assertEqual(float(ig_low[cow]), 0.0)
        self.assertFalse(occluder_at_stop({
            "occluder": [[0.0, 1.0, 0.0]],
            "patch_dist": [[None, 9.0, None]],
            "ignore": [[0.0, 0.0, 0.0]],
        }))

    def test_old_scene_without_occluder_is_unchanged(self):
        objects = ["cow", "grass"]
        patches = {
            "cow": [[0.0, 1.0], [0.0, 0.0]],
            "grass": [[0.0, 0.0], [0.0, 0.0]],
        }
        scene = {"patches": patches, "ignore": [[0.0, 0.0], [0.0, 0.0]]}
        present, log_d, ignore = patch_targets(scene, objects, 1, (2, 2))
        vis, vis_ig = vis_labels(present, ignore)
        mask, vis2, ig2 = animal_occluder_supervision(scene, objects, present, ignore, vis, vis_ig, 0.1)
        self.assertIsNone(mask)
        self.assertIs(vis2, vis)
        self.assertIs(ig2, vis_ig)
        self.assertEqual(float(vis[0]), 1.0)
        self.assertAlmostEqual(float(log_d[0, 0, 1]), math.log(8.0), places=5)
        self.assertEqual(float(occluder_frac_of({})), 0.1)
        self.assertEqual(float(occluder_frac_of({"c2nav": {"occluder_frac": 0}})), 0.0)

    def test_ignored_cell_is_not_a_negative(self):
        logit = torch.zeros(1, 1, 1, 1)
        out = {
            "present_logit": logit,
            "log_dist": torch.zeros(1, 1, 1, 1),
            "biome": torch.zeros(1, 2),
            "time": torch.zeros(1, 2),
            "weather": torch.zeros(1, 2),
            "brightness": torch.zeros(1, 2),
            "vis_logit": torch.zeros(1, 1),
        }
        batch = {
            "present": torch.zeros(1, 1, 1, 1),
            "log_dist": torch.zeros(1, 1, 1, 1),
            "ignore": torch.zeros(1, 1, 1),
            "biome": torch.zeros(1, dtype=torch.long),
            "time": torch.zeros(1, dtype=torch.long),
            "weather": torch.zeros(1, dtype=torch.long),
            "brightness": torch.zeros(1, dtype=torch.long),
            "vis": torch.zeros(1, 1),
            "vis_ignore": torch.zeros(1, 1),
            "has_c": torch.zeros(1),
            "used_wide": torch.zeros(1),
            "stop": torch.zeros(1),
            "a_only": torch.ones(1),
            "target_index": torch.zeros(1, dtype=torch.long),
        }
        weights = {"a": 1.0, "mode": 1.0, "wp": 1.0, "stop": 1.0}
        ones = torch.ones(1)
        _loss, plain = nav_loss(out, batch, weights, ones, ones, ones, ones, torch.tensor(1.0), 0.0, skip_c=True)
        covered = dict(batch)
        covered["present_ignore"] = torch.ones(1, 1, 1, 1)
        _loss, masked = nav_loss(out, covered, weights, ones, ones, ones, ones, torch.tensor(1.0), 0.0, skip_c=True)
        self.assertGreater(plain["present"], 0.0)
        self.assertEqual(masked["present"], 0.0)
        self.assertEqual(masked["vis"], plain["vis"])

    def test_hard_collection_max_steps(self):
        cfg = load_config()
        self.assertEqual(int(cfg["difficulty_table"]["hard"]["max_steps"]), 1500)
        hard = apply_difficulty_max_steps(cfg, "hard")
        self.assertEqual(int(hard["judge"]["max_steps"]), 1500)
        easy = apply_difficulty_max_steps(cfg, "easy")
        self.assertEqual(int(easy["judge"]["max_steps"]), int(cfg["judge"]["max_steps"]))
        self.assertIsNone((cfg["difficulty_table"]["easy"]).get("max_steps"))


if __name__ == "__main__":
    unittest.main()
