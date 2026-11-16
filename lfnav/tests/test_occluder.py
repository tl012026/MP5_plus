"""Occluders in front of an animal, and old scenes without that field."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

torch.set_num_threads(4)

from lfnav.config import load_config
from lfnav.data.collect_demos import apply_difficulty_max_steps
from lfnav.env.minedojo_env import convert_rays
from lfnav.gt.scene_state import is_occluder_name, occluder_at_stop, occluder_names_of, scene_state_from_obs
from lfnav.types import SceneState
from lfnav.policy.c2nav import patch_targets, vis_labels
from lfnav.policy.train_c2nav import animal_occluder_supervision, nav_loss, occluder_frac_of


class OccluderRayTest(unittest.TestCase):
    def _one_cell(self, block_name, block_dist, entity_name, entity_dist):
        cfg = load_config()
        plan = {
            "aligned": (0, 1),
            "wide": (1, 1),
            "rows_cols": [(0, 0)],
            "wide_angles_deg": [],
            "subrays": 1,
        }
        rays = {
            "block_name": [block_name],
            "block_distance": [block_dist],
            "entity_name": [entity_name],
            "entity_distance": [entity_dist],
        }
        converted = convert_rays(rays, plan, cfg)
        key = (cfg.get("obs_keys") or {}).get("aligned_rays") or "lfnav_rays"
        scene = scene_state_from_obs({key: converted["aligned"]}, cfg)
        data = scene.to_dict()
        objects = list(cfg["labels"]["objects"])
        n_h, n_w = len(scene.occluder), len(scene.occluder[0])
        present, log_d, ignore = patch_targets(data, objects, 1, (n_h, n_w))
        vis, vis_ig = vis_labels(present, ignore)
        mask, vis2, ig2 = animal_occluder_supervision(
            data, objects, present, ignore, vis, vis_ig, occluder_frac_of(cfg),
        )
        cow = objects.index("cow")
        return {
            "cfg": cfg,
            "ray": converted["aligned"][0],
            "scene": scene,
            "objects": objects,
            "cow": cow,
            "present": present,
            "log_d": log_d,
            "mask": mask,
            "vis": vis2,
            "vis_ig": ig2,
        }

    def test_tallgrass_hides_cow(self):
        out = self._one_cell("tallgrass", 1.5, "cow", 2.5)
        ray = out["ray"]
        cow = out["cow"]
        self.assertEqual([(hit["raw"], hit["distance"]) for hit in ray["visible_hits"]], [("tallgrass", 1.5)])
        self.assertEqual([(hit["raw"], hit["distance"]) for hit in ray["occluded_hits"]], [("cow", 2.5)])
        self.assertIsNone(out["scene"].patch_dist_by_class["cow"][0][0])
        self.assertEqual(out["scene"].patch_dist_by_class["grass"][0][0], 1.5)
        self.assertGreaterEqual(out["scene"].occluded_by_class["cow"][0][0], 0.5)
        self.assertEqual(float(out["present"][cow, 0, 0]), 0.0)
        self.assertEqual(float(out["log_d"][cow, 0, 0]), 0.0)
        self.assertGreaterEqual(float(out["mask"][cow, 0, 0]), 0.5)
        self.assertEqual(float(out["vis"][cow]), 0.0)
        self.assertEqual(float(out["vis_ig"][cow]), 1.0)
        self.assertLess(float(out["mask"][out["objects"].index("grass"), 0, 0]), 0.5)
        self.assertIn("tallgrass", out["cfg"]["labels"]["aliases"]["grass"])
        self.assertFalse(is_occluder_name("grass block"))

    def test_solid_block_hides_cow_as_negative(self):
        out = self._one_cell("dirt", 1.5, "cow", 2.5)
        ray = out["ray"]
        cow = out["cow"]
        self.assertEqual(ray["visible_hits"][0]["raw"], "dirt")
        self.assertEqual(ray["occluded_hits"], [])
        self.assertIsNone(out["scene"].patch_dist_by_class["cow"][0][0])
        self.assertLess(out["scene"].occluded_by_class["cow"][0][0], 0.5)
        self.assertEqual(float(out["present"][cow, 0, 0]), 0.0)
        self.assertEqual(float(out["log_d"][cow, 0, 0]), 0.0)
        self.assertLess(float(out["mask"][cow, 0, 0]), 0.5)
        self.assertEqual(float(out["vis"][cow]), 0.0)
        self.assertEqual(float(out["vis_ig"][cow]), 0.0)

    def test_cow_in_front_is_visible(self):
        out = self._one_cell("air", 9.0, "cow", 2.0)
        cow = out["cow"]
        self.assertEqual([(hit["raw"], hit["distance"]) for hit in out["ray"]["visible_hits"]], [("cow", 2.0)])
        self.assertEqual(out["ray"]["occluded_hits"], [])
        self.assertEqual(out["scene"].patch_dist_by_class["cow"][0][0], 2.0)
        self.assertLess(out["scene"].occluded_by_class["cow"][0][0], 0.5)
        self.assertGreaterEqual(float(out["present"][cow, 0, 0]), 0.5)
        self.assertAlmostEqual(float(out["log_d"][cow, 0, 0]), math.log(2.0), places=5)
        self.assertLess(float(out["mask"][cow, 0, 0]), 0.5)
        self.assertEqual(float(out["vis"][cow]), 1.0)
        self.assertEqual(float(out["vis_ig"][cow]), 0.0)

    def test_subrays_are_split_then_merged(self):
        cfg = load_config()
        plan = {
            "aligned": (0, 4),
            "wide": (4, 4),
            "rows_cols": [(0, 0)],
            "wide_angles_deg": [],
            "subrays": 2,
        }
        rays = {
            "block_name": ["tallgrass", "air", "air", "air"],
            "block_distance": [1.5, 9.0, 9.0, 9.0],
            "entity_name": ["cow", "cow", "null", "null"],
            "entity_distance": [4.0, 2.0, 0.0, 0.0],
        }
        ray = convert_rays(rays, plan, cfg)["aligned"][0]
        visible = [(hit["raw"], hit["distance"]) for hit in ray["visible_hits"]]
        occluded = [(hit["raw"], hit["distance"]) for hit in ray["occluded_hits"]]
        self.assertIn(("tallgrass", 1.5), visible)
        self.assertIn(("cow", 2.0), visible)
        self.assertIn(("cow", 4.0), occluded)
        self.assertNotIn(("cow", 4.0), visible)

    def test_hit_names_counts_each_raw_hit(self):
        cfg = load_config()
        key = cfg["obs_keys"]["aligned_rays"]
        rays = [
            {
                "row": 0, "col": 0, "label": "tallgrass", "distance": 1.5, "raw": "tallgrass",
                "hits": [
                    {"raw": "minecraft:TallGrass", "distance": 1.5, "kind": "block"},
                    {"raw": "Cow", "distance": 2.5, "kind": "entity"},
                    {"raw": "cow", "distance": 4.0, "kind": "entity"},
                ],
            },
            {
                "row": 0, "col": 1, "label": "grass", "distance": 3.0, "raw": "grass block",
                "hits": [{"raw": "grass block", "distance": 3.0, "kind": "block"}],
            },
            {"row": 0, "col": 0, "label": "cow", "distance": 2.5, "raw": "cow"},
        ]
        scene = scene_state_from_obs({key: rays}, cfg)
        self.assertEqual(scene.hit_names, {"tallgrass": 1, "cow": 2, "grass block": 1})
        again = SceneState.from_dict(scene.to_dict())
        self.assertEqual(again.hit_names, scene.hit_names)

    def test_minedojo_display_names_in_labels(self):
        cfg = load_config()
        names = occluder_names_of(cfg)
        for raw in ("grass", "plant", "flower", "dead bush", "tallgrass", "tall grass", "double_plant"):
            self.assertTrue(is_occluder_name(raw, names), raw)
        self.assertFalse(is_occluder_name("grass block", names))
        self.assertFalse(is_occluder_name("minecraft:Grass Block", names))

    def test_custom_occluder_names(self):
        self.assertTrue(is_occluder_name("minecraft:Tall Grass", ("tall_grass",)))
        self.assertFalse(is_occluder_name("tallgrass", ("dirt",)))
        self.assertFalse(is_occluder_name("tallgrass", ()))
        bare = dict(load_config())
        labels = dict(bare["labels"])
        labels.pop("occluder_names", None)
        bare["labels"] = labels
        self.assertEqual(occluder_names_of(bare), ("tallgrass", "tall_grass", "double_plant"))
        cfg = dict(load_config())
        cfg["labels"] = dict(cfg["labels"])
        cfg["labels"]["occluder_names"] = ["dirt"]
        plan = {
            "aligned": (0, 1),
            "wide": (1, 1),
            "rows_cols": [(0, 0)],
            "wide_angles_deg": [],
            "subrays": 1,
        }
        key = cfg["obs_keys"]["aligned_rays"]
        hidden = convert_rays({
            "block_name": ["dirt"],
            "block_distance": [1.5],
            "entity_name": ["cow"],
            "entity_distance": [2.5],
        }, plan, cfg)
        scene = scene_state_from_obs({key: hidden["aligned"]}, cfg)
        self.assertGreaterEqual(scene.occluded_by_class["cow"][0][0], 0.5)
        self.assertEqual(scene.hit_names, {"dirt": 1, "cow": 1})
        shown = convert_rays({
            "block_name": ["tallgrass"],
            "block_distance": [1.5],
            "entity_name": ["cow"],
            "entity_distance": [2.5],
        }, plan, cfg)
        open_scene = scene_state_from_obs({key: shown["aligned"]}, cfg)
        self.assertLess(open_scene.occluded_by_class["cow"][0][0], 0.5)
        self.assertIsNone(open_scene.patch_dist_by_class["cow"][0][0])

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
