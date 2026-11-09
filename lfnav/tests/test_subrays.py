"""Aligned n×n sub-rays: n=1 matches the old patch-centre rays; n=3 fills gaps."""

from __future__ import annotations

import unittest

from lfnav.config import load_config
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.env.minedojo_env import aligned_lidar_rays, convert_rays, lidar_plan
from lfnav.geom import ray_directions, subray_directions, subray_grid_from_cfg
from lfnav.gt.judge import aligned_target_distance
from lfnav.gt.scene_state import scene_state_from_obs


def _tiny_cfg(**extra):
    cfg = load_config()
    cfg["image_hw"] = [14, 14]
    cfg["patch"] = 14
    cfg["wide_lidar"] = False
    cfg["aligned_ray_range"] = 32
    cfg.update(extra)
    return cfg


class SubrayGeomTest(unittest.TestCase):
    def test_n1_matches_patch_centres(self):
        n_h, n_w = 16, 26
        old = ray_directions(n_h, n_w, 70, "vertical")
        grid = subray_directions(n_h, n_w, 70, "vertical", n_sub=1)
        for i in range(n_h):
            for j in range(n_w):
                self.assertEqual(len(grid[i][j]), 1)
                for a, b in zip(old[i][j], grid[i][j][0]):
                    self.assertAlmostEqual(a, b, places=12)

    def test_n1_lidar_plan_matches_old_length(self):
        cfg = {"image_hw": [224, 364], "patch": 14, "fov_deg": 70, "fov_axis": "vertical", "aligned_subrays": 1, "wide_lidar": False}
        self.assertEqual(len(aligned_lidar_rays(cfg)), (224 // 14) * (364 // 14))
        plan = lidar_plan(cfg)
        self.assertEqual(plan["subrays"], 1)
        self.assertEqual(plan["aligned"], (0, len(plan["rows_cols"])))


class ConvertSubraysTest(unittest.TestCase):
    def test_n1_convert_matches_old_cells(self):
        plan = {
            "aligned": (0, 2),
            "wide": (2, 2),
            "rows_cols": [(0, 0), (0, 1)],
            "wide_angles_deg": [],
            "subrays": 1,
        }
        rays = {
            "block_name": ["pig", "air"],
            "block_distance": [4.0, 12.0],
            "entity_name": ["null", "null"],
            "entity_distance": [0.0, 0.0],
        }
        converted = convert_rays(rays, plan, load_config())
        self.assertEqual(len(converted["aligned"]), 2)
        self.assertEqual(converted["aligned"][0]["label"], "pig")
        self.assertEqual(converted["aligned"][0]["distance"], 4.0)
        self.assertIsNone(converted["aligned"][1]["label"])

    def test_n3_marks_object_on_one_subcell_and_min_distance(self):
        cfg = load_config()
        cfg["aligned_subrays"] = 3
        plan = {
            "aligned": (0, 9),
            "wide": (9, 9),
            "rows_cols": [(0, 0)],
            "wide_angles_deg": [],
            "subrays": 3,
        }
        names = ["air"] * 9
        dists = [12.0] * 9
        names[0] = "pig"
        dists[0] = 5.0
        names[8] = "pig"
        dists[8] = 3.0
        names[4] = "cow"
        dists[4] = 6.0
        rays = {
            "block_name": names,
            "block_distance": dists,
            "entity_name": ["null"] * 9,
            "entity_distance": [0.0] * 9,
        }
        converted = convert_rays(rays, plan, cfg)
        cells = {(r["row"], r["col"], r["label"]): r["distance"] for r in converted["aligned"] if r["label"]}
        self.assertEqual(cells[(0, 0, "pig")], 3.0)
        self.assertEqual(cells[(0, 0, "cow")], 6.0)
        primary = converted["aligned"][0]
        self.assertEqual(primary["label"], "pig")
        self.assertEqual(primary["distance"], 3.0)


class FakeSubraysTest(unittest.TestCase):
    def test_n1_matches_center_implementation(self):
        cfg = _tiny_cfg(aligned_subrays=1)
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.4, "z": 6.0, "radius": 0.35}]})
        obs = env.reset()
        key = (cfg.get("obs_keys") or {}).get("aligned_rays") or "lfnav_rays"
        new = list(obs[key])
        old = env._aligned_rays_centers()
        self.assertEqual(
            [(r["row"], r["col"], r["label"], r["distance"]) for r in new],
            [(r["row"], r["col"], r["label"], r["distance"]) for r in old],
        )

    def test_n3_hits_off_center_target(self):
        cfg = _tiny_cfg(aligned_subrays=1)
        grid = subray_grid_from_cfg({**cfg, "aligned_subrays": 3})
        corner = grid[0][0][0]
        t = 8.0
        layout = {"entities": [{"name": "pig", "x": corner[0] * t, "z": corner[2] * t, "radius": 0.08}]}
        miss = FakeMineDojoEnv(cfg, layout).reset()
        hit = FakeMineDojoEnv({**cfg, "aligned_subrays": 3}, layout).reset()
        self.assertIsNone(aligned_target_distance(miss, cfg, "pig"))
        self.assertIsNotNone(aligned_target_distance(hit, {**cfg, "aligned_subrays": 3}, "pig"))
        scene = scene_state_from_obs(hit, {**cfg, "aligned_subrays": 3})
        self.assertGreaterEqual(scene.object("pig").present, 1.0)

    def test_fake_min_distance(self):
        cfg = _tiny_cfg(aligned_subrays=3)
        env = FakeMineDojoEnv(cfg, {
            "entities": [
                {"name": "pig", "x": 0.0, "z": 9.0, "radius": 0.5},
                {"name": "pig", "x": 0.0, "z": 4.0, "radius": 0.5},
            ],
        })
        dist = aligned_target_distance(env.reset(), cfg, "pig")
        self.assertIsNotNone(dist)
        self.assertLess(dist, 5.0)
        self.assertGreater(dist, 3.5)


if __name__ == "__main__":
    unittest.main()
