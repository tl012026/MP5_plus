"""GT target observation geometry for C. No MineDojo."""

from __future__ import annotations

import math
import unittest

import numpy as np

from lfnav.policy.c2nav import (
    C2NavAgent,
    dir_lcr_from_present,
    gt_target_obs,
    patch_bearings,
    patch_targets,
    pool_gt_to_grid,
    pool_to_grid,
    target_geometry,
    upsample_nearest,
    vis_labels,
)

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

CFG = {"image_hw": [224, 364], "fov_deg": 70}


class TargetObsGeomTest(unittest.TestCase):
    def _bearings(self, h=4, w=8):
        return patch_bearings(h, w, CFG)

    def test_target_on_right_has_positive_bearing(self):
        present = np.zeros((8, 4, 8), np.float32)
        present[3, :, 6:8] = 1.0
        ignore = np.zeros((4, 8), np.float32)
        obs = gt_target_obs(present, ignore, 3, self._bearings(), 5.0)
        self.assertTrue(obs.visible)
        self.assertGreater(float(obs.bearing_deg), 0.0)

    def test_target_on_left_has_negative_bearing(self):
        present = np.zeros((8, 4, 8), np.float32)
        present[3, :, 0:2] = 1.0
        ignore = np.zeros((4, 8), np.float32)
        obs = gt_target_obs(present, ignore, 3, self._bearings(), 5.0)
        self.assertTrue(obs.visible)
        self.assertLess(float(obs.bearing_deg), 0.0)

    def test_no_target_is_invisible(self):
        present = np.zeros((8, 4, 8), np.float32)
        ignore = np.zeros((4, 8), np.float32)
        obs = gt_target_obs(present, ignore, 3, self._bearings(), 5.0)
        self.assertFalse(obs.visible)

    def test_pool_16x26_to_8x13_keeps_right_side(self):
        raw = np.zeros((16, 26), np.float32)
        raw[:, 24:26] = 1.0
        pooled = pool_to_grid(raw, (8, 13), "max")
        self.assertEqual(tuple(pooled.shape), (8, 13))
        self.assertGreater(float(pooled[:, -1].max()), 0.5)
        present = np.zeros((8, 8, 13), np.float32)
        present[0] = pooled
        obs = gt_target_obs(present, np.zeros((8, 13), np.float32), 0, self._bearings(8, 13), 4.0)
        self.assertTrue(obs.visible)
        self.assertGreater(float(obs.bearing_deg), 0.0)


@unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
class TargetGeometryGridTest(unittest.TestCase):
    def _geom(self, h, w, cols, ignore=None, bearings=None):
        import torch

        present = np.zeros((1, 1, h, w), np.float32)
        present[0, 0, :, cols] = 1.0
        log_d = np.full((1, 1, h, w), math.log(8.0), np.float32)
        vis, sn, cs, dist = target_geometry(
            torch.from_numpy(present),
            torch.from_numpy(log_d),
            0,
            bearings,
            vis_tau=0.5,
            ignore=ignore,
            cfg=CFG,
        )
        yaw = math.degrees(math.atan2(float(sn), float(cs))) if float(vis) >= 0.5 else 0.0
        return float(vis), yaw, float(dist)

    def test_right_left_center_match_across_grids(self):
        cases = (
            ("right", [24, 25], [12], 1.0),
            ("left", [0, 1], [0], -1.0),
            ("center", [12, 13], [6], 0.0),
        )
        for name, fine_cols, coarse_cols, sign in cases:
            vis_f, yaw_f, _ = self._geom(16, 26, fine_cols)
            vis_c, yaw_c, _ = self._geom(8, 13, coarse_cols)
            self.assertGreaterEqual(vis_f, 0.5, name)
            self.assertGreaterEqual(vis_c, 0.5, name)
            if sign > 0:
                self.assertGreater(yaw_f, 0.0, name)
                self.assertGreater(yaw_c, 0.0, name)
            elif sign < 0:
                self.assertLess(yaw_f, 0.0, name)
                self.assertLess(yaw_c, 0.0, name)
            else:
                self.assertLess(abs(yaw_f), 1.0, name)
                self.assertLess(abs(yaw_c), 1.0, name)
            self.assertLessEqual(abs(yaw_f - yaw_c), 4.0, f"{name} fine={yaw_f} coarse={yaw_c}")

    def test_no_target_is_invisible(self):
        vis, yaw, _ = self._geom(8, 13, [])
        self.assertLess(vis, 0.5)
        self.assertEqual(yaw, 0.0)

    def test_mismatched_26_col_bearings_use_actual_grid(self):
        wrong = patch_bearings(16, 26, CFG)
        vis, yaw, _ = self._geom(8, 13, [12], bearings=wrong)
        self.assertGreaterEqual(vis, 0.5)
        self.assertGreater(yaw, 0.0)

    def test_ignore_16x26_pools_to_8x13(self):
        ignore = np.zeros((16, 26), np.float32)
        ignore[:, 24:26] = 1.0
        pooled = pool_to_grid(ignore, (8, 13), "min")
        self.assertEqual(tuple(pooled.shape), (8, 13))
        self.assertGreater(float(pooled[:, -1].min()), 0.5)
        vis, _, _ = self._geom(8, 13, [12], ignore=ignore)
        self.assertLess(vis, 0.5)


class GtPoolRulesTest(unittest.TestCase):
    def test_one_positive_subpatch_stays_positive(self):
        present = np.zeros((16, 26), np.float32)
        present[3, 5] = 1.0
        ignore = np.zeros((16, 26), np.float32)
        pooled, ign, _dist = pool_gt_to_grid(present, ignore, None, (8, 13))
        self.assertEqual(tuple(pooled.shape), (8, 13))
        self.assertGreater(float(pooled[1, 2]), 0.5)
        self.assertLess(float(ign[1, 2]), 0.5)
        scene = {"patches": {"pig": present.tolist()}, "ignore": ignore.tolist()}
        pres, _log_d, ig = patch_targets(scene, ["pig"], 2, (8, 13))
        vis, _vis_ig = vis_labels(pres, ig)
        self.assertEqual(float(vis[0]), 1.0)
        obs = gt_target_obs(pres, ig, 0, patch_bearings(8, 13, CFG), 4.0)
        self.assertTrue(obs.visible)

    def test_partial_ignore_keeps_positive_and_not_ignored(self):
        present = np.zeros((16, 26), np.float32)
        ignore = np.zeros((16, 26), np.float32)
        present[0, 0] = 1.0
        ignore[0, 1] = 1.0
        ignore[1, 0] = 1.0
        ignore[1, 1] = 1.0
        pooled, ign, _dist = pool_gt_to_grid(present, ignore, None, (8, 13))
        self.assertGreater(float(pooled[0, 0]), 0.5)
        self.assertLess(float(ign[0, 0]), 0.5)
        scene = {"patches": {"pig": present.tolist()}, "ignore": ignore.tolist()}
        pres, _log_d, ig = patch_targets(scene, ["pig"], 2, (8, 13))
        vis, vis_ig = vis_labels(pres, ig)
        self.assertEqual(float(vis[0]), 1.0)
        self.assertEqual(float(vis_ig[0]), 0.0)
        self.assertLess(float(ig[0, 0]), 0.5)
        obs = gt_target_obs(pres, ig, 0, patch_bearings(8, 13, CFG), 4.0)
        self.assertTrue(obs.visible)

    def test_distance_takes_min_of_block(self):
        present = np.zeros((16, 26), np.float32)
        present[0:2, 0:2] = 1.0
        dist = np.zeros((16, 26), np.float32)
        dist[0, 0] = 12.0
        dist[0, 1] = 8.0
        dist[1, 0] = 5.0
        dist[1, 1] = 9.0
        _p, _ig, pooled_d = pool_gt_to_grid(present, np.zeros((16, 26), np.float32), dist, (8, 13))
        self.assertAlmostEqual(float(pooled_d[0, 0]), 5.0, places=5)
        scene = {
            "patches": {"pig": present.tolist()},
            "ignore": np.zeros((16, 26), np.float32).tolist(),
            "patch_dist": {"pig": dist.tolist()},
        }
        _pres, log_d, _ig = patch_targets(scene, ["pig"], 2, (8, 13))
        self.assertAlmostEqual(float(log_d[0, 0, 0]), math.log(5.0), places=5)


class DirLcrTest(unittest.TestCase):
    def test_column_direction_on_right_third(self):
        present = np.zeros((16, 26), np.float32)
        present[:, 24] = 1.0
        self.assertEqual(dir_lcr_from_present(present), "right")
        present[:, :] = 0.0
        present[:, 0] = 1.0
        self.assertEqual(dir_lcr_from_present(present), "left")
        present[:, :] = 0.0
        present[:, 13] = 1.0
        self.assertEqual(dir_lcr_from_present(present), "center")

    def test_upsample_8x13_then_lcr(self):
        coarse = np.zeros((8, 13), np.float32)
        coarse[:, 12] = 1.0
        up = upsample_nearest(coarse, (16, 26))
        self.assertEqual(tuple(up.shape), (16, 26))
        self.assertEqual(dir_lcr_from_present(coarse), "right")
        self.assertEqual(dir_lcr_from_present(up), "right")


class StopTauConfigTest(unittest.TestCase):
    def test_stop_tau_zero_is_kept(self):
        agent = C2NavAgent({"dry_run": True, "c2nav": {"stop_tau": 0, "encoder": "fake"}})
        self.assertEqual(float(agent.tau), 0.0)

    def test_stop_tau_unset_defaults_to_half(self):
        agent = C2NavAgent({"dry_run": True, "c2nav": {"encoder": "fake"}})
        self.assertEqual(float(agent.tau), 0.5)


if __name__ == "__main__":
    unittest.main()
