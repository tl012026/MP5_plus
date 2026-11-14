"""GT target observation geometry for C. No MineDojo."""

from __future__ import annotations

import math
import unittest

import numpy as np

from lfnav.policy.c2nav import C2NavAgent, align_hw, patch_bearings, target_geometry
from lfnav.policy.train_c2nav import gt_target_obs, pool_to_grid

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
        pooled = align_hw(ignore, 8, 13, "max")
        self.assertEqual(tuple(pooled.shape), (8, 13))
        self.assertGreater(float(pooled[:, -1].min()), 0.5)
        vis, _, _ = self._geom(8, 13, [12], ignore=ignore)
        self.assertLess(vis, 0.5)


class StopTauConfigTest(unittest.TestCase):
    def test_stop_tau_zero_is_kept(self):
        agent = C2NavAgent({"dry_run": True, "c2nav": {"stop_tau": 0, "encoder": "fake"}})
        self.assertEqual(float(agent.tau), 0.0)

    def test_stop_tau_unset_defaults_to_half(self):
        agent = C2NavAgent({"dry_run": True, "c2nav": {"encoder": "fake"}})
        self.assertEqual(float(agent.tau), 0.5)


if __name__ == "__main__":
    unittest.main()
