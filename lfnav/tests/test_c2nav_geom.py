"""GT target observation geometry for C. No MineDojo."""

from __future__ import annotations

import unittest

import numpy as np

from lfnav.policy.c2nav import patch_bearings
from lfnav.policy.train_c2nav import gt_target_obs, pool_to_grid


class TargetObsGeomTest(unittest.TestCase):
    def _bearings(self, h=4, w=8):
        return patch_bearings(h, w, {"image_hw": [224, 364], "fov_deg": 70})

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


if __name__ == "__main__":
    unittest.main()
