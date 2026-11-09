"""Schedule and angle bookkeeping for capture_rotations. No MineDojo."""

from __future__ import annotations

import unittest

import numpy as np

from lfnav.gt.capture_rotations import (
    as_hwc,
    delta_minus_15k,
    rotation_schedule,
    skip_seed,
    wrap_delta_deg,
    yaw_sweep_names,
)


class CaptureScheduleTest(unittest.TestCase):
    def test_yaw_sweep_returns_to_start_then_goes_left(self):
        names = yaw_sweep_names()
        self.assertEqual(names[:3], ["turn_right_15"] * 3)
        self.assertEqual(names[3:6], ["turn_left_15"] * 3)
        self.assertEqual(names[6:], ["turn_left_15"] * 3)

    def test_schedule_is_camera_only(self):
        banned = {"forward", "jump_forward", "sprint"}
        names = rotation_schedule()
        self.assertTrue(set(names).isdisjoint(banned))
        self.assertIn("look_down_15", names)
        self.assertIn("look_up_15", names)
        self.assertGreater(len(names), 20)

    def test_delta_minus_15k(self):
        self.assertAlmostEqual(delta_minus_15k(15.0), 0.0)
        self.assertAlmostEqual(delta_minus_15k(14.2), -0.8, places=5)
        self.assertAlmostEqual(delta_minus_15k(-15.4), -0.4, places=5)
        self.assertAlmostEqual(wrap_delta_deg(350, 10), -20.0)

    def test_skip_underwater_or_no_sky(self):
        self.assertEqual(skip_seed({"pos": [0, 50, 0], "sea_level": 63, "can_see_sky": True}), "below sea_level")
        self.assertEqual(skip_seed({"pos": [0, 70, 0], "sea_level": 63, "can_see_sky": False}), "can_see_sky is false")
        self.assertIsNone(skip_seed({"pos": [0, 70, 0], "sea_level": 63, "can_see_sky": True}))

    def test_chw_rgb(self):
        chw = np.zeros((3, 8, 10), np.uint8)
        chw[0] = 7
        hwc = as_hwc(chw)
        self.assertEqual(hwc.shape, (8, 10, 3))
        self.assertEqual(int(hwc[0, 0, 0]), 7)


if __name__ == "__main__":
    unittest.main()
