"""Waypoint / mode label encode. No MineDojo."""

from __future__ import annotations

import unittest

from lfnav.policy.c2nav_labels import (
    FORWARD_BINS,
    MODE_NAMES,
    N_TURN,
    TURN_BINS,
    compose_waypoint,
    explore_turn_soft,
    flip_turn_index,
    forward_index,
    turn_center,
    turn_index,
)


class ModeSpaceTest(unittest.TestCase):
    def test_four_modes_order(self):
        self.assertEqual(MODE_NAMES, ("scan", "explore", "approach", "reacquire"))


class TurnLabelTest(unittest.TestCase):
    def test_boundary_mapping(self):
        expected = {
            -180.0: 180,
            180.0: 180,
            225.0: -135,
            270.0: -90,
            -105.0: -90,
            -75.0: -90,
            -60.0: -45,
            -30.0: -45,
            30.0: 45,
            0.0: 0,
        }
        for yaw, center in expected.items():
            idx = turn_index(yaw)
            self.assertEqual(TURN_BINS[idx], center, (yaw, TURN_BINS[idx]))
            self.assertEqual(turn_center(idx), float(center))

    def test_flip_180_is_self_and_twice_restores(self):
        idx = turn_index(180.0)
        self.assertEqual(TURN_BINS[idx], 180)
        self.assertEqual(flip_turn_index(idx), idx)
        for i in range(len(TURN_BINS)):
            self.assertEqual(flip_turn_index(flip_turn_index(i)), i, TURN_BINS[i])


class ExploreSoftTest(unittest.TestCase):
    def test_tied_min_sectors_uniform_over_turns(self):
        from lfnav.policy.c1 import _face_sector

        counts = [5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        soft = explore_turn_soft(counts, N_TURN)
        self.assertAlmostEqual(float(soft.sum()), 1.0, places=5)
        t1 = compose_waypoint(_face_sector(1))["turn_index"]
        t2 = compose_waypoint(_face_sector(2))["turn_index"]
        self.assertGreater(float(soft[t1]), 0.0)
        self.assertGreater(float(soft[t2]), 0.0)
        self.assertAlmostEqual(float(soft[t1]), float(soft[t2]), places=5)
        self.assertAlmostEqual(float(soft[t1] + soft[t2]), 1.0, places=5)


class ForwardLabelTest(unittest.TestCase):
    def test_explicit_map(self):
        expected = ((-1, 0), (0, 1), (1, 2), (2, 3), (9, 3))
        for blocks, cls in expected:
            self.assertEqual(forward_index(blocks), cls, (blocks, cls))

    def test_zero_is_not_truthy_bug(self):
        self.assertEqual(forward_index(0), 1)
        self.assertNotEqual(forward_index(0), forward_index(1))
        self.assertEqual(FORWARD_BINS[forward_index(0)], 0)


if __name__ == "__main__":
    unittest.main()
