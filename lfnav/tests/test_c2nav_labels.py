"""Waypoint / mode label encode. No MineDojo."""

from __future__ import annotations

import unittest

from lfnav.policy.c2nav_labels import (
    FORWARD_BINS,
    MODE_NAMES,
    TURN_BINS,
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
