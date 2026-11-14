"""Waypoint label encode/decode. No MineDojo."""

from __future__ import annotations

import unittest

from lfnav.policy.c2nav_labels import (
    FORWARD_BINS,
    N_TURN,
    TURN_WIDTH,
    clip_forward_blocks,
    compose_waypoint,
    forward_blocks_from_class,
    forward_index,
    invert_waypoint,
    turn_center,
    turn_index,
)


class ForwardLabelTest(unittest.TestCase):
    def test_explicit_map(self):
        expected = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
        for blocks, cls in expected.items():
            self.assertEqual(forward_index(blocks), cls)
            self.assertEqual(forward_blocks_from_class(cls), blocks)

    def test_zero_is_not_truthy_bug(self):
        self.assertEqual(forward_index(0), 1)
        self.assertNotEqual(forward_index(0), forward_index(1))

    def test_clip_outside(self):
        clipped, was = clip_forward_blocks(6)
        self.assertEqual(clipped, 4)
        self.assertTrue(was)
        self.assertEqual(forward_index(6), 5)
        clipped_lo, was_lo = clip_forward_blocks(-3)
        self.assertEqual(clipped_lo, -1)
        self.assertTrue(was_lo)
        self.assertEqual(forward_index(-3), 0)


class WaypointRoundtripTest(unittest.TestCase):
    def test_forward_roundtrip(self):
        for blocks in (-1, 0, 1, 4, 6):
            idx = forward_index(blocks)
            actions = invert_waypoint(0, idx)
            again = compose_waypoint(actions)
            want = max(FORWARD_BINS[0], min(FORWARD_BINS[-1], int(round(blocks))))
            self.assertEqual(again["forward"], want)
            self.assertEqual(again["forward_index"], forward_index(want))

    def test_turn_named_angles(self):
        for yaw in (0.0, 15.0, -15.0, 45.0, -180.0, 180.0):
            idx = turn_index(yaw)
            actions = invert_waypoint(idx, forward_index(0))
            again = compose_waypoint(actions)
            self.assertEqual(again["turn_index"], idx)

    def test_turn_bin_boundaries(self):
        for i in range(N_TURN):
            center = turn_center(i)
            lo = center - TURN_WIDTH / 2.0
            hi = center + TURN_WIDTH / 2.0 - 1e-6
            for yaw in (center, lo + 1e-3, hi):
                self.assertEqual(turn_index(yaw), i, (i, yaw, center))
            actions = invert_waypoint(i, 1)
            again = compose_waypoint(actions)
            self.assertEqual(again["turn_index"], i)


if __name__ == "__main__":
    unittest.main()
