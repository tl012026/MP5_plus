"""C0/C1 baseline: nearest stop, 8-sector explore, stuck escape, water, recover."""

from __future__ import annotations

import unittest

from lfnav.config import load_config
from lfnav.control.controller import Controller
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.eval.run import run_episode
from lfnav.memory import Memory
from lfnav.policy.c1 import RuleBased, sector8_counts
from lfnav.types import PolicyInput, PolicyOutput, TaskSpec


def _cfg(**extra):
    cfg = load_config()
    cfg["image_hw"] = [14, 14]
    cfg["perception_backend"] = "gt"
    cfg["policy_name"] = "c1"
    cfg["backend"] = "fake"
    cfg["judge"] = dict(cfg.get("judge") or {}, max_steps=80, radius=3.0)
    cfg["match"] = dict(cfg.get("match") or {}, stop_mode="sr_b", stop_rule="nearest")
    cfg["policy"] = dict(cfg.get("policy") or {})
    cfg.update(extra)
    return cfg


class MatcherNearestStopTest(unittest.TestCase):
    def test_episode_stops_on_nearest_ray(self):
        cfg = _cfg()
        task = TaskSpec("1-4", "t", "pig", "pig")
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": 2.0}], "biome": "plains"})
        rec, events = run_episode(env, task, cfg, seed=0)
        self.assertTrue(events[-1]["match"]["stop"])
        self.assertTrue(rec["sr_b"])


class OpeningScanTest(unittest.TestCase):
    def test_scan_aborts_when_target_appears(self):
        cfg = _cfg()
        policy = RuleBased(cfg)
        policy.reset()
        miss = PolicyInput(match={"status": "not_found", "satisfied": {}}, memory={"local_grid": [[0.0] * 15] * 15, "env_steps": 0})
        self.assertEqual(policy.act(miss).action_chunk, ["turn_right_15"])
        found = PolicyInput(
            match={"status": "found", "target_direction": "center", "target_distance": "mid", "satisfied": {"target": True}},
            memory={"env_steps": 1, "local_grid": [[0.0] * 15] * 15},
        )
        chunk = policy.act(found).action_chunk
        self.assertIn("forward", chunk)
        self.assertNotEqual(chunk, ["turn_right_15"])
        self.assertLessEqual(chunk.count("forward"), 2)


class EightSectorTest(unittest.TestCase):
    def test_behind_sector_can_win(self):
        cfg = _cfg()
        cfg["policy"]["c1_opening_scan"] = False
        policy = RuleBased(cfg)
        policy.reset()
        grid = [[4.0] * 15 for _ in range(8)] + [[0.0] * 15 for _ in range(7)]
        counts = sector8_counts(grid)
        self.assertEqual(min(range(8), key=lambda i: counts[i]), min(i for i, n in enumerate(counts) if n == min(counts)))
        self.assertLess(counts[4], counts[0])
        out = policy.act(PolicyInput(match={"status": "not_found"}, memory={"local_grid": grid, "env_steps": 0}))
        turns = out.action_chunk.count("turn_right_45") + out.action_chunk.count("turn_left_45")
        self.assertGreaterEqual(turns, 3)
        self.assertIn("forward", out.action_chunk)


class StuckDisplacementTest(unittest.TestCase):
    def test_sixty_still_steps_trigger_escape(self):
        cfg = _cfg()
        env = FakeMineDojoEnv(cfg, {"tall": [(0, 1), (1, 0), (0, -1), (-1, 0)]})
        ctrl = Controller(env, cfg)
        for _ in range(59):
            ctrl.execute("jump_forward")
            self.assertEqual(ctrl.n_stuck, 0)
        result = ctrl.execute("jump_forward")
        self.assertGreaterEqual(ctrl.n_stuck, 1)
        self.assertGreater(ctrl.stuck_steps_total, 0)
        self.assertTrue(result.stuck)
        self.assertIsNotNone(result.escape_k)

    def test_pure_turns_are_not_stuck(self):
        cfg = _cfg()
        env = FakeMineDojoEnv(cfg, {})
        ctrl = Controller(env, cfg)
        for _ in range(70):
            ctrl.execute("turn_right_15")
        self.assertEqual(ctrl.n_stuck, 0)


class VisitSkipTest(unittest.TestCase):
    def test_stuck_does_not_increment_cell(self):
        memory = Memory()
        pos = [0.5, 64.0, 0.5]
        memory.observe(pos, 0.0, "forward", increment_visits=False)
        self.assertEqual(memory.visits, {})
        memory.observe(pos, 0.0, "forward", increment_visits=True)
        self.assertEqual(memory.visits[(0, 0)], 1)


class RecoverOvershootTest(unittest.TestCase):
    def test_lost_mid_target_backs_up_then_scans(self):
        cfg = _cfg()
        cfg["policy"]["c1_opening_scan"] = False
        policy = RuleBased(cfg)
        policy.reset()
        seen = PolicyInput(
            match={"status": "found", "target_direction": "center", "target_distance": "mid", "satisfied": {"target": True}},
            memory={"env_steps": 5, "local_grid": [[0.0] * 15] * 15},
        )
        policy.act(seen)
        lost = PolicyInput(
            match={"status": "not_found", "satisfied": {}},
            memory={"env_steps": 10, "local_grid": [[0.0] * 15] * 15},
        )
        self.assertEqual(policy.act(lost).action_chunk, ["backward"])
        self.assertEqual(policy.act(lost).action_chunk, ["turn_right_15"])
        found_again = PolicyInput(
            match={"status": "found", "target_direction": "center", "target_distance": "near", "satisfied": {"target": True}},
            memory={"env_steps": 20, "local_grid": [[0.0] * 15] * 15},
        )
        chunk = policy.act(found_again).action_chunk
        self.assertTrue(chunk)
        self.assertNotEqual(chunk, ["turn_right_15"])


class MaxStepsCutTest(unittest.TestCase):
    def test_long_chunk_stops_at_max_steps(self):
        cfg = _cfg()
        cfg["judge"]["max_steps"] = 5
        cfg["frames"] = dict(cfg.get("frames") or {}, enabled=False)

        class LongChunk:
            def reset(self):
                return None

            def act(self, _pin):
                return PolicyOutput(["forward"] * 20)

        task = TaskSpec("1-1", "t", "tree", "tree")
        env = FakeMineDojoEnv(cfg, {"blocks": {(0, 20): "log"}, "biome": "forest"})
        rec, _events = run_episode(env, task, cfg, policy=LongChunk(), seed=0)
        self.assertEqual(rec["steps"], 5)
        self.assertLessEqual(rec["self_reported_steps"], 5)

    def test_escape_respects_budget(self):
        cfg = _cfg()
        cfg["actions"] = dict(cfg.get("actions") or {})
        cfg["actions"]["stuck_window"] = 4
        env = FakeMineDojoEnv(cfg, {"tall": [(0, 1), (1, 0), (0, -1), (-1, 0)]})
        ctrl = Controller(env, cfg)
        for _ in range(3):
            ctrl.execute("jump_forward")
            self.assertEqual(ctrl.n_stuck, 0)
        result = ctrl.execute("jump_forward", budget=3)
        self.assertLessEqual(result.env_steps, 3)
        self.assertTrue(result.stuck)


class WaterDepthTest(unittest.TestCase):
    def test_shallow_yes_deep_no(self):
        cfg = _cfg()
        shallow = FakeMineDojoEnv(cfg, {"water": [(0, 1)]})
        moved = Controller(shallow, cfg).execute("forward")
        self.assertGreater(moved.obs["location_stats"]["pos"][2], 0.5)
        deep = FakeMineDojoEnv(cfg, {"deep_water": [(0, 1)]})
        blocked = Controller(deep, cfg).execute("forward")
        self.assertLess(abs(blocked.obs["location_stats"]["pos"][2]), 0.2)


if __name__ == "__main__":
    unittest.main()
