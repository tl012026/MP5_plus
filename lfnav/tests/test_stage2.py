"""Controller, memory, C0/C1, and the eval shell on the fake environment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lfnav.config import load_config
from lfnav.control.controller import Controller
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.eval.run import run_batch, run_episode
from lfnav.memory import Memory
from lfnav.policy.c0 import RandomWalk
from lfnav.policy.c3 import DiffusionPolicy
from lfnav.types import PolicyInput, TaskSpec

ROOT = Path(__file__).resolve().parents[2]


def small_cfg():
    cfg = load_config()
    cfg["image_hw"] = [14, 14]
    cfg["perception_backend"] = "gt"
    cfg["policy_name"] = "c1"
    cfg["backend"] = "fake"
    cfg["judge"] = dict(cfg.get("judge") or {})
    cfg["judge"]["max_steps"] = 40
    cfg["match"] = dict(cfg.get("match") or {})
    cfg["match"]["stop_mode"] = "sr_b"
    return cfg


class MemoryTest(unittest.TestCase):
    def test_heading_aligned_crop(self):
        memory = Memory(size=15)
        memory.visits[(0, 4)] = 1
        pos = [0.5, 64.0, 0.5]
        ahead = memory.local_grid(pos, 0.0)
        behind = memory.local_grid(pos, 180.0)
        self.assertEqual(ahead[3][7], 1.0)
        self.assertEqual(behind[3][7], 0.0)
        self.assertEqual(behind[11][7], 1.0)
        memory.observe(pos, 0.0, "forward", {"status": "not_found"})
        self.assertEqual(memory.recent_actions, ["forward"])


class ControllerTest(unittest.TestCase):
    def test_jump_and_avoid_water(self):
        cfg = small_cfg()
        solid = FakeMineDojoEnv(cfg, {"solids": [(0, 1)]})
        result = Controller(solid, cfg).execute("forward")
        self.assertTrue(result.jumped)
        self.assertGreater(result.obs["location_stats"]["pos"][2], 0.5)

        water = FakeMineDojoEnv(cfg, {"water": [(0, 1)]})
        avoided = Controller(water, cfg).execute("forward")
        self.assertLess(abs(avoided.obs["location_stats"]["pos"][2]), 0.2)
        self.assertNotEqual(avoided.obs["location_stats"]["yaw"] % 360.0, 0.0)

    def test_waypoints(self):
        cfg = small_cfg()
        env = FakeMineDojoEnv(cfg, {})
        result = Controller(env, cfg).execute_waypoints([(15, 1)])
        self.assertGreater(result.env_steps, 1)
        self.assertGreater(result.obs["location_stats"]["pos"][2], 0.4)


class EpisodeTest(unittest.TestCase):
    def test_c1_reaches_pig(self):
        cfg = small_cfg()
        task = TaskSpec.from_dict(json.loads((ROOT / "lfnav/tasks/specs/1-4.json").read_text()))
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": 6.0}], "biome": "plains"})
        rec, events = run_episode(env, task, cfg, seed=0)
        self.assertTrue(rec["sr_b"], rec)
        self.assertTrue(rec["success"])
        self.assertTrue(events[-1]["match"]["stop"])

    def test_c0_runs(self):
        cfg = small_cfg()
        cfg["policy_name"] = "c0"
        task = TaskSpec.from_dict(json.loads((ROOT / "lfnav/tasks/specs/1-1.json").read_text()))
        env = FakeMineDojoEnv(cfg, {"blocks": {(0, 3): "log"}, "biome": "forest"})
        rec, _events = run_episode(env, task, cfg, policy=RandomWalk(cfg), seed=1)
        self.assertIn(rec["outcome"], ("success", "timeout", "fail_too_far", "false_stop"))

    def test_c3_is_unimplemented(self):
        with self.assertRaises(NotImplementedError):
            DiffusionPolicy().act(PolicyInput())

    def test_skip_done_and_atomic_json(self):
        cfg = small_cfg()
        cfg["judge"]["max_steps"] = 8
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            seeds = out / "seeds.txt"
            seeds.write_text("0\n")
            tasks = [str(ROOT / "lfnav/tasks/specs/1-4.json")]
            run_batch(cfg, tasks, [0], out, skip_done=False)
            dest = out / "1-4" / "0.json"
            self.assertTrue(dest.is_file())
            text = dest.read_text()
            self.assertNotIn(".tmp", text)
            dest.write_text(json.dumps({"task_name": "1-4", "seed": 0, "success": True, "marker": "keep"}))
            run_batch(cfg, tasks, [0], out, skip_done=True)
            kept = json.loads(dest.read_text())
            self.assertEqual(kept["marker"], "keep")
            self.assertTrue((out / "summary.csv").is_file())
            self.assertTrue((out / "1-4" / "0" / "events.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
