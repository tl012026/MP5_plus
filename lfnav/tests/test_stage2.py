"""Controller, memory, C0/C1, and the eval shell on the fake environment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lfnav.config import load_config
from lfnav.control.controller import Controller
from unittest.mock import patch

from lfnav.env.fake import FakeMineDojoEnv
from lfnav.eval.run import run_batch, run_episode
from lfnav.memory import Memory
from lfnav.obsutil import yaw_of
from lfnav.policy.c0 import RandomWalk
from lfnav.policy.c3 import DiffusionPolicy
from lfnav.types import ObjectBelief, PolicyInput, SceneState, TaskSpec

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
        self.assertNotEqual(yaw_of(avoided.obs, cfg) % 360.0, 0.0)

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
        json.dumps(rec)
        json.dumps(events)

    def test_false_stop_when_perception_invents_the_target(self):
        cfg = small_cfg()
        task = TaskSpec.from_dict(json.loads((ROOT / "lfnav/tasks/specs/1-4.json").read_text()))

        class AlwaysFound:
            def reset(self):
                return None

            def perceive(self, _obs):
                return SceneState(objects={
                    "pig": ObjectBelief(1.0, {"left": 0.0, "center": 1.0, "right": 0.0}, {"near": 1.0, "mid": 0.0, "far": 0.0}),
                })

            def features(self, _obs):
                return None

        env = FakeMineDojoEnv(cfg, {})
        rec, events = run_episode(env, task, cfg, perception=AlwaysFound(), seed=0)
        self.assertTrue(rec["false_stop"])
        self.assertFalse(rec["success"])
        self.assertEqual(rec["outcome"], "false_stop")
        self.assertGreater(events[-1]["perception"]["objects"]["pig"]["present"], 0.5)
        self.assertEqual(events[-1]["gt"]["objects"]["pig"]["present"], 0.0)

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

    def test_close_matches_episode_count_including_errors(self):
        cfg = small_cfg()
        cfg["judge"]["max_steps"] = 4
        calls = []
        original = FakeMineDojoEnv.close

        def track(self):
            calls.append(self)
            return original(self)

        def boom(env, task, cfg, perception=None, policy=None, seed=0):
            if seed == 2:
                raise RuntimeError("boom")
            return run_episode(env, task, cfg, perception=perception, policy=policy, seed=seed)

        tasks = [str(ROOT / "lfnav/tasks/specs/1-4.json"), str(ROOT / "lfnav/tasks/specs/1-1.json")]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(FakeMineDojoEnv, "close", track):
                with patch("lfnav.eval.run.run_episode", boom):
                    with self.assertRaises(RuntimeError):
                        run_batch(cfg, tasks, [0, 2], Path(tmp), skip_done=False)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len({id(env) for env in calls}), 2)

    def test_reuse_closes_once_per_task(self):
        cfg = small_cfg()
        cfg["reuse_env"] = True
        cfg["judge"]["max_steps"] = 4
        calls = []
        original = FakeMineDojoEnv.close

        def track(self):
            calls.append(self)
            return original(self)

        tasks = [str(ROOT / "lfnav/tasks/specs/1-4.json"), str(ROOT / "lfnav/tasks/specs/1-1.json")]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(FakeMineDojoEnv, "close", track):
                run_batch(cfg, tasks, [0, 1], Path(tmp), skip_done=False)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
