"""Post-reset /summon sampling and settle no-ops. No MineDojo import."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from lfnav.config import load_config
from lfnav.env.minedojo_env import _noop_tick, lidar_plan, sample_mob_offset, spawn_task_mobs
from lfnav.gt.initial import privileged_ray_distance
from lfnav.types import TaskSpec

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "lfnav" / "tasks" / "specs"


class SampleRingTest(unittest.TestCase):
    def test_hard_is_behind_and_reproducible(self):
        min_d, max_d = 16.0, 32.0
        yaw = 0.0
        a = sample_mob_offset(min_d, max_d, yaw, seed=11, index=0, behind=True)
        b = sample_mob_offset(min_d, max_d, yaw, seed=11, index=0, behind=True)
        self.assertEqual((a["dx"], a["dy"], a["dz"]), (b["dx"], b["dy"], b["dz"]))
        slack = 1.5
        for seed in range(40):
            off = sample_mob_offset(min_d, max_d, yaw, seed, 0, behind=True)
            dist = math.hypot(off["dx"], off["dz"])
            self.assertGreaterEqual(dist, min_d - slack)
            self.assertLessEqual(dist, max_d + slack)
            # yaw=0 faces +z; behind is -z, ±45° → |x| <= |z| and z < 0
            self.assertLess(off["dz"], 0)
            self.assertLessEqual(abs(off["dx"]), abs(off["dz"]) + 1)
            self.assertGreaterEqual(off["bearing_deg"], 135.0)
            self.assertLessEqual(off["bearing_deg"], 225.0)

    def test_easy_covers_the_ring(self):
        signs = set()
        for seed in range(80):
            off = sample_mob_offset(4, 8, 0.0, seed, 0, behind=False)
            signs.add((off["dx"] > 0, off["dz"] > 0))
            dist = math.hypot(off["dx"], off["dz"])
            self.assertGreaterEqual(dist, 2.0)
            self.assertLessEqual(dist, 10.0)
        self.assertGreaterEqual(len(signs), 3)


class SettleNoopTest(unittest.TestCase):
    def test_settle_uses_action_space_no_op_not_commands(self):
        cfg = load_config()
        cfg["minedojo"] = dict(cfg["minedojo"])
        cfg["minedojo"]["settle_ticks"] = 5
        cfg["minedojo"]["verify_ticks"] = 2
        cfg["minedojo"]["spawn_retries"] = 1
        cfg["difficulty"] = "easy"
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        cmds = []
        steps = []

        class Space:
            def no_op(self):
                return "NOOP"

        class Sim:
            def execute_cmd(self, cmd):
                cmds.append(cmd)

        class Inner:
            def __init__(self):
                self.action_space = Space()
                self.env = Sim()

            def step(self, action):
                steps.append(action)
                obs = {
                    "location_stats": {"yaw": 0.0, "pos": [0.0, 64.0, 0.0]},
                    "lfnav_rays": [],
                    "lfnav_wide_rays": [{"label": "pig", "distance": 6.0, "yaw_offset": 180.0, "pitch": -3.0}],
                }
                return obs, 0.0, False, {}

        wrapper = SimpleNamespace(_env=Inner(), _obs=None, cfg=cfg, spawn_meta={})

        def adopt(obs):
            wrapper._obs = obs
            return obs

        wrapper._adopt = adopt
        wrapper._obs = {
            "location_stats": {"yaw": 0.0, "pos": [0.0, 64.0, 0.0]},
            "lfnav_rays": [],
            "lfnav_wide_rays": [],
        }
        obs = spawn_task_mobs(wrapper, task, cfg, seed=3)
        self.assertTrue(steps)
        self.assertTrue(all(a == "NOOP" for a in steps))
        self.assertGreaterEqual(steps.count("NOOP"), 5)
        self.assertTrue(any(c.startswith("/summon pig") for c in cmds))
        self.assertFalse(any("wait" in c.lower() for c in cmds))
        nop = _noop_tick(wrapper)
        self.assertIs(nop, wrapper._obs)
        self.assertIsNotNone(privileged_ray_distance(obs, cfg, "pig"))
        self.assertFalse(wrapper.spawn_meta["spawn_failed"])

    def test_missing_target_marks_spawn_failed(self):
        cfg = load_config()
        cfg["minedojo"] = dict(cfg["minedojo"])
        cfg["minedojo"]["settle_ticks"] = 1
        cfg["minedojo"]["verify_ticks"] = 1
        cfg["minedojo"]["spawn_retries"] = 3
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))

        class Space:
            def no_op(self):
                return "NOOP"

        class Inner:
            def __init__(self):
                self.action_space = Space()
                self.env = SimpleNamespace(execute_cmd=lambda *_a, **_k: None)

            def step(self, action):
                obs = {
                    "location_stats": {"yaw": 0.0, "pos": [0.0, 64.0, 0.0]},
                    "lfnav_rays": [],
                    "lfnav_wide_rays": [],
                }
                return obs, 0.0, False, {}

        wrapper = SimpleNamespace(_env=Inner(), cfg=cfg, spawn_meta={})
        wrapper._adopt = lambda obs: obs
        wrapper._obs = Inner().step("boot")[0]
        spawn_task_mobs(wrapper, task, cfg, seed=1)
        self.assertTrue(wrapper.spawn_meta["spawn_failed"])
        self.assertEqual(wrapper.spawn_meta["spawn_attempts"], 3)


class WidePitchTest(unittest.TestCase):
    def test_three_pitch_rings(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        cfg["wide_yaw_step"] = 180
        cfg["wide_pitches_deg"] = [0, -3, -8]
        plan = lidar_plan(cfg)
        n_yaw = 2
        self.assertEqual(len(plan["wide_angles_deg"]), n_yaw * 3)
        pitches = sorted({p for p, _y in plan["wide_angles_deg"]})
        self.assertEqual(pitches, [-8.0, -3.0, 0.0])


if __name__ == "__main__":
    unittest.main()
