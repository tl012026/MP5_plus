"""Post-reset /summon sampling and settle no-ops. No MineDojo import."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from lfnav.config import load_config
from lfnav.env.minedojo_env import (
    lidar_plan,
    sample_mob_offset,
    spawn_task_mobs,
    task_mentions_grass,
    vegetation_fill_commands,
)
from lfnav.gt.initial import apply_spawn_offset_distance, initial_scene_metrics
from lfnav.types import TaskSpec

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "lfnav" / "tasks" / "specs"


def _stub(cfg, cmds, steps):
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
            return {
                "location_stats": {"yaw": 0.0, "pos": [0.0, 64.0, 0.0]},
                "lfnav_rays": [],
                "lfnav_wide_rays": [],
            }, 0.0, False, {}

    wrapper = SimpleNamespace(_env=Inner(), _obs=None, cfg=cfg, spawn_meta={})
    wrapper._adopt = lambda obs: obs
    wrapper._obs = Inner().step("boot")[0]
    steps.clear()
    return wrapper


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
            self.assertLess(off["dz"], 0)
            self.assertLessEqual(abs(off["dx"]), abs(off["dz"]) + 1)
            self.assertGreaterEqual(off["bearing_deg"], 135.0)
            self.assertLessEqual(off["bearing_deg"], 225.0)


class SpawnOnceTest(unittest.TestCase):
    def _cfg(self, **md):
        cfg = load_config()
        cfg["minedojo"] = dict(cfg["minedojo"])
        cfg["minedojo"]["settle_ticks"] = 1
        cfg["minedojo"]["settle_after_spawn"] = 2
        cfg["minedojo"]["clear_entities"] = True
        cfg["minedojo"]["spawn_verify_retry"] = False
        cfg["minedojo"]["spawn_drop_height"] = 8
        cfg["minedojo"].update(md)
        return cfg

    def test_kill_before_each_animal_summoned_once(self):
        cfg = self._cfg()
        cfg["difficulty"] = "easy"
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        cmds, steps = [], []
        wrapper = _stub(cfg, cmds, steps)
        spawn_task_mobs(wrapper, task, cfg, seed=3)
        self.assertTrue(all(a == "NOOP" for a in steps))
        self.assertEqual(cmds[0], "/kill @e[type=!player]")
        summons = [c for c in cmds if c.startswith("/summon ")]
        self.assertEqual(len(summons), 1)
        self.assertTrue(summons[0].startswith("/summon pig"))
        self.assertIn("~8", summons[0])
        self.assertLess(cmds.index("/kill @e[type=!player]"), next(i for i, c in enumerate(cmds) if c.startswith("/summon")))
        self.assertEqual(wrapper.spawn_meta["n_targets_spawned"], 1)
        self.assertFalse(wrapper.spawn_meta["spawn_failed"])
        self.assertTrue(wrapper.spawn_meta["natural_mobs_cleared"])
        fills = [c for c in cmds if c.startswith("/fill ")]
        self.assertTrue(fills)
        self.assertTrue(wrapper.spawn_meta["vegetation_cleared"])
        self.assertTrue(any("resistance 10 4" in c for c in cmds))
        self.assertTrue(any("slowness 1000000 10 true" in c for c in cmds))
        self.assertLess(cmds.index(fills[0]), next(i for i, c in enumerate(cmds) if c.startswith("/summon")))

    def test_grass_tasks_skip_vegetation_clear(self):
        cfg = self._cfg()
        task = TaskSpec.from_dict(json.loads((SPECS / "4-4.json").read_text()))
        self.assertTrue(task_mentions_grass(task))
        cmds, steps = [], []
        wrapper = _stub(cfg, cmds, steps)
        spawn_task_mobs(wrapper, task, cfg, seed=3)
        self.assertFalse(any(c.startswith("/fill ") for c in cmds))
        self.assertFalse(wrapper.spawn_meta["vegetation_cleared"])

    def test_fill_chunks_stay_under_limit(self):
        cmds = vegetation_fill_commands()
        self.assertGreaterEqual(len(cmds), 4)
        for cmd in cmds:
            parts = cmd.split()
            x0, y0, z0, x1, y1, z1 = (int(p.replace("~", "") or "0") for p in parts[1:7])
            vol = (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1) * (abs(z1 - z0) + 1)
            self.assertLessEqual(vol, 32768)
            self.assertIn("replace", cmd)

    def test_default_drop_height_is_two(self):
        cfg = self._cfg()
        del cfg["minedojo"]["spawn_drop_height"]
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        cmds, steps = [], []
        wrapper = _stub(cfg, cmds, steps)
        spawn_task_mobs(wrapper, task, cfg, seed=3)
        summons = [c for c in cmds if c.startswith("/summon pig")]
        self.assertTrue(any("~2" in c for c in summons))

    def test_hard_offset_distance_is_the_initial_distance(self):
        cfg = self._cfg()
        cfg["difficulty"] = "hard"
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        cmds, steps = [], []
        wrapper = _stub(cfg, cmds, steps)
        spawn_task_mobs(wrapper, task, cfg, seed=9)
        off = wrapper.spawn_meta["target_spawn_offset"]
        dist = math.hypot(off["dx"], off["dz"])
        self.assertGreaterEqual(dist, 16.0 - 1.5)
        self.assertLessEqual(dist, 32.0 + 1.5)
        self.assertLess(off["dz"], 0)
        self.assertLessEqual(abs(off["dx"]), abs(off["dz"]) + 1)
        metrics = apply_spawn_offset_distance(
            initial_scene_metrics(wrapper._obs, wrapper, cfg, task),
            wrapper.spawn_meta,
        )
        self.assertEqual(metrics["initial_target_distance"], dist)
        self.assertIn("initial_target_distance_rays", metrics)

    def test_verify_retry_still_works_when_enabled(self):
        cfg = self._cfg(spawn_verify_retry=True, spawn_retries=3, verify_ticks=1, settle_after_spawn=0)
        cfg["difficulty"] = "easy"
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        cmds, steps = [], []
        wrapper = _stub(cfg, cmds, steps)
        spawn_task_mobs(wrapper, task, cfg, seed=1)
        summons = [c for c in cmds if c.startswith("/summon pig")]
        self.assertEqual(len(summons), 3)
        self.assertEqual(wrapper.spawn_meta["spawn_attempts"], 3)
        self.assertFalse(wrapper.spawn_meta["spawn_failed"])


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
