"""Gradle copies, unseen-target fields, difficulty reroll, ring spawn, SPL, array map."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lfnav.config import load_config
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.env.gradle_home import copy_dest, copy_gradle_home
from lfnav.env.minedojo_env import mob_ring_box
from lfnav.eval.baseline_index import all_cells, n_cells
from lfnav.eval.run import EnvReuse, open_episode, run_episode
from lfnav.gt.judge import judge_episode, spl_approx
from lfnav.matcher import Matcher
from lfnav.policy.c2 import featurize
from lfnav.types import ObjectBelief, PolicyInput, SceneState, TaskSpec

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "lfnav" / "tasks" / "specs"


class GradleCopyTest(unittest.TestCase):
    def test_copy_excludes_daemon_and_locks_and_keeps_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            master = tmp / "master"
            (master / "daemon" / "x").mkdir(parents=True)
            (master / "daemon" / "x" / "daemon.log").write_text("no")
            (master / "caches").mkdir()
            (master / "caches" / "ok.bin").write_text("ok")
            (master / "caches" / "busy.lock").write_text("lock")
            (master / "init.d").mkdir()
            init = "allprojects { repositories { maven { url '/scratch/USERNAME/env/m2' } } }\n"
            (master / "init.d" / "mixingradle-local.gradle").write_text(init)
            dest = tmp / "copy"
            info = copy_gradle_home(master, dest)
            self.assertGreater(info["elapsed_s"], -1.0)
            self.assertFalse((dest / "daemon").exists())
            self.assertFalse((dest / "caches" / "busy.lock").exists())
            self.assertEqual((dest / "caches" / "ok.bin").read_text(), "ok")
            self.assertEqual((dest / "init.d" / "mixingradle-local.gradle").read_text(), init)
            self.assertEqual(copy_dest("99", tmpdir=tmp), tmp / "gradle_99")
            self.assertEqual(copy_dest("99", worker_id=2, tmpdir=tmp), tmp / "gradle_99_w2")

    def test_shell_sets_env_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            master = tmp / "master"
            (master / "init.d").mkdir(parents=True)
            (master / "daemon" / "x").mkdir(parents=True)
            (master / "daemon" / "x" / "n.log").write_text("d")
            (master / "init.d" / "mixingradle-local.gradle").write_text("url '/scratch/USERNAME/env/m2'\n")
            (master / "a.lock").write_text("l")
            (master / "keep.txt").write_text("k")
            node = tmp / "node"
            node.mkdir()
            script = ROOT / "lfnav" / "env" / "gradle_home.sh"
            marker = tmp / "home.txt"
            proc = subprocess.run(
                [
                    "bash", "-c",
                    "set -euo pipefail; "
                    f"export MP5='{ROOT}'; export SCR='{tmp}'; "
                    "export SLURM_JOB_ID=4242; "
                    f"export SLURM_TMPDIR='{node}'; "
                    f"export GRADLE_MASTER='{master}'; "
                    f"source '{script}'; "
                    "printf '%s' \"$GRADLE_USER_HOME\" > "
                    f"'{marker}'; "
                    "test ! -d \"$GRADLE_USER_HOME/daemon\"; "
                    "test ! -f \"$GRADLE_USER_HOME/a.lock\"; "
                    "grep -F '/scratch/USERNAME/env/m2' \"$GRADLE_USER_HOME/init.d/mixingradle-local.gradle\"; "
                    "test -d \"$GRADLE_USER_HOME\"",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            home = Path(marker.read_text())
            self.assertEqual(home, node / "gradle_4242")
            self.assertFalse(home.exists(), proc.stdout + proc.stderr)
            self.assertIn("gradle copy", proc.stdout)


class MatcherNoneTest(unittest.TestCase):
    def test_visible_and_hidden(self):
        task = TaskSpec.from_dict(json.loads((SPECS / "4-4.json").read_text()))
        visible = SceneState(objects={
            "pig": ObjectBelief(0.9, {"left": 0.0, "center": 1.0, "right": 0.0}, {"near": 1.0, "mid": 0.0, "far": 0.0}),
            "grass": ObjectBelief(1.0, {"left": 0.0, "center": 1.0, "right": 0.0}, {"near": 1.0, "mid": 0.0, "far": 0.0}),
        }, biome={"plains": 1.0}, time={"day": 1.0}, weather={"clear": 1.0}, brightness={"sufficient": 1.0})
        hidden = SceneState(objects={
            "pig": ObjectBelief(0.0, {"left": 1.0, "center": 0.0, "right": 0.0}, {"near": 1.0, "mid": 0.0, "far": 0.0}),
            "grass": ObjectBelief(1.0, {"left": 0.0, "center": 1.0, "right": 0.0}, {"near": 1.0, "mid": 0.0, "far": 0.0}),
        }, biome={"plains": 1.0}, time={"day": 1.0}, weather={"clear": 1.0}, brightness={"sufficient": 1.0})
        found = Matcher().match(task, visible)
        missed = Matcher().match(task, hidden)
        self.assertEqual(found.target_direction, "center")
        self.assertEqual(found.target_distance, "near")
        self.assertIsNone(missed.target_direction)
        self.assertIsNone(missed.target_distance)
        a = featurize(PolicyInput(match=found.to_dict()))
        b = featurize(PolicyInput(match=missed.to_dict()))
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(int(a.shape[0]), 16)
        self.assertEqual(float(b[3]), 0.0)
        self.assertEqual(float(a[3]), 1.0)
        self.assertEqual(list(b[:3]), [0.0, 0.0, 0.0])


class BlockRerollTest(unittest.TestCase):
    def test_third_world_is_used(self):
        cfg = load_config()
        cfg["backend"] = "fake"
        cfg["difficulty"] = "medium"
        task = TaskSpec.from_dict(json.loads((SPECS / "1-1.json").read_text()))
        calls = []

        def make(cfg, task, seed):
            calls.append(seed)
            dist = 1.0 if len(calls) < 3 else 12.0
            env = FakeMineDojoEnv(cfg, {"blocks": {(0, int(dist)): "log"}})
            env.distance_to = lambda name, d=dist: d
            return env

        with patch("lfnav.eval.run.make_env", make):
            session = EnvReuse(cfg)
            _env, _obs, meta = open_episode(session, task, cfg, 7)
            session.close()
        self.assertEqual(calls, [7, 7 * 1000 + 1, 7 * 1000 + 2])
        self.assertEqual(meta["world_seed"], 7 * 1000 + 2)
        self.assertEqual(meta["reroll_count"], 2)
        self.assertFalse(meta["difficulty_unmet"])

    def test_exhausted_rerolls_are_marked(self):
        cfg = load_config()
        cfg["backend"] = "fake"
        cfg["difficulty"] = "medium"
        cfg["difficulty_table"] = dict(cfg["difficulty_table"])
        cfg["difficulty_table"]["medium"] = dict(cfg["difficulty_table"]["medium"], max_reroll=2)
        task = TaskSpec.from_dict(json.loads((SPECS / "1-1.json").read_text()))
        calls = []

        def make(cfg, task, seed):
            calls.append(seed)
            env = FakeMineDojoEnv(cfg, {"blocks": {(0, 1): "log"}})
            env.distance_to = lambda name: 1.0
            return env

        with patch("lfnav.eval.run.make_env", make):
            session = EnvReuse(cfg)
            _env, _obs, meta = open_episode(session, task, cfg, 7)
            session.close()
        self.assertEqual(len(calls), 3)
        self.assertEqual(meta["world_seed"], 7 * 1000 + 2)
        self.assertEqual(meta["reroll_count"], 2)
        self.assertTrue(meta["difficulty_unmet"])
        self.assertFalse(meta.get("spawn_underground"))


class UndergroundRerollTest(unittest.TestCase):
    def test_sky_false_rerolls_then_accepts(self):
        cfg = load_config()
        cfg["backend"] = "fake"
        cfg["difficulty"] = "easy"
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        calls = []

        def make(cfg, task, seed):
            calls.append(seed)
            sky = seed != 7
            env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": 6.0}], "can_see_sky": sky})
            return env

        with patch("lfnav.eval.run.make_env", make):
            session = EnvReuse(cfg)
            _env, _obs, meta = open_episode(session, task, cfg, 7)
            session.close()
        self.assertEqual(calls, [7, 7 * 1000 + 1])
        self.assertEqual(meta["world_seed"], 7 * 1000 + 1)
        self.assertEqual(meta["underground_rerolls"], 1)
        self.assertFalse(meta["spawn_underground"])

    def test_all_underground_is_marked_and_not_run(self):
        cfg = load_config()
        cfg["backend"] = "fake"
        cfg["difficulty"] = "easy"
        cfg["difficulty_table"] = dict(cfg["difficulty_table"])
        cfg["difficulty_table"]["easy"] = dict(cfg["difficulty_table"]["easy"], max_reroll=2)
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        calls = []

        def make(cfg, task, seed):
            calls.append(seed)
            return FakeMineDojoEnv(cfg, {"can_see_sky": False})

        with patch("lfnav.eval.run.make_env", make):
            session = EnvReuse(cfg)
            env, obs, meta = open_episode(session, task, cfg, 7)
            rec, events = run_episode(env, task, cfg, seed=7, obs=obs, episode_meta=meta)
            session.close()
        self.assertEqual(calls, [7, 7 * 1000 + 1, 7 * 1000 + 2])
        self.assertTrue(meta["spawn_underground"])
        self.assertEqual(meta["underground_rerolls"], 2)
        self.assertEqual(rec["outcome"], "spawn_underground")
        self.assertFalse(rec["success"])
        self.assertEqual(rec["steps"], 0)
        self.assertEqual(events, [])


class MobRingBoxTest(unittest.TestCase):
    def test_samples_stay_near_the_band(self):
        import random
        rng = random.Random(0)
        min_d, max_d = 8.0, 16.0
        low, high = mob_ring_box(min_d, max_d, seed=3, behind=False)
        w = max(min_d / 2.0, 0.5)
        slack = math.hypot(w, max_d) - max_d + 1e-6
        for _ in range(4000):
            x = rng.uniform(low[0], high[0])
            z = rng.uniform(low[2], high[2])
            dist = math.hypot(x, z)
            self.assertGreaterEqual(dist, min_d - 1e-6)
            self.assertLessEqual(dist, max_d + slack)
        back_low, back_high = mob_ring_box(min_d, max_d, seed=0, behind=True)
        self.assertLess(back_high[2], 0)
        self.assertLessEqual(back_low[2], -min_d + 1e-6)


class SplAndArrayTest(unittest.TestCase):
    def test_spl_formula_and_beyond_range(self):
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        scene = SceneState(objects={"pig": ObjectBelief(1.0, {"center": 1.0}, {"near": 1.0})})
        ok = judge_episode(task, True, 10, scene, 2.0, 20.0, [0, 64, 0], [0, 64, 10], radius=3, initial_target_distance=13)
        self.assertAlmostEqual(ok["spl"], 10.0 / 20.0)
        self.assertEqual(spl_approx(True, 10.0, 20.0), 0.5)
        late = judge_episode(task, False, 3000, None, None, 40, [0], [8], max_steps=3000, initial_target_distance="> 32")
        self.assertIsNone(late["spl"])

    def test_array_covers_forty_eight_unique_cells(self):
        cells = all_cells()
        self.assertEqual(n_cells(), 48)
        self.assertEqual(len(cells), 48)
        self.assertEqual(len(set(cells)), 48)
        self.assertEqual(cells[0], ("1-1", "easy"))
        self.assertEqual(cells[1], ("1-1", "medium"))
        self.assertEqual(cells[2], ("1-1", "hard"))
        self.assertEqual(cells[-1], ("4-4", "hard"))
        cfg = load_config(ROOT / "configs/lfnav/baseline_eval.yaml")
        self.assertTrue(cfg.get("wide_lidar"))
        self.assertFalse(cfg.get("policy_sees_wide_lidar"))
        self.assertEqual(int(cfg["judge"]["max_steps"]), 3000)


if __name__ == "__main__":
    unittest.main()
