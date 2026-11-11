"""Post-reset /summon sampling and settle no-ops. No MineDojo import."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lfnav.config import load_config
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.env.minedojo_env import (
    convert_rays,
    count_hostile_hits,
    lidar_plan,
    sample_mob_offset,
    spawn_rng_seed,
    spawn_task_mobs,
    task_mentions_grass,
    vegetation_fill_commands,
)
from lfnav.env.probe_commands import (
    after_summon,
    check_peaceful,
    format_table,
    raw_name_counts,
    spin_scan,
    summon_trial,
)
from lfnav.gt.initial import apply_spawn_offset_distance, initial_scene_metrics
from lfnav.types import TaskSpec

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "lfnav" / "tasks" / "specs"


def _stub(cfg, cmds, steps, invalid=None):
    invalid = set(invalid or [])
    obs = {
        "location_stats": {"yaw": 0.0, "pos": [0.0, 64.0, 0.0]},
        "lfnav_rays": [],
        "lfnav_wide_rays": [],
    }

    class InnerSpace:
        def no_op(self):
            return "NOOP"

    class ChatSpace:
        def no_op(self):
            return {"chat": ""}

    class MineDojoSim:
        def __init__(self):
            self.action_space = ChatSpace()

        def execute_cmd(self, cmd):
            raise RuntimeError(f"command not sent via _send_chat and not on whitelist: {cmd}")

        def step(self, action):
            chat = action.get("chat") if isinstance(action, dict) else None
            out = dict(obs)
            info = {}
            if chat:
                cmds.append(chat)
                if chat in invalid:
                    info["chat"] = "Invalid cmd"
                    out["chat"] = "Invalid cmd"
            return out, 0.0, False, info

    class Inner:
        def __init__(self):
            self.action_space = InnerSpace()
            self.env = MineDojoSim()

        def step(self, action):
            steps.append(action)
            return dict(obs), 0.0, False, {}

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

    def test_task_id_changes_layout(self):
        kwargs = dict(min_d=4.0, max_d=8.0, yaw_deg=0.0, seed=9, index=0)
        a = sample_mob_offset(**kwargs, task_id="1-4")
        b = sample_mob_offset(**kwargs, task_id="1-4")
        c = sample_mob_offset(**kwargs, task_id="2-4")
        self.assertEqual((a["dx"], a["dz"]), (b["dx"], b["dz"]))
        self.assertNotEqual((a["dx"], a["dz"]), (c["dx"], c["dz"]))
        self.assertNotEqual(spawn_rng_seed(9, "1-4", 0), spawn_rng_seed(9, "2-4", 0))


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
        self.assertNotIn("/difficulty peaceful", cmds)
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
        self.assertTrue(any("ActiveEffects" in c and "Id:11" in c for c in cmds))
        self.assertFalse(any("/effect" in c and "resistance" in c for c in cmds))
        self.assertTrue(any("slowness 1000000 10 true" in c for c in cmds))
        self.assertLess(cmds.index(fills[0]), next(i for i, c in enumerate(cmds) if c.startswith("/summon")))

    def test_peaceful_is_first_command(self):
        cfg = self._cfg(peaceful=True)
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        cmds, steps = [], []
        wrapper = _stub(cfg, cmds, steps)
        spawn_task_mobs(wrapper, task, cfg, seed=3)
        self.assertEqual(cmds[0], "/difficulty peaceful")
        self.assertTrue(wrapper.spawn_meta.get("peaceful"))
        self.assertTrue(wrapper.spawn_meta.get("peaceful_sent"))
        self.assertTrue(wrapper.spawn_meta.get("peaceful_ok"))
        self.assertLess(cmds.index("/difficulty peaceful"), cmds.index("/kill @e[type=!player]"))

    def test_dynamic_configs_enable_peaceful(self):
        for rel in ("configs/lfnav/eval_dynamic.yaml", "configs/lfnav/demo_dynamic.yaml", "configs/lfnav/main.yaml"):
            cfg = load_config(ROOT / rel)
            self.assertTrue(cfg["minedojo"]["peaceful"], rel)
        self.assertFalse(load_config()["minedojo"]["peaceful"])

    def test_send_chat_raises_on_invalid_cmd(self):
        from lfnav.env.minedojo_env import _send_chat
        cfg = self._cfg()
        cmds, steps = [], []
        wrapper = _stub(cfg, cmds, steps, invalid={"/nope"})
        with self.assertRaises(RuntimeError) as ctx:
            _send_chat(wrapper, "/nope")
        self.assertIn("Invalid cmd", str(ctx.exception))

    def test_execute_cmd_without_send_chat_is_rejected_on_fake(self):
        cfg = load_config()
        env = FakeMineDojoEnv(cfg, {})
        with self.assertRaises(RuntimeError) as ctx:
            env.execute_cmd("/kill @e[type=!player]")
        self.assertIn("_send_chat", str(ctx.exception))
        _send_chat = __import__("lfnav.env.minedojo_env", fromlist=["_send_chat"])._send_chat
        _send_chat(env, "/difficulty peaceful")
        self.assertEqual(env.chat_log[-1], "/difficulty peaceful")

    def test_chat_actor_uses_inner_sim_dict_not_outer_numpy(self):
        from lfnav.env.minedojo_env import _chat_actor, _send_chat

        cfg = load_config()
        env = FakeMineDojoEnv(cfg, {})
        outer = env.action_space.no_op()
        self.assertFalse(isinstance(outer, dict))
        actor = _chat_actor(env)
        self.assertEqual(type(actor).__name__, "MineDojoSim")
        inner = actor.action_space.no_op()
        self.assertIsInstance(inner, dict)
        _send_chat(env, "/difficulty peaceful")
        self.assertEqual(env.chat_log[-1], "/difficulty peaceful")
        self.assertEqual(actor.last_chat, "/difficulty peaceful")
        self.assertIn("chat", actor.action_space.no_op())

    def test_chat_actor_lists_layers_when_missing(self):
        from lfnav.env.minedojo_env import _chat_actor

        class Leaf:
            pass

        class Mid:
            def __init__(self):
                self.env = Leaf()

        wrapper = SimpleNamespace(_env=Mid())
        with self.assertRaises(RuntimeError) as ctx:
            _chat_actor(wrapper)
        self.assertIn("Mid", str(ctx.exception))
        self.assertIn("Leaf", str(ctx.exception))

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


class HostileCountTest(unittest.TestCase):
    def test_zombie_rays_are_counted(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "zombie", "x": 0.0, "z": 3.0}]})
        n = count_hostile_hits(env.reset(), cfg)
        self.assertGreater(n, 0)
        pig = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": 3.0}]})
        self.assertEqual(count_hostile_hits(pig.reset(), cfg), 0)

    def test_convert_rays_keeps_raw_and_unlabeled_zombie_counts(self):
        cfg = load_config()
        plan = {
            "aligned": (0, 1),
            "wide": (1, 1),
            "rows_cols": [(0, 0)],
            "wide_angles_deg": [],
            "subrays": 1,
        }
        rays = {
            "block_name": ["minecraft:air"],
            "block_distance": [12.0],
            "entity_name": ["minecraft:Zombie"],
            "entity_distance": [4.0],
        }
        converted = convert_rays(rays, plan, cfg)
        cell = converted["aligned"][0]
        self.assertEqual(cell["raw"], "zombie")
        self.assertIsNone(cell["label"])
        obs = {"lfnav_rays": converted["aligned"], "lfnav_wide_rays": []}
        self.assertGreater(count_hostile_hits(obs, cfg), 0)
        unlabeled = {"lfnav_rays": [{"label": None, "raw": "minecraft:zombie", "distance": 3.0}], "lfnav_wide_rays": []}
        self.assertGreater(count_hostile_hits(unlabeled, cfg), 0)
        self.assertEqual(count_hostile_hits({"lfnav_rays": [{"label": "zombie", "distance": 3.0}], "lfnav_wide_rays": []}, cfg), 0)

    def test_probe_fails_when_detector_cannot_see_zombie(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        env = FakeMineDojoEnv(cfg, {})
        with tempfile.TemporaryDirectory() as tmp:
            detail = check_peaceful(env, cfg, out=tmp)
        self.assertEqual(detail, "detector cannot see zombie")
        self.assertIn("/difficulty easy", env.chat_log)
        self.assertIn("/summon zombie ~ ~ ~4", env.chat_log)

    def test_raw_counts_and_spin_see_pig(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": 3.0}]})
        obs = env.reset()
        counts = raw_name_counts(obs, cfg)
        self.assertGreater(counts.get("pig", 0), 0)
        seen, _obs = spin_scan(env, cfg, "pig")
        self.assertTrue(seen)
        empty = FakeMineDojoEnv(cfg, {})
        empty.reset()
        missed, _obs = spin_scan(empty, cfg, "pig")
        self.assertFalse(missed)

    def test_after_summon_writes_jpgs_and_table(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": 3.0}]})
        env.reset()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            seen = after_summon(env, cfg, "pig", "pig_send_chat", dest)
            self.assertTrue(seen)
            for tick in (5, 10, 15, 20):
                self.assertTrue((dest / f"pig_send_chat_{tick}.jpg").is_file(), tick)
            row = summon_trial(env, cfg, "pig", "send_chat", "/summon pig ~ ~1 ~3", "pig_send_chat", dest)
            self.assertTrue(row["seen"])
            exec_row = summon_trial(env, cfg, "pig", "execute_cmd", "/summon pig ~ ~1 ~3", "pig_execute_cmd", dest)
            self.assertIsNotNone(exec_row["error"])
            text = format_table([exec_row, row])
            self.assertIn("execute_cmd", text)
            self.assertIn("send_chat", text)
            self.assertIn("pig", text)


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
