"""C2Nav unit tests. No MineDojo."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfnav.config import load_config
from lfnav.data.demo_format import write_episode
from lfnav.eval.val_c2nav import val_plan
from lfnav.policy.c2nav import CKPT_STRUCTURE, C2NavAgent, C2NavNet, assert_structure, spec_from_cfg, structure_record
from lfnav.policy.c2nav_labels import compose_waypoint, flip_turn_index, gt_stop, invert_waypoint, labels_from_step
from lfnav.policy.state_tracker import STEPS_NORM, TrackerNoise, TrackerState, flip_vector, replay, update
from lfnav.types import PolicyInput, PolicyOutput

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

ROOT = Path(__file__).resolve().parents[2]


class StateTrackerTest(unittest.TestCase):
    def test_offline_matches_online(self):
        steps = [
            {"pos": [0.0, 64.0, 0.0], "yaw": 0.0, "executed_actions": ["turn_right_15", "turn_right_15"], "match": {"satisfied": {"target": False}}},
            {"pos": [0.0, 64.0, 0.0], "yaw": 30.0, "executed_actions": ["forward"], "target_obs": {"visible": True, "bearing_deg": 10.0, "log_distance": 1.5}, "match": {"satisfied": {"target": True}, "target_direction": "center"}},
            {"pos": [0.2, 64.0, 1.0], "yaw": 30.0, "executed_actions": [], "match": {"satisfied": {"target": True}, "target_direction": "center"}, "target_distance_num": 4.0},
        ]
        offline = replay(steps)
        online = TrackerState()
        online_rows = []
        prev = []
        for i, step in enumerate(steps):
            from lfnav.policy.state_tracker import observation_from_step

            obs = observation_from_step(step)
            online = update(online, step["pos"], step["yaw"], [] if i == 0 else prev, obs)
            online_rows.append(online)
            prev = list(step.get("executed_actions") or [])
        for a, b in zip(offline, online_rows):
            np.testing.assert_allclose(a.vector(), b.vector(), atol=1e-6)

    def test_steps_frac_uses_fixed_3000(self):
        self.assertEqual(STEPS_NORM, 3000)
        state = TrackerState(max_steps=1000, env_steps=300)
        self.assertAlmostEqual(state.features()["steps_frac"], 300.0 / 3000.0)
        long = TrackerState(max_steps=1000, env_steps=4000)
        self.assertEqual(long.features()["steps_frac"], 1.0)

    def test_flip_twice_restores(self):
        vec = [0.3, 0.9, 1.2, 0.1, 0.8, 0.4, 0.2, 0.1, 0.2, 0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 3.0, 0.0, 0.1]
        twice = flip_vector(flip_vector(vec))
        np.testing.assert_allclose(twice, vec, atol=1e-9)
        self.assertEqual(flip_turn_index(flip_turn_index(3)), 3)
        self.assertEqual(flip_turn_index(flip_turn_index(0)), 0)


class WaypointTest(unittest.TestCase):
    def test_compose_invert_roundtrip(self):
        for actions in (
            ["turn_right_15", "forward", "forward"],
            ["turn_left_45", "turn_left_45"],
            ["backward"],
            ["forward"],
            ["turn_right_45", "turn_right_15", "forward", "forward", "forward"],
        ):
            wp = compose_waypoint(actions)
            back = invert_waypoint(wp["turn_index"], wp["forward_index"])
            again = compose_waypoint(back)
            self.assertEqual(wp["turn_index"], again["turn_index"], actions)
            self.assertEqual(wp["forward_index"], again["forward_index"], actions)

    def test_stop_gt(self):
        step = {
            "match": {"satisfied": {"target": True, "biome:plains": True}, "missing": [], "target_direction": "center"},
            "target_distance_num": 2.0,
        }
        self.assertTrue(gt_stop(step, radius=3.0))
        step["target_distance_num"] = 9.0
        self.assertFalse(gt_stop(step, radius=3.0))

    def test_legacy_has_no_mode_labels(self):
        self.assertIsNone(labels_from_step({"action_chunk": ["forward"], "match": {}}))
        row = labels_from_step({
            "expert_branch": "wide_turn",
            "used_wide": True,
            "executed_actions": ["turn_right_15"],
            "pos": [0, 64, 0],
            "match": {"satisfied": {"target": False}, "missing": ["target"]},
        })
        self.assertTrue(row["used_wide"])
        self.assertEqual(row["mode"], 1)


class EasyTrainConfigTest(unittest.TestCase):
    def test_c2nav_easy_yaml(self):
        cfg = load_config(ROOT / "configs" / "lfnav" / "c2nav_easy.yaml")
        nav = cfg["c2nav"]
        self.assertEqual(nav["features_version"], "nav_v3_easy")
        self.assertEqual(nav["train_difficulties"], ["easy"])
        self.assertEqual(nav["val_seeds"], [1, 13, 23, 26, 37, 45])
        self.assertEqual(nav["epochs"], 20)
        self.assertAlmostEqual(float(nav["lr"]), 3e-4)
        self.assertAlmostEqual(float(nav["weight_decay"]), 0.05)
        self.assertAlmostEqual(float(nav["dropout"]), 0.3)
        self.assertAlmostEqual(float(nav["label_smoothing"]), 0.1)
        dirs = [str(p) for p in nav["demo_dirs"]]
        self.assertTrue(all("demos_main_v3" in p for p in dirs))
        self.assertFalse(any("demos_main" in Path(p).name and "v3" not in Path(p).name for p in dirs))
        cells, seed, max_steps = val_plan(cfg)
        self.assertEqual(seed, 13)
        self.assertEqual(max_steps, 1000)
        self.assertEqual(cells, [("1-3", "easy"), ("1-4", "easy"), ("2-3", "easy"), ("4-4", "easy")])


class CheckpointTest(unittest.TestCase):
    def test_missing_field_raises(self):
        with self.assertRaises(RuntimeError):
            assert_structure({"version": "c2nav_v1"})
        spec = spec_from_cfg(load_config())
        payload = structure_record(spec, model={})
        for key in CKPT_STRUCTURE:
            self.assertIn(key, payload)
        broken = dict(payload)
        broken.pop("hidden")
        with self.assertRaises(RuntimeError):
            assert_structure(broken)


class SmokeDemoTest(unittest.TestCase):
    def test_replay_and_labels_from_v3_episode(self):
        record = {
            "task_name": "1-4",
            "success": True,
            "seed": 1,
            "difficulty": "easy",
            "config_name": "main",
            "hostile_seen": 0,
            "peaceful_sent": True,
            "initial_target_distance": 6.0,
        }
        rows = [
            {
                "decision": 0,
                "env_step": 0,
                "pos": [0.0, 64.0, 0.0],
                "yaw": 0.0,
                "pitch": 0.0,
                "executed_actions": ["turn_right_15"],
                "expert_branch": "scan",
                "used_wide": False,
                "target_distance_num": None,
                "action_chunk": ["turn_right_15"],
                "match": {"satisfied": {"target": False}, "missing": ["target"], "stop": False},
                "scene": {"objects": {}, "biome": {"plains": 1.0}, "time": {"day": 1.0}, "weather": {"clear": 1.0}, "brightness": {"sufficient": 1.0}, "patches": {}, "ignore": []},
            },
            {
                "decision": 1,
                "env_step": 1,
                "pos": [0.0, 64.0, 0.0],
                "yaw": 15.0,
                "pitch": 0.0,
                "executed_actions": ["forward", "forward"],
                "expert_branch": "approach",
                "used_wide": False,
                "target_distance_num": 2.5,
                "action_chunk": ["forward", "forward"],
                "match": {"satisfied": {"target": True}, "missing": [], "stop": False, "target_direction": "center"},
                "scene": {"objects": {"pig": {"present": 1.0}}, "biome": {"plains": 1.0}, "time": {"day": 1.0}, "weather": {"clear": 1.0}, "brightness": {"sufficient": 1.0}, "patches": {}, "ignore": []},
            },
            {
                "decision": 2,
                "env_step": 3,
                "pos": [0.0, 64.0, 2.0],
                "yaw": 15.0,
                "pitch": 0.0,
                "executed_actions": [],
                "expert_branch": "stop",
                "used_wide": False,
                "target_distance_num": 1.5,
                "action_chunk": None,
                "match": {"satisfied": {"target": True}, "missing": [], "stop": True, "target_direction": "center"},
                "scene": {"objects": {"pig": {"present": 1.0}}, "biome": {"plains": 1.0}, "time": {"day": 1.0}, "weather": {"clear": 1.0}, "brightness": {"sufficient": 1.0}, "patches": {}, "ignore": []},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_episode(tmp, "success", "1-4", "easy", 1, record, rows)
            from lfnav.data.demo_format import read_episode

            packed = read_episode(folder)
            states = replay(packed["steps"])
            self.assertEqual(len(states), 3)
            labs = [labels_from_step(s) for s in packed["steps"]]
            self.assertEqual(labs[0]["mode"], 0)
            self.assertEqual(labs[1]["mode"], 1)
            self.assertEqual(labs[2]["mode"], 4)
            self.assertTrue(labs[2]["stop"])
            PolicyOutput(stop=True, info={"p_stop": 0.9})
            noise = TrackerNoise(drop_visible=1.0)
            noisy = replay(packed["steps"], noise=noise, rng=__import__("random").Random(0))
            self.assertEqual(len(noisy), 3)


@unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
class ForwardSmokeTest(unittest.TestCase):
    def test_collate_train_step_and_infer_forward(self):
        import random

        import torch

        from lfnav.policy.train_c2nav import NavSample, collate, nav_loss, _to_torch

        spec = spec_from_cfg(load_config())
        spec["dim"] = 8
        spec["grid"] = (4, 6)
        spec["hidden"] = 16
        spec["channels"] = 4
        gh, gw, dim = 4, 6, 8
        k = len(spec["objects"])
        tokens = np.random.randn(2, gh, gw, dim).astype(np.float32)
        rows = []
        for i in range(2):
            present = np.zeros((k, gh, gw), np.float32)
            present[0, 1, 2] = 1.0
            rows.append(NavSample(
                row=i,
                present=present,
                log_dist=np.zeros((k, gh, gw), np.float32),
                ignore=np.zeros((gh, gw), np.float32),
                biome=0,
                time=0,
                weather=0,
                brightness=0,
                s_vec=np.zeros(spec["s_dim"], np.float32),
                cond=np.zeros(spec["cond_dim"], np.float32),
                geom=np.asarray([1.0, 0.0, 1.0, 1.0], np.float32),
                target_index=0,
                mode=1,
                turn=3,
                forward=2,
                stop=i,
                used_wide=False,
                has_c=True,
                seed=i,
                world_seed=i,
                episode_id=f"1-4/easy/{i}",
            ))
        batch_np = collate(rows, tokens, False, random.Random(0))
        bearings = np.zeros((gh, gw), np.float32)
        batch = _to_torch(batch_np, torch.device("cpu"), bearings=bearings)
        model = C2NavNet(spec)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.train()
        opt.zero_grad()
        out = model(batch)
        wts = {"a": 1.0, "mode": 1.0, "wp": 1.0, "stop": 1.0}
        loss, _ = nav_loss(
            out, batch, wts,
            torch.ones(k), torch.ones(len(spec["modes"])),
            torch.ones(spec["n_turn"]), torch.ones(len(spec["forward_bins"])),
            torch.tensor(1.0), 0.1,
        )
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            infer_out = model(batch)
        self.assertEqual(tuple(infer_out["mode"].shape), (2, len(spec["modes"])))
        agent = C2NavAgent({"dry_run": True, "backend": "fake"})
        agent.model = model
        agent.spec = spec
        agent.bearings = bearings
        agent._tokens = np.asarray(batch_np["tokens"][0], np.float32)
        out_act = agent.act(PolicyInput(task={"target": "pig"}, match={}, memory={"pos": [0.0, 64.0, 0.0], "yaw": 0.0}))
        self.assertIsNotNone(out_act)


if __name__ == "__main__":
    unittest.main()
