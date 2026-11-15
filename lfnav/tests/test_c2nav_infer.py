"""Inference GT obs, waypoint split, and opening S. No MineDojo."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lfnav.config import load_config
from lfnav.data.demo_format import read_episode
from lfnav.policy.c2_data import demo_roots, iter_success_episodes, load_task
from lfnav.policy.c2nav import (
    C2NavAgent,
    OBS_GT,
    STOP_GATE_CONDITIONS,
    STOP_GATE_NOT_VISIBLE,
    STOP_GATE_STOP,
    STOP_GATE_TOO_FAR,
    encode_s,
    geom_from_obs,
    gt_target_obs,
    infer_stop_gate,
    obs_source_of,
    patch_bearings,
    patch_targets,
    spec_from_cfg,
    stop_max_dist_of,
    stop_requires_visible_of,
)
from lfnav.policy.state_tracker import STEPS_NORM, TargetObs, replay
from lfnav.policy.c2nav_labels import FORWARD_BINS, TURN_BINS, invert_waypoint, turn_index
from lfnav.types import PolicyInput

ROOT = Path(__file__).resolve().parents[2]


def _gt_agent(cfg=None):
    base = {"dry_run": True, "backend": "fake", "c2nav": {"obs_source": "gt", "encoder": "dinov2_vits14"}}
    if cfg is None:
        return C2NavAgent(base)
    merged = dict(base)
    extra_nav = dict(cfg.get("c2nav") or {})
    merged.update({k: v for k, v in cfg.items() if k != "c2nav"})
    nav = dict(base["c2nav"])
    nav.update(extra_nav)
    merged["c2nav"] = nav
    return C2NavAgent(merged)


class ObsSourceGtTest(unittest.TestCase):
    def test_obs_source_default_pred(self):
        self.assertEqual(obs_source_of({}), "pred")
        self.assertEqual(obs_source_of({"c2nav": {}}), "pred")
        self.assertEqual(obs_source_of({"c2nav": {"obs_source": None}}), "pred")
        self.assertEqual(obs_source_of({"c2nav": {"obs_source": "gt"}}), "gt")

    def test_gt_does_not_load_encoder(self):
        with patch("lfnav.perception.encoder.load_encoder") as load_enc:
            agent = _gt_agent()
            load_enc.assert_not_called()
            self.assertIsNone(agent.encoder)
            self.assertEqual(agent.obs_source, OBS_GT)
            scene = {
                "patches": {"pig": np.zeros((16, 26), np.float32).tolist()},
                "ignore": np.zeros((16, 26), np.float32).tolist(),
                "biome": {"plains": 1.0},
                "time": {"day": 1.0},
                "weather": {"clear": 1.0},
                "brightness": {"sufficient": 1.0},
            }
            with patch.object(agent.model, "perception") as perc:
                agent.perceive(scene)
                agent.act(PolicyInput(task={"target": "pig"}, match={}, memory={"pos": [0.0, 64.0, 0.0], "yaw": 0.0}))
                perc.assert_not_called()
            load_enc.assert_not_called()
            self.assertIsNone(agent.encoder)

    def test_train_and_infer_gt_obs_match(self):
        cfg = load_config(ROOT / "configs" / "lfnav" / "c2nav_easy.yaml")
        spec = spec_from_cfg(cfg)
        objects = list(spec["objects"])
        grid = tuple(spec["grid"])
        bearings = patch_bearings(int(grid[0]), int(grid[1]), cfg)
        raw = np.zeros((16, 26), np.float32)
        raw[:, 20:22] = 1.0
        scene = {
            "patches": {"cow": raw.tolist()},
            "ignore": np.zeros((16, 26), np.float32).tolist(),
        }
        ti = objects.index("cow")
        p, _, ig = patch_targets(scene, objects, int(spec["token_pool"]), grid)
        from lfnav.policy.c2nav import attach_lcr_dist

        train_obs = attach_lcr_dist(gt_target_obs(p, ig, ti, bearings, 6.0), raw, scene.get("ignore"), 6.0, cfg)
        agent = _gt_agent({"c2nav": {"obs_source": "gt"}, "image_hw": [224, 364], "fov_deg": 70, "labels": cfg.get("labels")})
        infer_obs = agent.gt_obs_from_scene(scene, "cow", 6.0)
        self.assertEqual(bool(train_obs.visible), bool(infer_obs.visible))
        self.assertTrue(train_obs.visible)
        self.assertAlmostEqual(float(train_obs.bearing_deg), float(infer_obs.bearing_deg), places=5)
        self.assertAlmostEqual(float(train_obs.log_distance), float(infer_obs.log_distance), places=5)
        np.testing.assert_allclose(geom_from_obs(train_obs), geom_from_obs(infer_obs), atol=1e-6)


class WaypointSplitTest(unittest.TestCase):
    def test_forward_counts_are_action_counts(self):
        for n_fwd in (0, 1, 2):
            names = invert_waypoint(turn_index(0), FORWARD_BINS.index(n_fwd) if n_fwd in FORWARD_BINS else 1)
            self.assertEqual(sum(1 for n in names if n == "forward"), n_fwd, names)
            self.assertFalse(any("turn" in n for n in names), names)

    def test_turns_use_45_then_15(self):
        expected = {
            -135: ["turn_left_45"] * 3,
            -90: ["turn_left_45"] * 2,
            -45: ["turn_left_45"],
            -15: ["turn_left_15"],
            0: [],
            15: ["turn_right_15"],
            45: ["turn_right_45"],
            90: ["turn_right_45"] * 2,
            135: ["turn_right_45"] * 3,
            180: ["turn_right_45"] * 4,
        }
        for deg in TURN_BINS:
            names = invert_waypoint(turn_index(deg), FORWARD_BINS.index(0))
            self.assertEqual(names, expected[int(deg)], deg)
            last45 = [i for i, n in enumerate(names) if n.endswith("_45")]
            first15 = [i for i, n in enumerate(names) if n.endswith("_15")]
            if last45 and first15:
                self.assertLess(max(last45), min(first15), names)

    def test_agent_emits_action_chunk_not_waypoints(self):
        agent = _gt_agent()
        agent.perceive({
            "patches": {"pig": np.zeros((8, 13), np.float32).tolist()},
            "ignore": np.zeros((8, 13), np.float32).tolist(),
        })
        out = agent.act(PolicyInput(task={"target": "pig"}, match={}, memory={"pos": [0.0, 64.0, 0.0], "yaw": 0.0}))
        self.assertTrue(out.stop or out.action_chunk)
        self.assertEqual(list(out.waypoints), [])


class OpeningStateTest(unittest.TestCase):
    def _episode_folders(self):
        cfg = load_config(ROOT / "configs" / "lfnav" / "c2nav_easy.yaml")
        nav = cfg.get("c2nav") if cfg.get("c2nav") is not None else {}
        found = []
        for root in demo_roots(nav.get("demo_dirs"), cfg):
            found.extend(iter_success_episodes(root))
            if len(found) >= 2:
                break
        return cfg, found[:2]

    def test_first_s_matches_train_replay_on_two_episodes(self):
        cfg, folders = self._episode_folders()
        self.assertGreaterEqual(len(folders), 2, "need two demos_main_v3 episodes")
        spec = spec_from_cfg(cfg)
        objects = list(spec["objects"])
        grid = tuple(spec["grid"])
        pool = int(spec["token_pool"] if spec.get("token_pool") is not None else 2)
        bearings = patch_bearings(int(grid[0]), int(grid[1]), cfg)
        for folder in folders:
            packed = read_episode(folder)
            self.assertTrue(packed["steps"], folder)
            task = load_task(packed["record"], folder, cfg)
            target = str(task.get("target") or "pig")
            ti = objects.index(target) if target in objects else 0
            gt_obs = []
            for step in packed["steps"]:
                present, _, ignore = patch_targets(step.get("scene") or {}, objects, pool, grid)
                gt_obs.append(gt_target_obs(present, ignore, ti, bearings, step.get("target_distance_num")))
            train_s = encode_s(replay(packed["steps"], max_steps=STEPS_NORM, noise=None, observations=gt_obs)[0])
            step0 = packed["steps"][0]
            agent = C2NavAgent({
                "dry_run": True,
                "backend": "fake",
                "image_hw": cfg.get("image_hw"),
                "fov_deg": cfg.get("fov_deg"),
                "labels": cfg.get("labels"),
                "c2nav": {"obs_source": "gt", "encoder": "dinov2_vits14"},
            })
            seen = {}
            real_policy = agent.model.policy

            def _policy(geom, s_vec, cond):
                seen["s"] = s_vec.detach().cpu().numpy()[0]
                return real_policy(geom, s_vec, cond)

            agent.model.policy = _policy
            agent.perceive(step0.get("scene") or {})
            agent.act(PolicyInput(
                task={"target": target},
                match=dict(step0.get("match") or {}),
                scene={"target_distance_num": step0.get("target_distance_num")},
                memory={"pos": step0.get("pos"), "yaw": step0.get("yaw")},
            ))
            infer_s = encode_s(agent._s)
            np.testing.assert_allclose(np.asarray(infer_s), np.asarray(train_s), atol=1e-6, err_msg=str(folder))
            self.assertIn("s", seen, folder)
            np.testing.assert_allclose(np.asarray(seen["s"]), np.asarray(train_s), atol=1e-6, err_msg=f"C input {folder}")


FOUND_MATCH = {"status": "found", "stop": False, "missing": [], "satisfied": {"target": True}}
MISSING_COND = {
    "status": "not_found",
    "stop": False,
    "missing": ["biome:forest"],
    "satisfied": {"target": True, "biome:forest": False},
}


def _force_p_stop(agent, p=0.9):
    import math

    import torch

    logit = math.log(p / (1.0 - p))
    real = agent.model.policy

    def wrapped(geom, s_vec, cond):
        out = dict(real(geom, s_vec, cond))
        out["stop"] = torch.full_like(out["stop"], logit)
        return out

    agent.model.policy = wrapped


def _act(agent, visible, match, dist_num=2.5):
    raw = np.zeros((16, 26), np.float32)
    if visible:
        raw[:, 10:12] = 1.0
    agent.reset()
    agent.perceive({
        "patches": {"pig": raw.tolist()},
        "ignore": np.zeros((16, 26), np.float32).tolist(),
    })
    scene = {}
    if dist_num is not None:
        scene["target_distance_num"] = dist_num
    return agent.act(PolicyInput(
        task={"target": "pig"},
        match=match,
        scene=scene,
        memory={"pos": [0.0, 64.0, 0.0], "yaw": 0.0},
    ))


class StopGateTest(unittest.TestCase):
    def test_stop_requires_visible_defaults_true(self):
        self.assertTrue(stop_requires_visible_of({}))
        self.assertTrue(stop_requires_visible_of({"c2nav": {}}))
        self.assertTrue(stop_requires_visible_of({"c2nav": {"stop_requires_visible": None}}))
        self.assertFalse(stop_requires_visible_of({"c2nav": {"stop_requires_visible": False}}))

    def test_high_p_stop_not_visible_does_not_stop(self):
        self.assertEqual(infer_stop_gate(0.9, 0.5, False, FOUND_MATCH, True), STOP_GATE_NOT_VISIBLE)
        agent = _gt_agent()
        _force_p_stop(agent, 0.9)
        out = _act(agent, visible=False, match=FOUND_MATCH)
        self.assertFalse(out.stop)
        self.assertTrue(out.action_chunk)
        self.assertEqual(out.info.get("stop_gate"), STOP_GATE_NOT_VISIBLE)
        self.assertFalse(out.info.get("visible"))
        self.assertEqual(int(agent.n_stop_blocked), 1)

    def test_high_p_stop_visible_and_conditions_stops(self):
        self.assertEqual(
            infer_stop_gate(0.9, 0.5, True, FOUND_MATCH, True, dist=2.5, max_dist=2.7),
            STOP_GATE_STOP,
        )
        agent = _gt_agent()
        _force_p_stop(agent, 0.9)
        out = _act(agent, visible=True, match=FOUND_MATCH, dist_num=2.5)
        self.assertTrue(out.stop)
        self.assertEqual(out.info.get("stop_gate"), STOP_GATE_STOP)
        self.assertTrue(out.info.get("visible"))
        self.assertEqual(int(agent.n_stop_blocked), 0)

    def test_too_far_does_not_stop(self):
        self.assertEqual(
            infer_stop_gate(0.9, 0.5, True, FOUND_MATCH, True, dist=3.1, max_dist=2.7),
            STOP_GATE_TOO_FAR,
        )
        agent = _gt_agent()
        _force_p_stop(agent, 0.9)
        out = _act(agent, visible=True, match=FOUND_MATCH, dist_num=3.1)
        self.assertFalse(out.stop)
        self.assertTrue(out.action_chunk)
        self.assertEqual(out.info.get("stop_gate"), STOP_GATE_TOO_FAR)
        self.assertTrue(out.info.get("visible"))
        self.assertAlmostEqual(float(out.info.get("dist")), 3.1, places=5)
        self.assertEqual(int(agent.n_stop_blocked_too_far), 1)
        self.assertEqual(int(agent.n_stop_blocked), 1)

    def test_close_enough_stops(self):
        agent = _gt_agent()
        _force_p_stop(agent, 0.9)
        out = _act(agent, visible=True, match=FOUND_MATCH, dist_num=2.5)
        self.assertTrue(out.stop)
        self.assertEqual(out.info.get("stop_gate"), STOP_GATE_STOP)
        self.assertAlmostEqual(float(out.info.get("dist")), 2.5, places=5)
        self.assertEqual(int(agent.n_stop_blocked_too_far), 0)

    def test_none_dist_does_not_stop(self):
        self.assertEqual(
            infer_stop_gate(0.9, 0.5, True, FOUND_MATCH, True, dist=None, max_dist=2.7),
            STOP_GATE_TOO_FAR,
        )
        agent = _gt_agent()
        _force_p_stop(agent, 0.9)
        agent.gt_obs_from_scene = lambda scene, target, dist_num: TargetObs(
            visible=True, bearing_deg=0.0, log_distance=None,
        )
        out = _act(agent, visible=True, match=FOUND_MATCH, dist_num=None)
        self.assertFalse(out.stop)
        self.assertTrue(out.action_chunk)
        self.assertEqual(out.info.get("stop_gate"), STOP_GATE_TOO_FAR)
        self.assertIsNone(out.info.get("dist"))
        self.assertEqual(int(agent.n_stop_blocked_too_far), 1)

    def test_stop_max_dist_zero_is_kept(self):
        self.assertEqual(float(stop_max_dist_of({"c2nav": {"stop_max_dist": 0}})), 0.0)
        self.assertAlmostEqual(float(stop_max_dist_of({"judge": {"radius": 3.0}})), 2.7)
        self.assertAlmostEqual(float(stop_max_dist_of({"c2nav": {"stop_max_dist": None}, "judge": {"radius": 3.0}})), 2.7)
        agent = _gt_agent({"c2nav": {"obs_source": "gt", "stop_max_dist": 0}})
        self.assertEqual(float(agent.stop_max_dist), 0.0)

    def test_high_p_stop_visible_but_conditions_fail_does_not_stop(self):
        self.assertEqual(infer_stop_gate(0.9, 0.5, True, MISSING_COND, True), STOP_GATE_CONDITIONS)
        agent = _gt_agent()
        _force_p_stop(agent, 0.9)
        out = _act(agent, visible=True, match=MISSING_COND)
        self.assertFalse(out.stop)
        self.assertTrue(out.action_chunk)
        self.assertEqual(out.info.get("stop_gate"), STOP_GATE_CONDITIONS)
        self.assertTrue(out.info.get("visible"))
        self.assertEqual(int(agent.n_stop_blocked), 1)


if __name__ == "__main__":
    unittest.main()
