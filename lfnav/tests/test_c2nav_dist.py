"""Frame-level distance bins, A-only grads, shared visibility, and the near stop gate."""

from __future__ import annotations

import unittest

import numpy as np

from lfnav.matcher import Matcher
from lfnav.policy.c2nav import (
    C2NavNet,
    DIST_BIN_NAMES,
    STOP_GATE_STOP,
    STOP_GATE_TOO_FAR,
    dist_bin_labels_from_objects,
    freeze_c_params,
    infer_stop_gate,
    obs_from_present_head,
    scene_from_perception,
    spec_from_cfg,
    stop_near_tau_of,
    train_a_only_of,
)
from lfnav.policy.train_c2nav import nav_loss
from lfnav.types import TaskSpec


def _scene():
    return {
        "objects": {
            "pig": {"present": 1.0, "distance": {"near": 0.1, "mid": 0.7, "far": 0.2}},
            "cow": {"present": 0.0, "distance": {"near": 0.9, "mid": 0.1, "far": 0.0}},
            "sheep": {"present": 1.0, "distance": {"near": 0.0, "mid": 0.0, "far": 0.0}},
        }
    }


class DistBinLabelTest(unittest.TestCase):
    def test_visible_argmax_and_invisible_masked(self):
        objects = ["pig", "cow", "sheep", "chicken"]
        bins, mask = dist_bin_labels_from_objects(_scene(), objects)
        self.assertEqual(int(bins[0]), DIST_BIN_NAMES.index("mid"))
        self.assertEqual(float(mask[0]), 1.0)
        self.assertEqual(float(mask[1]), 0.0)
        self.assertEqual(float(mask[2]), 0.0)
        self.assertEqual(float(mask[3]), 0.0)
        bins2, mask2 = dist_bin_labels_from_objects({"objects": {}}, objects)
        self.assertTrue(np.all(mask2 == 0))
        self.assertEqual(int(bins2.sum()), 0)


class TrainAOnlyGradTest(unittest.TestCase):
    def test_c_grads_stay_empty(self):
        import torch

        spec = spec_from_cfg({})
        spec["dim"] = 8
        spec["grid"] = (2, 2)
        spec["hidden"] = 8
        spec["channels"] = 4
        spec["dropout"] = 0.0
        model = C2NavNet(spec)
        n_train = freeze_c_params(model)
        self.assertGreater(n_train, 0)
        k = len(spec["objects"])
        present = np.zeros((1, k, 2, 2), np.float32)
        present[0, 0, 0, 0] = 1.0
        dist_bin = np.zeros((1, k), np.int64)
        dist_bin[0, 0] = 1
        mask = np.zeros((1, k), np.float32)
        mask[0, 0] = 1.0
        batch = {
            "tokens": torch.randn(1, 2, 2, 8),
            "present": torch.from_numpy(present),
            "log_dist": torch.zeros(1, k, 2, 2),
            "ignore": torch.zeros(1, 2, 2),
            "biome": torch.zeros(1, dtype=torch.long),
            "time": torch.zeros(1, dtype=torch.long),
            "weather": torch.zeros(1, dtype=torch.long),
            "brightness": torch.zeros(1, dtype=torch.long),
            "s_vec": torch.zeros(1, spec["s_dim"]),
            "cond": torch.zeros(1, spec["cond_dim"]),
            "geom": torch.zeros(1, spec["geom_dim"]),
            "target_index": torch.zeros(1, dtype=torch.long),
            "bearings": torch.zeros(2, 2),
            "mode": torch.zeros(1, dtype=torch.long),
            "turn": torch.zeros(1, dtype=torch.long),
            "turn_soft": torch.zeros(1, spec["n_turn"]),
            "forward": torch.ones(1, dtype=torch.long),
            "stop": torch.zeros(1),
            "used_wide": torch.zeros(1),
            "has_c": torch.ones(1),
            "vis": torch.ones(1, k),
            "vis_ignore": torch.zeros(1, k),
            "a_only": torch.zeros(1),
            "dist_bin": torch.from_numpy(dist_bin),
            "dist_bin_mask": torch.from_numpy(mask),
            "target_log_dist": torch.zeros(1),
            "target_dist_mask": torch.ones(1),
        }
        model.train()
        out = model(batch)
        loss, parts = nav_loss(
            out, batch, {"a": 1.0, "mode": 1.0, "wp": 1.0, "stop": 1.0},
            torch.ones(k), torch.ones(len(spec["modes"])), torch.ones(spec["n_turn"]),
            torch.ones(len(spec["forward_bins"])), torch.tensor(1.0), 0.0, skip_c=True,
        )
        loss.backward()
        self.assertEqual(parts["mode"], 0.0)
        self.assertEqual(parts["stop"], 0.0)
        self.assertEqual(parts["turn"], 0.0)
        self.assertEqual(parts["forward"], 0.0)
        saw_c = False
        for name, param in model.named_parameters():
            head = name.split(".", 1)[0]
            if head in ("c1", "c2", "mode", "turn", "fwd", "stop"):
                saw_c = True
                self.assertFalse(param.requires_grad, name)
                self.assertIsNone(param.grad, name)
        self.assertTrue(saw_c)
        self.assertIsNotNone(model.dist_bin_head[-1].weight.grad)
        self.assertFalse(train_a_only_of({}))
        self.assertTrue(train_a_only_of({"c2nav": {"train_a_only": True}}))
        self.assertEqual(float(stop_near_tau_of({"c2nav": {"stop_near_tau": 0}})), 0.0)


class VisibilityAgreeTest(unittest.TestCase):
    def test_b_and_c_share_vis_head(self):
        import torch

        spec = spec_from_cfg({})
        spec["vis_tau"] = 0.7
        h, w = 2, 2
        k = len(spec["objects"])
        pig = spec["objects"].index("pig")
        bearings = np.zeros((h, w), np.float32)
        present = np.zeros((1, k, h, w), np.float32)
        present[0, pig, 0, 1] = 0.99
        log_d = np.zeros((1, k, h, w), np.float32)
        vis = torch.full((1, k), -4.0)
        vis[0, pig] = -4.0
        dist_logit = torch.zeros(1, k, 3)
        a = {
            "present_p": torch.from_numpy(present),
            "log_dist": torch.from_numpy(log_d),
            "vis_logit": vis,
            "dist_bin_logit": dist_logit,
        }
        scene = scene_from_perception(a, spec, bearings)
        obs = obs_from_present_head(present[0], log_d[0], pig, bearings, float(vis[0, pig]), spec["vis_tau"])
        match = Matcher({}).match(TaskSpec(id="t", source="", description="", target="pig"), scene)
        self.assertFalse(obs.visible)
        self.assertFalse(match.satisfied["target"])
        vis2 = torch.full((1, k), -4.0)
        vis2[0, pig] = 4.0
        present2 = np.zeros_like(present)
        a["present_p"] = torch.from_numpy(present2)
        a["vis_logit"] = vis2
        scene2 = scene_from_perception(a, spec, bearings)
        obs2 = obs_from_present_head(present2[0], log_d[0], pig, bearings, float(vis2[0, pig]), spec["vis_tau"])
        match2 = Matcher({}).match(TaskSpec(id="t", source="", description="", target="pig"), scene2)
        self.assertTrue(obs2.visible)
        self.assertTrue(match2.satisfied["target"])
        self.assertEqual(bool(obs.visible), bool(match.satisfied["target"]))
        self.assertEqual(bool(obs2.visible), bool(match2.satisfied["target"]))


class StopNearGateTest(unittest.TestCase):
    def test_near_and_numeric_both_required(self):
        self.assertEqual(
            infer_stop_gate(0.0, 0.5, True, {"status": "found", "missing": []}, True, dist=2.0, max_dist=2.7, p_near=0.8, near_tau=0.5),
            STOP_GATE_STOP,
        )
        self.assertEqual(
            infer_stop_gate(0.0, 0.5, True, {"status": "found", "missing": []}, True, dist=2.0, max_dist=2.7, p_near=0.5, near_tau=0.5),
            STOP_GATE_STOP,
        )
        self.assertEqual(
            infer_stop_gate(0.0, 0.5, True, {"status": "found", "missing": []}, True, dist=2.0, max_dist=2.7, p_near=0.49, near_tau=0.5),
            STOP_GATE_TOO_FAR,
        )
        self.assertEqual(
            infer_stop_gate(0.0, 0.5, True, {"status": "found", "missing": []}, True, dist=4.0, max_dist=2.7, p_near=0.9, near_tau=0.5),
            STOP_GATE_TOO_FAR,
        )
        self.assertEqual(
            infer_stop_gate(0.0, 0.5, True, {"status": "found", "missing": []}, True, dist=2.0, max_dist=2.7, p_near=None, near_tau=0.5),
            STOP_GATE_TOO_FAR,
        )
        self.assertEqual(
            infer_stop_gate(0.0, 0.5, True, {"status": "found", "missing": []}, True, dist=2.0, max_dist=2.7),
            STOP_GATE_STOP,
        )
        self.assertEqual(
            infer_stop_gate(0.0, 0.5, True, {"status": "found", "missing": []}, True, dist=2.0, max_dist=2.7, p_near=0.0, near_tau=0),
            STOP_GATE_STOP,
        )
        self.assertEqual(
            infer_stop_gate(0.0, 0.5, True, {"status": "found", "missing": []}, True, dist=4.0, max_dist=2.7, p_near=0.9, near_tau=0),
            STOP_GATE_TOO_FAR,
        )


if __name__ == "__main__":
    unittest.main()
