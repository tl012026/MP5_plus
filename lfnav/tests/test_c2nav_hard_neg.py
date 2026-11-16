"""Hard-negative mining rule and the visibility / present loss weight."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

torch.set_num_threads(4)

from lfnav.policy.c2nav import GEOM_DIM
from lfnav.policy.train_c2nav import (
    NavSample,
    hard_neg_counts,
    hard_neg_epoch_report,
    hard_neg_of,
    hard_neg_weight_keys,
    hard_neg_weights,
    hard_negative_mask,
    mine_hard_negatives,
    nav_loss,
)


def _logit(p: float) -> float:
    p = float(p)
    return math.log(p / (1.0 - p))


def _sample(store, row, vis, vis_ignore, present, ignore, target_index=0, target_dist=None):
    k, h, w = present.shape
    mask = 0.0 if target_dist is None else 1.0
    log_d = 0.0 if target_dist is None else float(math.log(float(target_dist)))
    return NavSample(
        row=int(row),
        present=np.asarray(present, np.float32),
        log_dist=np.zeros((k, h, w), np.float32),
        ignore=np.asarray(ignore, np.float32),
        biome=0,
        time=0,
        weather=0,
        brightness=0,
        s_vec=np.zeros(4, np.float32),
        cond=np.zeros(4, np.float32),
        geom=np.zeros(GEOM_DIM, np.float32),
        target_index=int(target_index),
        mode=0,
        turn=0,
        forward=1,
        stop=0,
        used_wide=False,
        has_c=False,
        seed=0,
        world_seed=0,
        episode_id=f"s{store}r{row}",
        vis=None if vis is None else np.asarray(vis, np.float32),
        vis_ignore=None if vis_ignore is None else np.asarray(vis_ignore, np.float32),
        a_only=True,
        feat_store=int(store),
        target_log_dist=log_d,
        target_dist_mask=mask,
    )


class _Stub(torch.nn.Module):
    def __init__(self, vis_p, dist):
        super().__init__()
        vis_p = np.asarray(vis_p, np.float64)
        dist = np.asarray(dist, np.float64)
        logit = np.log(vis_p / (1.0 - vis_p)).astype(np.float32)
        self.vis_logit = torch.tensor(logit, dtype=torch.float32)
        self.log_dist = torch.tensor(np.log(dist).astype(np.float32), dtype=torch.float32)
        self.at = 0

    def perception(self, tokens):
        b = int(tokens.shape[0])
        sl = slice(self.at, self.at + b)
        self.at += b
        return {
            "vis_logit": self.vis_logit[sl].to(device=tokens.device),
            "target_log_dist": self.log_dist[sl].to(device=tokens.device),
        }


class HardNegRuleTest(unittest.TestCase):
    def test_all_four_conditions(self):
        self.assertTrue(bool(hard_negative_mask(0.7, 8.0, 0.0, 0.0, 0.7, 8.0)))
        self.assertFalse(bool(hard_negative_mask(0.699, 8.0, 0.0, 0.0, 0.7, 8.0)))
        self.assertFalse(bool(hard_negative_mask(0.7, 8.01, 0.0, 0.0, 0.7, 8.0)))
        self.assertFalse(bool(hard_negative_mask(0.9, 1.0, 1.0, 0.0, 0.7, 8.0)))
        self.assertFalse(bool(hard_negative_mask(0.9, 1.0, 0.5, 0.0, 0.7, 8.0)))
        self.assertFalse(bool(hard_negative_mask(0.9, 1.0, 0.0, 1.0, 0.7, 8.0)))
        mask = hard_negative_mask(
            [0.9, 0.9, 0.9, 0.2],
            [4.0, 9.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            0.7,
            8.0,
        )
        self.assertEqual(mask.tolist(), [True, False, False, False])


class HardNegMineTest(unittest.TestCase):
    def _rows(self):
        z = np.zeros((2, 1, 2), np.float32)
        ig = np.zeros((1, 2), np.float32)
        present_patch = z.copy()
        present_patch[0, 0, 0] = 1.0
        present_patch[1, 0, 1] = 1.0
        ignore_patch = np.array([[0.0, 1.0]], np.float32)
        rows = [
            _sample(0, 0, [0, 0], [0, 0], z, ig, target_index=0, target_dist=2.0),
            _sample(0, 1, [1, 0], [0, 1], z, ig, target_index=0, target_dist=5.0),
            _sample(1, 0, [0, 0], [0, 0], z, ig, target_index=1, target_dist=3.0),
            _sample(1, 1, None, None, present_patch, ignore_patch, target_index=0, target_dist=None),
        ]
        vis_p = [
            [0.9, 0.9],
            [0.95, 0.99],
            [0.2, 0.85],
            [0.2, 0.95],
        ]
        dist = [
            [4.0, 20.0],
            [1.0, 1.0],
            [1.0, 4.0],
            [1.0, 2.0],
        ]
        tokens = [np.zeros((2, 1, 2, 4), np.float32), np.zeros((2, 1, 2, 4), np.float32)]
        return rows, vis_p, dist, tokens

    def test_mine_keeps_only_false_near_visible(self):
        rows, vis_p, dist, tokens = self._rows()
        model = _Stub(vis_p, dist)
        found = mine_hard_negatives(model, rows, tokens, "cpu", ["sheep", "pig"], 0.7, 8.0)
        got = {(rec["feat_store"], rec["row"], rec["class_name"]) for rec in found}
        self.assertEqual(got, {(0, 0, "sheep"), (1, 0, "pig"), (1, 1, "pig")})
        self.assertEqual(hard_neg_counts(found, ["sheep", "pig"]), {"sheep": 1, "pig": 2})
        by_key = {(rec["feat_store"], rec["row"], rec["class_index"]): rec for rec in found}
        self.assertAlmostEqual(by_key[(0, 0, 0)]["vis_p"], 0.9, places=4)
        self.assertAlmostEqual(by_key[(0, 0, 0)]["pred_dist"], 4.0, places=3)
        self.assertAlmostEqual(by_key[(1, 0, 1)]["pred_dist"], 4.0, places=3)
        weights = hard_neg_weights(rows, {(r["feat_store"], r["row"], r["class_index"]) for r in found}, 3.0, 2)
        self.assertEqual(weights[0].tolist(), [3.0, 1.0])
        self.assertEqual(weights[1].tolist(), [1.0, 1.0])
        self.assertEqual(weights[2].tolist(), [1.0, 3.0])
        self.assertEqual(weights[3].tolist(), [1.0, 3.0])
        self.assertTrue(np.all(hard_neg_weights(rows, set(), 3.0, 2) == 1.0))

    def test_val_report(self):
        rows, vis_p, dist, tokens = self._rows()
        model = _Stub(vis_p, dist)
        found = mine_hard_negatives(model, rows, tokens, "cpu", ["sheep", "pig"], 0.7, 8.0)
        model.at = 0
        report = hard_neg_epoch_report(model, rows, tokens, "cpu", ["sheep", "pig"], found, report_tau=0.7)
        self.assertAlmostEqual(report["fpr"]["sheep"]["value"], 0.5, places=4)
        self.assertEqual(report["fpr"]["sheep"]["n"], 2)
        self.assertAlmostEqual(report["fpr"]["pig"]["value"], 1.0, places=4)
        self.assertEqual(report["fpr"]["pig"]["n"], 3)
        self.assertAlmostEqual(report["hard_fpr"]["sheep"]["value"], 1.0, places=4)
        self.assertEqual(report["hard_fpr"]["sheep"]["n"], 1)
        self.assertAlmostEqual(report["hard_fpr"]["pig"]["value"], 1.0, places=4)
        self.assertEqual(report["hard_fpr"]["pig"]["n"], 2)
        self.assertAlmostEqual(report["recall"]["sheep"]["value"], 0.5, places=4)
        self.assertEqual(report["recall"]["sheep"]["n"], 2)
        self.assertIsNone(report["recall"]["pig"]["value"])
        self.assertEqual(report["recall"]["pig"]["n"], 0)
        self.assertAlmostEqual(report["recall_all"]["value"], 0.5, places=4)
        self.assertEqual(report["recall_all"]["n"], 2)
        self.assertEqual(report["dist_le_3_median"]["n"], 2)
        self.assertAlmostEqual(report["dist_le_3_median"]["median"], 1.5, places=3)

    def test_classes_weights_only_named_class(self):
        z = np.zeros((2, 1, 1), np.float32)
        ig = np.zeros((1, 1), np.float32)
        rows = [
            _sample(0, 0, [0, 0], [0, 0], z, ig),
            _sample(0, 1, [0, 0], [0, 0], z, ig),
            _sample(1, 0, [0, 0], [0, 0], z, ig),
        ]
        found = [
            {"feat_store": 0, "row": 0, "class_index": 0, "class_name": "sheep"},
            {"feat_store": 0, "row": 1, "class_index": 1, "class_name": "cow"},
            {"feat_store": 1, "row": 0, "class_index": 0, "class_name": "pig"},
            {"feat_store": 1, "row": 0, "class_index": 1, "class_name": "cow"},
        ]
        cow = hard_neg_weights(rows, hard_neg_weight_keys(found, ["cow"]), 3.0, 2)
        self.assertEqual(cow[0].tolist(), [1.0, 1.0])
        self.assertEqual(cow[1].tolist(), [1.0, 3.0])
        self.assertEqual(cow[2].tolist(), [1.0, 3.0])
        self.assertEqual(int(np.sum(cow != 1.0)), 2)
        every = hard_neg_weight_keys(found, None)
        self.assertEqual(
            every,
            {(int(rec["feat_store"]), int(rec["row"]), int(rec["class_index"])) for rec in found},
        )
        all_w = hard_neg_weights(rows, every, 3.0, 2)
        self.assertEqual(all_w[0].tolist(), [3.0, 1.0])
        self.assertEqual(all_w[2].tolist(), [3.0, 3.0])
        self.assertIsNone(hard_neg_of({})["classes"])
        self.assertIsNone(hard_neg_of({"c2nav": {"hard_neg": {"classes": None}}})["classes"])
        self.assertEqual(hard_neg_of({"c2nav": {"hard_neg": {"classes": ["cow"]}}})["classes"], ["cow"])
        self.assertEqual(hard_neg_of({"c2nav": {"hard_neg": {"classes": []}}})["classes"], [])


class HardNegWeightTest(unittest.TestCase):
    def _run(self, vis_logit, present_logit, y_vis, y_present, hard):
        k = len(vis_logit)
        out = {
            "present_logit": torch.tensor(present_logit, dtype=torch.float32).view(1, k, 1, 1),
            "log_dist": torch.zeros(1, k, 1, 1),
            "biome": torch.zeros(1, 3),
            "time": torch.zeros(1, 3),
            "weather": torch.zeros(1, 3),
            "brightness": torch.zeros(1, 3),
            "vis_logit": torch.tensor(vis_logit, dtype=torch.float32).view(1, k),
        }
        batch = {
            "present": torch.tensor(y_present, dtype=torch.float32).view(1, k, 1, 1),
            "log_dist": torch.zeros(1, k, 1, 1),
            "ignore": torch.zeros(1, 1, 1),
            "biome": torch.zeros(1, dtype=torch.long),
            "time": torch.zeros(1, dtype=torch.long),
            "weather": torch.zeros(1, dtype=torch.long),
            "brightness": torch.zeros(1, dtype=torch.long),
            "vis": torch.tensor(y_vis, dtype=torch.float32).view(1, k),
            "vis_ignore": torch.zeros(1, k),
            "has_c": torch.zeros(1),
            "used_wide": torch.zeros(1),
            "stop": torch.zeros(1),
            "a_only": torch.ones(1),
            "target_index": torch.zeros(1, dtype=torch.long),
        }
        if hard is not None:
            batch["hard_neg_w"] = torch.tensor(hard, dtype=torch.float32).view(1, k)
        weights = {"a": 1.0, "mode": 1.0, "wp": 1.0, "stop": 1.0}
        return nav_loss(
            out, batch, weights,
            torch.ones(k), torch.ones(4), torch.ones(10), torch.ones(4), torch.tensor(1.0), 0.0,
            skip_c=True,
        )

    def test_weight_scales_only_the_marked_class(self):
        base_loss, base = self._run([1.5], [-0.4], [0.0], [0.0], None)
        wide_loss, wide = self._run([1.5], [-0.4], [0.0], [0.0], [3.0])
        self.assertAlmostEqual(wide["present"], 3.0 * base["present"], places=5)
        self.assertAlmostEqual(wide["vis"], 3.0 * base["vis"], places=5)
        self.assertEqual(wide["biome"], base["biome"])
        self.assertEqual(wide["dist_bin"], base["dist_bin"])
        self.assertEqual(wide["target_dist"], base["target_dist"])
        again_loss, again_parts = self._run([1.5], [-0.4], [0.0], [0.0], None)
        self.assertEqual(again_parts["present"], base["present"])
        self.assertEqual(again_parts["vis"], base["vis"])
        self.assertEqual(float(again_loss), float(base_loss))
        _, mixed = self._run([1.5, -0.2], [-0.4, 0.3], [0.0, 0.0], [0.0, 0.0], [3.0, 1.0])
        vis_logit = torch.tensor([1.5, -0.2])
        present_logit = torch.tensor([-0.4, 0.3])
        bce_v = torch.nn.functional.binary_cross_entropy_with_logits(vis_logit, torch.zeros(2), reduction="none")
        bce_p = torch.nn.functional.binary_cross_entropy_with_logits(present_logit, torch.zeros(2), reduction="none")
        self.assertAlmostEqual(mixed["vis"], float((3.0 * bce_v[0] + bce_v[1]) / 2.0), places=5)
        self.assertAlmostEqual(mixed["present"], float((3.0 * bce_p[0] + bce_p[1]) / 2.0), places=5)

    def test_disabled_config_leaves_loss_unweighted(self):
        plain = hard_neg_of({})
        self.assertFalse(plain["enable"])
        self.assertEqual(plain["vis_tau"], 0.7)
        self.assertEqual(plain["max_pred_dist"], 8.0)
        self.assertEqual(plain["weight"], 3.0)
        zeros = hard_neg_of({"c2nav": {"hard_neg": {"enable": False, "vis_tau": 0, "max_pred_dist": 0, "weight": 0}}})
        self.assertFalse(zeros["enable"])
        self.assertEqual(zeros["vis_tau"], 0.0)
        self.assertEqual(zeros["max_pred_dist"], 0.0)
        self.assertEqual(zeros["weight"], 0.0)
        turned_off = hard_neg_of({"c2nav": {"train_a_only": True}})
        self.assertFalse(turned_off["enable"])


if __name__ == "__main__":
    unittest.main()
