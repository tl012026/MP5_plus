"""Round-3 C2Nav checks: vis labels, FP noise, ego sectors, weights, resume, config 0."""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfnav.policy.c2nav import (
    a_extra_per_epoch_of,
    far_class_ignore,
    forward_backward_weight_cap_of,
    obs_from_present_head,
    spec_from_cfg,
    stop_max_dist_of,
    torch_threads_of,
    vis_labels,
)
from lfnav.policy.state_tracker import TargetObs, TrackerNoise, apply_c_obs_noise, rotate_sectors
from lfnav.policy.train_c2nav import class_weights, resume_start_epoch, save_checkpoint

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

ROOT = Path(__file__).resolve().parents[2]


class VisLabelTest(unittest.TestCase):
    def test_vis_positive_and_hud_ignore(self):
        present = np.zeros((2, 4, 4), np.float32)
        ignore = np.zeros((4, 4), np.float32)
        present[0, 1, 1] = 1.0
        vis, vis_ig = vis_labels(present, ignore)
        self.assertEqual(float(vis[0]), 1.0)
        self.assertEqual(float(vis[1]), 0.0)
        self.assertEqual(float(vis_ig[0]), 0.0)
        ignore[1, 1] = 1.0
        vis2, vis_ig2 = vis_labels(present, ignore)
        self.assertEqual(float(vis2[0]), 0.0)
        self.assertEqual(float(vis_ig2[0]), 0.0)

    def test_class_ignore_skips_loss(self):
        present = np.zeros((2, 2, 2), np.float32)
        present[1, 0, 0] = 1.0
        ignore = np.zeros((2, 2), np.float32)
        vis, vis_ig = vis_labels(present, ignore, class_ignore=np.asarray([0.0, 1.0], np.float32))
        self.assertEqual(float(vis_ig[1]), 1.0)
        self.assertEqual(float(vis[1]), 0.0)

    def test_far_radar_sets_class_ignore(self):
        objects = ["pig", "cow"]
        step = {
            "wide_rays": [
                {"label": "pig", "distance": 20.0, "yaw_offset": 5.0},
                {"label": "cow", "distance": 8.0, "yaw_offset": 0.0},
            ]
        }
        ig = far_class_ignore(step, objects, hfov_deg=90.0)
        self.assertEqual(float(ig[0]), 1.0)
        self.assertEqual(float(ig[1]), 0.0)
        step_out = {"wide_rays": [{"label": "pig", "distance": 20.0, "yaw_offset": 80.0}]}
        ig2 = far_class_ignore(step_out, objects, hfov_deg=90.0)
        self.assertEqual(float(ig2[0]), 0.0)


class PersistentFpTest(unittest.TestCase):
    def test_fp_only_on_invisible(self):
        vis = TargetObs(visible=True, bearing_deg=10.0, log_distance=1.0)
        hid = TargetObs(visible=False)
        noise = TrackerNoise(p_fp_start=1.0, fp_duration_min=3, fp_duration_max=3, hfov_deg=90.0, log_dist_bank=[1.0])
        steps = [{"pos": [0.0, 64.0, 0.0], "yaw": 0.0} for _ in range(4)]
        out = apply_c_obs_noise([vis, vis, hid, hid], steps, noise, random.Random(0))
        self.assertTrue(out[0].visible)
        self.assertAlmostEqual(float(out[0].bearing_deg), 10.0, places=3)
        self.assertTrue(out[2].visible)
        self.assertTrue(out[3].visible)

    def test_fp_start_rate_and_duration(self):
        noise = TrackerNoise(p_fp_start=0.03, fp_duration_min=2, fp_duration_max=8, hfov_deg=90.0, log_dist_bank=[1.0])
        n = 4000
        starts = 0
        rng = random.Random(1)
        for _ in range(n):
            hid = TargetObs(visible=False)
            step = {"pos": [0.0, 64.0, 0.0], "yaw": 0.0}
            out = apply_c_obs_noise([hid], [step], noise, rng)
            if out[0].visible:
                starts += 1
        rate = starts / float(n)
        self.assertGreater(rate, 0.015)
        self.assertLess(rate, 0.05)

        class _Rng(random.Random):
            def __init__(self, seed):
                super().__init__(seed)
                self.drawn = []

            def randint(self, a, b):
                val = super().randint(a, b)
                self.drawn.append(val)
                return val

        rng_d = _Rng(2)
        noise1 = TrackerNoise(p_fp_start=1.0, fp_duration_min=2, fp_duration_max=8, hfov_deg=90.0, log_dist_bank=[1.0])
        hidden = [TargetObs(visible=False) for _ in range(12)]
        steps = [{"pos": [0.0, 64.0, 0.0], "yaw": 0.0} for _ in range(12)]
        out = apply_c_obs_noise(hidden, steps, noise1, rng_d)
        self.assertTrue(rng_d.drawn)
        self.assertTrue(all(2 <= d <= 8 for d in rng_d.drawn))
        run = int(rng_d.drawn[0])
        self.assertTrue(all(obs.visible for obs in out[:run]))

    def test_fp_bearing_follows_turn(self):
        noise = TrackerNoise(
            p_fp_start=1.0, fp_duration_min=6, fp_duration_max=6,
            hfov_deg=90.0, log_dist_bank=[math.log(8.0)],
        )
        hidden = [TargetObs(visible=False) for _ in range(4)]
        steps = [
            {"pos": [0.0, 64.0, 0.0], "yaw": 0.0},
            {"pos": [0.0, 64.0, 0.0], "yaw": 0.0},
            {"pos": [0.0, 64.0, 0.0], "yaw": 90.0},
            {"pos": [0.0, 64.0, 0.0], "yaw": 90.0},
        ]
        out = apply_c_obs_noise(hidden, steps, noise, random.Random(0))
        self.assertTrue(out[0].visible)
        self.assertTrue(out[2].visible)
        delta = (out[2].bearing_deg - out[0].bearing_deg + 180.0) % 360.0 - 180.0
        self.assertAlmostEqual(delta, -90.0, delta=2.0)


class EgoSectorTest(unittest.TestCase):
    def test_rotate_90_moves_heading_into_left_bin(self):
        raw = [10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        rotated = rotate_sectors(raw, 90.0)
        self.assertEqual(rotated[0], raw[2])
        self.assertEqual(rotated[6], raw[0])
        back = rotate_sectors(rotated, -90.0)
        self.assertEqual(back, raw)


class ClassWeightCapTest(unittest.TestCase):
    def test_cap_three_before_mean_norm(self):
        values = [0] * 999 + [1]
        w, counts = class_weights(values, 2, cap=3.0)
        total = float(counts.sum())
        raw = np.sqrt(total / (2.0 * counts))
        self.assertGreater(float(raw[1]), 3.0)
        capped = np.minimum(raw, 3.0)
        self.assertAlmostEqual(float(capped[1]), 3.0)
        mean = float(capped.mean())
        np.testing.assert_allclose(w, capped / mean, atol=1e-5)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)

    def test_backward_cap_ten_others_three(self):
        self.assertEqual(float(forward_backward_weight_cap_of({})), 10.0)
        self.assertEqual(float(forward_backward_weight_cap_of({"c2nav": {}})), 10.0)
        self.assertEqual(float(forward_backward_weight_cap_of({"c2nav": {"forward_backward_weight_cap": None}})), 10.0)
        self.assertEqual(float(forward_backward_weight_cap_of({"c2nav": {"forward_backward_weight_cap": 0}})), 0.0)
        values = [0] + [1] * 999
        caps = np.asarray([10.0, 3.0], np.float64)
        w, counts = class_weights(values, 2, cap=caps)
        raw = np.sqrt(float(counts.sum()) / (2.0 * counts))
        self.assertGreater(float(raw[0]), 10.0)
        self.assertLess(float(raw[1]), 3.0)
        capped = np.asarray([10.0, float(raw[1])], np.float64)
        np.testing.assert_allclose(w, capped / capped.mean(), atol=1e-5)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)


class ConfigZeroTest(unittest.TestCase):
    def test_zero_config_is_kept(self):
        spec = spec_from_cfg({"c2nav": {
            "dropout": 0, "vis_tau": 0, "torch_threads": 0, "a_extra_per_epoch": 0, "hidden": 0,
        }})
        self.assertEqual(float(spec["dropout"]), 0.0)
        self.assertEqual(float(spec["vis_tau"]), 0.0)
        self.assertEqual(int(spec["torch_threads"]), 0)
        self.assertEqual(int(spec["a_extra_per_epoch"]), 0)
        self.assertEqual(int(spec["hidden"]), 0)
        self.assertEqual(int(torch_threads_of({"c2nav": {"torch_threads": 0}})), 0)
        self.assertEqual(int(a_extra_per_epoch_of({"c2nav": {"a_extra_per_epoch": 0}})), 0)
        self.assertEqual(float(stop_max_dist_of({"c2nav": {"stop_max_dist": 0}})), 0.0)


class PresentArgmaxTest(unittest.TestCase):
    def test_visible_without_strong_patch_uses_argmax(self):
        present = np.zeros((1, 2, 2), np.float32)
        present[0, 0, 1] = 0.2
        log_d = np.zeros((1, 2, 2), np.float32)
        log_d[0, 0, 1] = 1.5
        yaw = np.asarray([[-10.0, 12.0], [-8.0, 9.0]], np.float32)
        obs = obs_from_present_head(present, log_d, 0, yaw, vis_logit=4.0, vis_tau=0.5, patch_min=0.3)
        self.assertTrue(obs.visible)
        self.assertAlmostEqual(float(obs.bearing_deg), 12.0, places=3)


@unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
class ResumeTest(unittest.TestCase):
    def test_resume_starts_at_next_epoch(self):
        from lfnav.policy.c2nav import C2NavNet, spec_from_cfg as spec_of

        spec = spec_of({"c2nav": {"hidden": 8, "channels": 4, "dim": 8, "grid": [2, 2]}})
        spec["dropout"] = 0.0
        model = C2NavNet(spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "epoch_02.pt"
            save_checkpoint(path, model, spec, 2, {"n": 0})
            self.assertEqual(resume_start_epoch(path), 3)


if __name__ == "__main__":
    unittest.main()
