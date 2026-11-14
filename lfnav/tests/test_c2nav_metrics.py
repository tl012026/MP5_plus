"""offline_metrics argument order. No MineDojo."""

from __future__ import annotations

import unittest

import numpy as np

from lfnav.config import load_config
from lfnav.policy.c2nav import C2NavNet, spec_from_cfg
from lfnav.policy.c2nav_labels import MODE_NAMES


class OfflineMetricsCallTest(unittest.TestCase):
    def test_three_samples_return_metrics_with_n(self):
        import torch

        from lfnav.policy.train_c2nav import NavSample, offline_metrics

        torch.manual_seed(0)
        spec = spec_from_cfg(load_config())
        spec["dim"] = 8
        spec["grid"] = (4, 6)
        spec["hidden"] = 16
        spec["channels"] = 4
        gh, gw, dim = 4, 6, 8
        k = len(spec["objects"])
        tokens = np.random.randn(3, gh, gw, dim).astype(np.float32)
        rows = []
        for i in range(3):
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
                mode=i % len(MODE_NAMES),
                turn=i % spec["n_turn"],
                forward=i % len(spec["forward_bins"]),
                stop=1 if i == 2 else 0,
                used_wide=False,
                has_c=True,
                seed=i,
                world_seed=i,
                episode_id=f"1-4/easy/{i}",
            ))
        model = C2NavNet(spec)
        bearings = np.zeros((gh, gw), np.float32)
        metrics = offline_metrics(
            model=model,
            rows=rows,
            tokens=tokens,
            device=torch.device("cpu"),
            spec=spec,
            cfg=load_config(),
            bearings=bearings,
            warn_empty_stop=False,
        )
        self.assertEqual(metrics["n"], 3)
        for key in ("mode_acc", "turn_acc", "forward_acc", "stop_f1", "stop_auc", "present_auc", "log_dist_mae"):
            self.assertIn("n", metrics[key], key)
            self.assertIsInstance(metrics[key]["n"], int)
        for name in MODE_NAMES:
            self.assertIn("n", metrics["mode_acc_by_name"][name], name)


if __name__ == "__main__":
    unittest.main()
