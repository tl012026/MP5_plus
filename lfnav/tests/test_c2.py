"""C2+GT dataset split, wide_weight, and (when torch is present) train/infer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lfnav.config import load_config
from lfnav.data.demo_format import write_episode
from lfnav.policy.c2 import BehaviorCloning
from lfnav.policy.c2_data import FeatureSpec, load_c2_split, pad_chunk
from lfnav.types import PolicyInput

ROOT = Path(__file__).resolve().parents[2]

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _episode(folder_root, task, difficulty, seed, chunks, used_wide=False):
    record = {"task_name": task, "seed": seed, "difficulty": difficulty, "success": True}
    rows = []
    for i, chunk in enumerate(chunks):
        rows.append({
            "decision": i,
            "env_step": i * 4,
            "match": {"status": "not_found", "target_direction": "left", "target_distance": "mid"},
            "scene": {"objects": {"pig": {"present": 0.9, "direction": {"left": 1.0}, "distance": {"mid": 1.0}}}},
            "memory": {"sector8": [1.0, 0, 0, 0, 0, 0, 0, 0], "recent_actions": ["forward"]},
            "action_chunk": list(chunk),
            "used_wide": used_wide,
        })
    return write_episode(folder_root, "success", task, difficulty, seed, record, rows)


class DatasetSplitTest(unittest.TestCase):
    def test_seed_split_does_not_leak_episodes(self):
        cfg = load_config()
        cfg["c2"] = dict(cfg.get("c2") or {})
        cfg["c2"]["wide_weight"] = 0.5
        with tempfile.TemporaryDirectory() as tmp:
            for seed in (1, 2, 3, 4):
                _episode(tmp, "1-4", "easy", seed, [["forward"], ["turn_left_15", "forward"]])
            _episode(tmp, "1-3", "easy", 1, [["forward"]], used_wide=True)
            split = load_c2_split(tmp, cfg, val_frac=0.25, rng_seed=0)
            train_ids = {row.episode_id for row in split.train}
            val_ids = {row.episode_id for row in split.val}
            self.assertTrue(split.train)
            self.assertTrue(split.val)
            self.assertFalse(train_ids & val_ids)
            train_seeds = {row.seed for row in split.train}
            val_seeds = {row.seed for row in split.val}
            self.assertFalse(train_seeds & val_seeds)
            wide = [row for row in split.train + split.val if row.used_wide]
            plain = [row for row in split.train + split.val if not row.used_wide]
            self.assertTrue(wide)
            self.assertAlmostEqual(wide[0].weight, 0.5)
            self.assertAlmostEqual(plain[0].weight, 1.0)
            self.assertGreater(split.used_wide_frac, 0.0)
            spec = FeatureSpec.from_cfg(cfg)
            for row in split.train + split.val:
                self.assertEqual(row.x.shape[0], spec.dim)
                self.assertEqual(len(row.y), spec.horizon)

    def test_short_chunk_pads_with_noop(self):
        spec = FeatureSpec.from_cfg(load_config())
        ids = pad_chunk(["forward"], spec)
        self.assertEqual(len(ids), spec.horizon)
        self.assertEqual(spec.actions[ids[0]], "forward")
        self.assertEqual(spec.actions[ids[1]], "noop")


class InferTest(unittest.TestCase):
    def test_actions_are_in_the_table(self):
        cfg = load_config()
        policy = BehaviorCloning(cfg)
        out = policy.act(PolicyInput(
            task={"target": "pig", "conditions": {}},
            match={"status": "not_found", "target_direction": "left", "target_distance": "mid"},
            scene={"objects": {}},
            memory={"sector8": [0.0] * 8, "recent_actions": [], "env_steps": 0},
        ))
        names = set(policy.spec.actions if policy.model is not None else ["forward", "turn_left_15", "turn_right_15", "turn_left_45", "turn_right_45", "look_reset"])
        self.assertEqual(len(out.action_chunk), 4)
        for name in out.action_chunk:
            self.assertIn(name, names)


@unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
class TorchTrainTest(unittest.TestCase):
    def test_loss_drops_on_tiny_data(self):
        from lfnav.policy.train_c2 import dry_run_torch

        dry_run_torch()

    def test_wide_weight_scales_loss(self):
        import torch

        from lfnav.policy.c2 import ChunkMLP
        from lfnav.policy.c2_data import FeatureSpec
        from lfnav.policy.train_c2 import chunk_loss

        spec = FeatureSpec.from_cfg(load_config())
        model = ChunkMLP(spec.dim, len(spec.actions), spec.horizon, 16)
        x = torch.zeros(2, spec.dim)
        y = torch.zeros(2, spec.horizon, dtype=torch.long)
        cw = torch.ones(len(spec.actions))
        w_full = torch.tensor([1.0, 1.0])
        w_half = torch.tensor([1.0, 0.5])
        full = float(chunk_loss(model, x, y, w_full, cw).item())
        half = float(chunk_loss(model, x, y, w_half, cw).item())
        self.assertLess(half, full)
        self.assertGreater(half, 0.0)
