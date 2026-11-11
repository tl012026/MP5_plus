"""C2+GT dataset split, wide_weight, and (when torch is present) train/infer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lfnav.config import load_config
from lfnav.data.demo_format import write_episode
from lfnav.policy.c2 import BehaviorCloning
from lfnav.policy.c2_data import FeatureSpec, load_c2_split, pad_chunk, select_training_episodes
from lfnav.types import PolicyInput

ROOT = Path(__file__).resolve().parents[2]

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _episode(folder_root, task, difficulty, seed, chunks, used_wide=False, **record_extra):
    record = {"task_name": task, "seed": seed, "difficulty": difficulty, "success": True}
    record.update(record_extra)
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

    def test_filter_main_legacy_and_exclude(self):
        from lfnav.data.demo_format import read_episode

        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            main = Path(tmp) / "demos_main"
            old = Path(tmp) / "demos_dynamic"
            _episode(main, "1-4", "easy", 1, [["forward"]], config_name="main", peaceful_ok=True, hostile_seen=0)
            _episode(old, "1-4", "easy", 1, [["turn_left_15"]], config_name="demo_dynamic")
            _episode(old, "2-3", "easy", 2, [["forward"]], config_name="demo_dynamic")
            _episode(main, "4-3", "easy", 3, [["forward"]], config_name="main", peaceful_ok=True, hostile_seen=0)
            _episode(main, "1-3", "easy", 4, [["forward"]], config_name="main", peaceful_ok=True, hostile_seen=2)
            _episode(main, "2-1", "easy", 5, [["forward"]], config_name="main", hostile_seen=0)
            folders = [p.parent for p in Path(tmp).glob("**/record.json")]
            kept, stats = select_training_episodes(folders, cfg)
            recs = [read_episode(f)["record"] for f in kept]
            keys = {(r["task_name"], int(r["seed"]), r.get("config_name")) for r in recs}
            self.assertIn(("1-4", 1, "main"), keys)
            self.assertIn(("2-3", 2, "demo_dynamic"), keys)
            self.assertIn(("2-1", 5, "main"), keys)
            self.assertFalse(any(r["task_name"] == "4-3" for r in recs))
            self.assertFalse(any(r["task_name"] == "1-3" for r in recs))
            self.assertGreaterEqual(stats["duplicate"], 1)
            self.assertGreaterEqual(stats["exclude_tasks"], 1)
            self.assertGreaterEqual(stats["main_hostile"], 1)

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
