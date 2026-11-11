"""Joint A+C2: feature cache, fake forward/backward, matcher reads A's scene."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfnav.config import load_config
from lfnav.data.demo_format import write_episode
from lfnav.eval.run import make_perception, make_policy, match_distance
from lfnav.matcher import Matcher
from lfnav.perception.encoder import FakeEncoder, reshape_patch_tokens
from lfnav.perception.extract_features import extract_demos, feature_path_for
from lfnav.policy.c2_data import select_training_episodes
from lfnav.policy.joint import JointAgent, belief_aligned_distance
from lfnav.types import ObjectBelief, SceneState, TaskSpec

ROOT = Path(__file__).resolve().parents[2]

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _episode(folder_root, task, difficulty, seed, chunks, **record_extra):
    record = {"task_name": task, "seed": seed, "difficulty": difficulty, "success": True}
    record.update(record_extra)
    rows = []
    for i, chunk in enumerate(chunks):
        rows.append({
            "decision": i,
            "env_step": i * 4,
            "match": {"status": "not_found", "target_direction": "left", "target_distance": "mid"},
            "scene": {
                "objects": {"pig": {"present": 0.9, "direction": {"left": 1.0}, "distance": {"mid": 1.0}}},
                "biome": {"plains": 1.0},
                "time": {"day": 1.0},
                "weather": {"clear": 1.0},
                "brightness": {"sufficient": 1.0},
            },
            "memory": {"sector8": [1.0, 0, 0, 0, 0, 0, 0, 0], "recent_actions": ["forward"]},
            "action_chunk": list(chunk),
            "used_wide": False,
        })
    dest = write_episode(folder_root, "success", task, difficulty, seed, record, rows)
    from PIL import Image

    frames = dest / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for i, _ in enumerate(chunks):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(frames / f"{i:05d}.jpg")
    return dest


class FeatureExtractTest(unittest.TestCase):
    def test_reshape_is_16x26_for_224x364(self):
        tokens = np.zeros((16 * 26, 8), dtype=np.float32)
        grid = reshape_patch_tokens(tokens, 224, 364, 14)
        self.assertEqual(grid.shape, (16, 26, 8))

    def test_npy_count_matches_filtered_decisions(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demos_main"
            a = _episode(root, "1-4", "easy", 1, [["forward"], ["turn_left_15"]], config_name="main", hostile_seen=0)
            _episode(root, "1-4", "easy", 2, [["forward"]], config_name="main", hostile_seen=0)
            _episode(root, "1-3", "easy", 3, [["forward"]], config_name="main", hostile_seen=2)
            folders = [p.parent for p in Path(tmp).glob("**/record.json")]
            kept, _ = select_training_episodes(folders, cfg)
            out = Path(tmp) / "feat"
            encoder = FakeEncoder(dim=8, grid=(2, 3))
            manifest = extract_demos([root], out, encoder, cfg)
            files = list(out.glob("*.npy"))
            self.assertEqual(len(files), len(kept))
            n_decisions = 0
            for folder in kept:
                packed = __import__("lfnav.data.demo_format", fromlist=["read_episode"]).read_episode(folder)
                n_decisions += sum(1 for s in packed["steps"] if s.get("action_chunk"))
            total_t = 0
            for path in files:
                payload = np.load(path, allow_pickle=True).item()
                total_t += int(payload["tokens"].shape[0])
                self.assertEqual(payload["tokens"].shape[1:3], (2, 3))
            self.assertEqual(manifest["n_decisions"], n_decisions)
            self.assertEqual(total_t, n_decisions)
            self.assertEqual(feature_path_for(out, {"task_name": "1-4", "difficulty": "easy", "seed": 1}, a).is_file(), True)


class MatcherUsesATest(unittest.TestCase):
    def test_make_policy_joint_is_the_perception(self):
        cfg = load_config()
        cfg["policy_name"] = "joint"
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"]["encoder"] = "fake"
        cfg["dry_run"] = True
        perc = make_perception(cfg)
        pol = make_policy(cfg)
        self.assertIs(perc, pol)
        self.assertEqual(perc.name, "joint")

    def test_b_reads_a_distance_not_gt_rays(self):
        cfg = load_config()
        cfg["policy_name"] = "joint"
        task = TaskSpec(id="1-4", source="", description="", target="pig", conditions={})
        near = SceneState(objects={
            "pig": ObjectBelief(present=1.0, direction={"center": 1.0}, distance={"near": 1.0}),
        })
        far = SceneState(objects={
            "pig": ObjectBelief(present=1.0, direction={"center": 1.0}, distance={"far": 1.0}),
        })

        class _JointPerc:
            name = "joint"

        obs = {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}
        d_near = match_distance(_JointPerc(), obs, cfg, task, near)
        d_far = match_distance(_JointPerc(), obs, cfg, task, far)
        matcher = Matcher(cfg)
        self.assertTrue(matcher.match(task, near, aligned_distance=d_near).stop)
        self.assertFalse(matcher.match(task, far, aligned_distance=d_far).stop)
        self.assertLess(belief_aligned_distance(near, "pig", cfg), 3.0)
        self.assertGreater(belief_aligned_distance(far, "pig", cfg), 3.0)
        # GT ray = 1.0 would stop even when A says far.
        self.assertTrue(matcher.match(task, far, aligned_distance=1.0).stop)

    def test_perceive_returns_scene_state(self):
        cfg = load_config()
        cfg["dry_run"] = True
        cfg["joint"] = {"encoder": "fake", "fake_dim": 8, "fake_grid": [2, 3]}
        agent = JointAgent(cfg)
        scene = agent.perceive({"rgb": np.zeros((8, 8, 3), dtype=np.uint8)})
        self.assertTrue(scene.objects)
        self.assertIn("pig", scene.objects)
        out = agent.act(__import__("lfnav.types", fromlist=["PolicyInput"]).PolicyInput(
            task={"target": "pig", "conditions": {}},
            match={"status": "not_found"},
            scene=scene.to_dict(),
            memory={"sector8": [0.0] * 8, "recent_actions": [], "env_steps": 0},
        ))
        self.assertEqual(len(out.action_chunk), 4)


@unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
class JointTrainTest(unittest.TestCase):
    def test_forward_backward(self):
        from lfnav.policy.train_joint import dry_run_torch

        after = dry_run_torch()
        self.assertLess(after, 1e9)

    def test_train_on_extracted_features(self):
        from lfnav.policy.train_joint import load_joint_split, train_loop

        cfg = load_config()
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"]["hidden"] = 16
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demos_main"
            for seed in (1, 2, 3, 4):
                _episode(root, "1-4", "easy", seed, [["forward"], ["forward"]], config_name="main", hostile_seen=0)
            feat = Path(tmp) / "feat"
            extract_demos([root], feat, FakeEncoder(dim=8, grid=(2, 3)), cfg)
            split = load_joint_split([root], feat, cfg, val_frac=0.25, rng_seed=0)
            self.assertTrue(split.train)
            self.assertTrue(split.val)
            ckpt = train_loop(split, Path(tmp) / "ckpt", cfg, epochs=1, batch=4, lr=1e-2, seed=0, lam=1.0)
            self.assertTrue(ckpt.is_file())


if __name__ == "__main__":
    unittest.main()
