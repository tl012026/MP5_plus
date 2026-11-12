"""Joint A+C2: feature cache, fake forward/backward, matcher reads A's scene."""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfnav.config import load_config
from lfnav.data.demo_format import FRAME_HW, preprocess_frame, save_decision_frame, write_episode
from lfnav.eval.run import make_perception, make_policy, match_distance
from lfnav.matcher import Matcher
from lfnav.perception.encoder import FakeEncoder, pool_patch_tokens, reshape_patch_tokens
from lfnav.perception.convert_features import convert_dir, load_memmaps, rows_float32
from lfnav.perception.extract_features import extract_demos, feature_path_for
from lfnav.policy.c2_data import select_training_episodes
from lfnav.policy.joint import (
    JointAgent,
    assert_checkpoint_flags,
    belief_aligned_distance,
    c2_input_dim,
    infer_policy_uses_pooled,
    joint_policy_flags,
    joint_spec,
)
from lfnav.policy.train_joint import apply_joint_cli, checkpoint_record
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

    def test_infer_preprocess_matches_demo_jpeg(self):
        rng = np.random.RandomState(0)
        raw = rng.randint(0, 256, size=(3, 512, 820), dtype=np.uint8)
        pre = preprocess_frame(raw)
        self.assertEqual(pre.shape, (*FRAME_HW, 3))
        self.assertEqual(pre.dtype, np.uint8)
        encoder = FakeEncoder(dim=4, grid=(16, 26))
        before_pool = encoder(pre)["tokens"]
        self.assertEqual(before_pool.shape[:2], (16, 26))
        after_pool = pool_patch_tokens(before_pool, 2)
        self.assertEqual(after_pool.shape[:2], (8, 13))
        cfg = load_config()
        cfg["dry_run"] = True
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"].update(encoder="fake", fake_dim=4, fake_grid=[16, 26], token_pool=2)
        agent = JointAgent(cfg)
        inferred = agent.features(raw)
        self.assertEqual(inferred["tokens"].shape[:2], (8, 13))
        np.testing.assert_allclose(inferred["tokens"], after_pool, rtol=1e-5, atol=1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_decision_frame(Path(tmp), 0, {"rgb": raw}, cfg)
            from PIL import Image

            loaded = np.asarray(Image.open(path).convert("RGB"))
            np.testing.assert_array_equal(loaded, pre)
            trained = pool_patch_tokens(encoder(loaded)["tokens"], 2)
            np.testing.assert_allclose(inferred["tokens"], trained, rtol=1e-5, atol=1e-5)

    def test_pool_16x26_is_8x13(self):
        grid = np.arange(16 * 26 * 3, dtype=np.float32).reshape(16, 26, 3)
        pooled = pool_patch_tokens(grid, 2)
        self.assertEqual(pooled.shape, (8, 13, 3))
        block = grid[:2, :2].mean(axis=(0, 1))
        np.testing.assert_allclose(pooled[0, 0], block)

    def test_npy_count_matches_filtered_decisions(self):
        cfg = load_config()
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"]["token_pool"] = 1
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

    def test_extract_and_agent_share_pooled_8x13(self):
        cfg = load_config()
        cfg["dry_run"] = True
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"].update(encoder="fake", fake_dim=4, fake_grid=[16, 26], token_pool=2)
        encoder = FakeEncoder(dim=4, grid=(16, 26))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demos_main"
            _episode(root, "1-4", "easy", 1, [["forward"]], config_name="main", hostile_seen=0)
            out = Path(tmp) / "feat"
            extract_demos([root], out, encoder, cfg)
            files = list(out.glob("*.npy"))
            self.assertEqual(len(files), 1)
            payload = np.load(files[0], allow_pickle=True).item()
            self.assertEqual(payload["tokens"].shape[1:3], (8, 13))
            zeros = np.zeros((8, 8, 3), dtype=np.uint8)
            extract_expected = pool_patch_tokens(encoder(zeros)["tokens"], 2)
            np.testing.assert_allclose(payload["tokens"][0].astype(np.float32), extract_expected)
            agent = JointAgent(cfg)
            encoded = agent.features(zeros)
            infer_expected = pool_patch_tokens(encoder(preprocess_frame(zeros))["tokens"], 2)
            self.assertEqual(encoded["tokens"].shape[:2], (8, 13))
            np.testing.assert_allclose(encoded["tokens"], infer_expected)
            np.testing.assert_allclose(agent._tokens, infer_expected)
            convert_dir(out, pool=2)
            tokens_mm, _, _ = load_memmaps(out)
            self.assertEqual(tuple(tokens_mm.shape[1:3]), (8, 13))
            np.testing.assert_allclose(
                np.asarray(tokens_mm[0], dtype=np.float32),
                extract_expected,
                rtol=1e-3,
                atol=1e-3,
            )

    def test_memmap_batch_matches_full_read(self):
        rng = np.random.RandomState(0)
        with tempfile.TemporaryDirectory() as tmp:
            feat = Path(tmp)
            n, h, w, d = 10, 8, 13, 4
            full = rng.randn(n, h, w, d).astype(np.float16)
            pooled = full.astype(np.float32).mean(axis=(1, 2)).astype(np.float16)
            np.save(feat / "ep__easy__1.npy", {
                "tokens": full,
                "pooled": pooled,
                "decisions": np.arange(n, dtype=np.int32),
                "episode": "ep__easy__1",
            })
            index = convert_dir(feat, pool=2)
            self.assertEqual(index["grid"], [8, 13])
            tokens, pooled_mm, packed = load_memmaps(feat)
            self.assertEqual(tuple(tokens.shape), (n, h, w, d))
            sl = np.array([0, 3, 9, 1])
            np.testing.assert_allclose(
                rows_float32(tokens, sl),
                np.asarray(full, dtype=np.float32)[sl],
                rtol=1e-3,
                atol=1e-3,
            )
            np.testing.assert_allclose(
                rows_float32(pooled_mm, sl),
                np.asarray(pooled, dtype=np.float32)[sl],
                rtol=1e-3,
                atol=1e-3,
            )
            self.assertEqual(packed["rows"][3]["decision"], 3)
            self.assertEqual(packed["rows"][3]["folder"], "ep/easy/1")
            self.assertEqual(packed["aux"], "steps.jsonl")

    def test_memmap_rss_does_not_grow_with_n(self):
        status = Path("/proc/self/status")
        if not status.is_file():
            self.skipTest("no /proc/self/status")
        h, w, d = 8, 13, 384
        batch = np.arange(8)

        def rss_bytes() -> int:
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
            self.fail("VmRSS missing")

        def fill(path: Path, n: int) -> None:
            mm = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float16, shape=(n, h, w, d))
            step = 16
            for i in range(0, n, step):
                j = min(i + step, n)
                mm[i:j] = np.float16((i % 9) / 9.0)
            mm.flush()
            del mm

        with tempfile.TemporaryDirectory() as tmp:
            small_n, big_n = 32, 2500
            small = Path(tmp) / "small.npy"
            big = Path(tmp) / "big.npy"
            fill(small, small_n)
            fill(big, big_n)
            gc.collect()
            before = rss_bytes()
            small_mm = np.load(small, mmap_mode="r")
            _ = rows_float32(small_mm, batch)
            mid = rss_bytes()
            big_mm = np.load(big, mmap_mode="r")
            got = rows_float32(big_mm, batch)
            after = rss_bytes()
            self.assertEqual(got.shape, (8, h, w, d))
            full_big = big_n * h * w * d * 2
            self.assertLess(after - before, full_big // 4)
            self.assertLess(after - mid, 32 * 1024 * 1024)


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


class PolicyHeadInterfaceTest(unittest.TestCase):
    def test_c2_input_dim_omits_pooled(self):
        cfg = load_config()
        spec = joint_spec(cfg)
        flags = joint_policy_flags(cfg)
        self.assertFalse(flags["policy_uses_pooled"])
        self.assertTrue(flags["detach_scene"])
        dim = 384
        self.assertEqual(c2_input_dim(spec, dim, False), spec.scene_dim + spec.aux_dim)
        self.assertEqual(c2_input_dim(spec, dim, True), spec.scene_dim + spec.aux_dim + dim)

    def test_checkpoint_flag_mismatch_raises(self):
        cfg = {"joint": {"policy_uses_pooled": False, "detach_scene": True}}
        assert_checkpoint_flags({"policy_uses_pooled": False, "detach_scene": True}, cfg)
        with self.assertRaises(RuntimeError):
            assert_checkpoint_flags({"policy_uses_pooled": True, "detach_scene": True}, cfg)
        with self.assertRaises(RuntimeError):
            assert_checkpoint_flags({"policy_uses_pooled": False, "detach_scene": False}, cfg)

    def test_missing_policy_uses_pooled_inferred_from_p1(self):
        spec = joint_spec(load_config())
        dim = 8
        v2 = {"joint": {"policy_uses_pooled": False, "detach_scene": True}}
        v1 = {"joint": {"policy_uses_pooled": True, "detach_scene": False}}
        no_pool = {
            "dim": dim,
            "model": {"p1.weight": np.zeros((16, c2_input_dim(spec, dim, False)), dtype=np.float32)},
        }
        with_pool = {
            "dim": dim,
            "model": {"p1.weight": np.zeros((16, c2_input_dim(spec, dim, True)), dtype=np.float32)},
        }
        self.assertFalse(infer_policy_uses_pooled(no_pool, spec))
        self.assertTrue(infer_policy_uses_pooled(with_pool, spec))
        with self.assertLogs("lfnav.policy.joint", level="WARNING") as logs:
            assert_checkpoint_flags(no_pool, v2, spec=spec)
        self.assertTrue(any("inferred" in row for row in logs.output))
        self.assertTrue(any("detach_scene" in row for row in logs.output))
        with self.assertLogs("lfnav.policy.joint", level="WARNING"):
            assert_checkpoint_flags(with_pool, v1, spec=spec)
        with self.assertLogs("lfnav.policy.joint", level="WARNING"):
            with self.assertRaises(RuntimeError):
                assert_checkpoint_flags(with_pool, v2, spec=spec)

    def test_cli_flags_reach_joint_policy_flags(self):
        class Args:
            policy_uses_pooled = True
            detach_scene = False

        cfg = apply_joint_cli({"joint": {}}, Args())
        flags = joint_policy_flags(cfg)
        self.assertTrue(flags["policy_uses_pooled"])
        self.assertFalse(flags["detach_scene"])

    def test_checkpoint_record_save_load_flags(self):
        flags = {"policy_uses_pooled": False, "detach_scene": True}
        rec = checkpoint_record(
            flags, "joint_v2", model={}, recent_k=4, memory_size=15, token_pool=2,
        )
        self.assertIn("policy_uses_pooled", rec)
        self.assertIn("detach_scene", rec)
        self.assertIn("version", rec)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.npy"
            np.save(path, rec)
            loaded = np.load(path, allow_pickle=True).item()
            self.assertEqual(loaded["policy_uses_pooled"], False)
            self.assertEqual(loaded["detach_scene"], True)
            self.assertEqual(loaded["version"], "joint_v2")


@unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
class JointTrainTest(unittest.TestCase):
    def test_forward_backward(self):
        from lfnav.policy.train_joint import dry_run_torch

        after = dry_run_torch()
        self.assertLess(after, 1e9)

    def test_c2_loss_does_not_train_a_when_detached(self):
        import torch
        from torch.nn import functional as F

        from lfnav.policy.joint import JointNet

        cfg = load_config()
        spec = joint_spec(cfg)
        dim = 8
        model = JointNet(dim, spec, hidden=16, horizon=spec.horizon, policy_uses_pooled=False, detach_scene=True)
        self.assertEqual(model.c2_in_dim, spec.scene_dim + spec.aux_dim)
        tokens = torch.randn(4, 4, 6, dim, requires_grad=True)
        aux = torch.randn(4, spec.aux_dim)
        y = torch.zeros(4, spec.horizon, dtype=torch.long)
        out = model(tokens, aux)
        loss = sum(F.cross_entropy(head, y[:, k]) for k, head in enumerate(out["action"]))
        loss.backward()
        a_names = ("conv", "fc", "present", "direction", "distance", "biome", "time", "weather", "brightness")
        for name, param in model.named_parameters():
            stem = name.split(".", 1)[0]
            if stem in a_names:
                self.assertTrue(param.grad is None or float(param.grad.abs().sum()) == 0.0, name)
            elif stem in ("p1", "p2", "heads"):
                self.assertIsNotNone(param.grad)
                self.assertGreater(float(param.grad.abs().sum()), 0.0)

    def test_joint_agent_rejects_mismatched_checkpoint(self):
        import torch
        from lfnav.policy.joint import JointNet

        cfg = load_config()
        cfg["dry_run"] = True
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"].update(encoder="fake", fake_dim=8, fake_grid=[2, 3], policy_uses_pooled=False, detach_scene=True)
        spec = joint_spec(cfg)
        model = JointNet(8, spec, hidden=8, horizon=spec.horizon, policy_uses_pooled=True, detach_scene=True)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "bad.pt"
            torch.save({
                "model": model.state_dict(),
                "policy_uses_pooled": True,
                "detach_scene": True,
                "recent_k": 4,
                "memory_size": 15,
                "token_pool": 1,
            }, ckpt)
            cfg["joint"]["checkpoint"] = str(ckpt)
            with self.assertRaises(RuntimeError):
                JointAgent(cfg)

    def test_train_on_extracted_features(self):
        import torch
        from lfnav.policy.train_joint import load_joint_split, train_loop

        cfg = load_config()
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"]["hidden"] = 16
        cfg["joint"]["token_pool"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demos_main"
            for seed in (1, 2, 3, 4):
                _episode(root, "1-4", "easy", seed, [["forward"], ["forward"]], config_name="main", hostile_seen=0)
            feat = Path(tmp) / "feat"
            extract_demos([root], feat, FakeEncoder(dim=8, grid=(2, 3)), cfg)
            convert_dir(feat, pool=1)
            split = load_joint_split([root], feat, cfg, val_frac=0.25, rng_seed=0)
            self.assertTrue(split.train)
            self.assertTrue(split.val)
            ckpt = train_loop(split, Path(tmp) / "ckpt", cfg, epochs=1, batch=4, lr=1e-2, seed=0, lam=1.0, version="joint_v2")
            self.assertTrue(ckpt.is_file())
            payload = torch.load(ckpt, map_location="cpu")
            self.assertIn("policy_uses_pooled", payload)
            self.assertIn("detach_scene", payload)
            self.assertEqual(payload["version"], "joint_v2")

    def test_torch_save_load_checkpoint_flags(self):
        import torch

        flags = joint_policy_flags(load_config())
        rec = checkpoint_record(
            flags, "joint_v2", model={}, recent_k=4, memory_size=15, token_pool=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "joint_v2.pt"
            torch.save(rec, path)
            loaded = torch.load(path, map_location="cpu")
            self.assertEqual(loaded["policy_uses_pooled"], flags["policy_uses_pooled"])
            self.assertEqual(loaded["detach_scene"], flags["detach_scene"])
            self.assertEqual(loaded["version"], "joint_v2")


if __name__ == "__main__":
    unittest.main()
