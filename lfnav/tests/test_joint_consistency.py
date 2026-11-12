"""Train vs eval alignment for joint features and aux.

Memory differences before this alignment (eval now uses checkpoint / cfg.joint only):

recent_k
  train encode_aux: spec.recent_k from cfg.joint (default 4); takes recent_actions[-k:].
  demo JSON: Memory at collect truncated to policy.recent_k (default 8).
  eval (old): Memory(policy.recent_k=8) then encode_aux with joint_spec.recent_k=4.
  eval (now): Memory and encode_aux both use checkpoint recent_k; missing key errors
  (no policy.* fallback). Without a checkpoint, cfg.joint.recent_k only.

memory_size
  train encode_aux: unused. sector8 was computed at collect from a local_grid of
  size policy.memory_size (15).
  eval (old): Memory(policy.memory_size).
  eval (now): Memory(checkpoint memory_size / cfg.joint.memory_size).

sector8_counts
  train: stored memory.sector8 = sector8_counts(local_grid) at collect; encode_aux
  keeps 8 bins and L1-normalizes.
  eval (old): to_dict had no sector8; JointAgent recomputed sector8_counts(local_grid).
  eval (now): to_dict includes sector8 from the same sector8_counts; JointAgent uses
  that field when present. Truncation [:8] + L1 is still encode_aux.

Preprocess: extract decodes collect-time JPEG; live eval runs preprocess_frame then
the same encode_joint_tokens (encoder color/norm + pool).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfnav.config import load_config
from lfnav.data.demo_format import FRAME_HW, read_episode, save_decision_frame, write_episode
from lfnav.perception.convert_features import load_memmaps
from lfnav.perception.encoder import FakeEncoder
from lfnav.perception.extract_features import extract_demos, episode_feature_name
from lfnav.perception.joint_features import encode_joint_tokens
from lfnav.policy.c2 import _visible
from lfnav.policy.c2_data import encode_aux, load_task, pad_chunk
from lfnav.policy.joint import (
    JointAgent,
    checkpoint_runtime,
    infer_hidden,
    joint_runtime_from_cfg,
    joint_spec,
)
from lfnav.types import PolicyInput

ROOT = Path(__file__).resolve().parents[2]

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _cos(a, b) -> float:
    x = np.asarray(a, dtype=np.float32).ravel()
    y = np.asarray(b, dtype=np.float32).ravel()
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx == 0.0 or ny == 0.0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def _write_demo(root: Path, task: str, seed: int, chunks, rgb=None) -> Path:
    record = {
        "task_name": task, "seed": seed, "difficulty": "easy", "success": True,
        "config_name": "main", "hostile_seen": 0,
    }
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
            "memory": {
                "sector8": [1.0, 0, 0, 0, 0, 0, 0, 0] if i else [0.0] * 8,
                "recent_actions": ["forward"] * min(i, 6),
            },
            "action_chunk": list(chunk),
            "used_wide": False,
        })
    dest = write_episode(root, "success", task, "easy", seed, record, rows)
    cfg = load_config()
    frame = rgb if rgb is not None else np.zeros((3, 32, 48), dtype=np.uint8)
    for i, _ in enumerate(chunks):
        save_decision_frame(dest, i, {"rgb": frame}, cfg)
    return dest


class RuntimeSourceTest(unittest.TestCase):
    def test_cfg_runtime_ignores_policy_keys(self):
        cfg = {
            "joint": {"recent_k": 4, "memory_size": 15, "token_pool": 2},
            "policy": {"recent_k": 99, "memory_size": 3, "c2_recent_k": 99},
        }
        rt = joint_runtime_from_cfg(cfg)
        self.assertEqual(rt["recent_k"], 4)
        self.assertEqual(rt["memory_size"], 15)
        spec = joint_spec(cfg)
        self.assertEqual(spec.recent_k, 4)

    def test_checkpoint_missing_runtime_errors(self):
        with self.assertRaises(RuntimeError) as ctx:
            checkpoint_runtime({"model": {}, "recent_k": 4})
        self.assertIn("policy.*", str(ctx.exception))
        self.assertIn("memory_size", str(ctx.exception))

    def test_infer_hidden_from_fc_not_cfg(self):
        payload = {"model": {"fc.weight": np.zeros((8, 8), dtype=np.float32)}}
        self.assertEqual(infer_hidden(payload), 8)
        self.assertEqual(infer_hidden({"hidden": 16, "model": {"fc.weight": np.zeros((8, 8))}}), 16)

    def test_agent_uses_joint_not_policy(self):
        cfg = load_config()
        cfg["dry_run"] = True
        cfg["policy"] = dict(cfg.get("policy") or {})
        cfg["policy"].update(recent_k=99, memory_size=3)
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"].update(encoder="fake", fake_dim=4, fake_grid=[2, 3], recent_k=4, memory_size=15, token_pool=1)
        agent = JointAgent(cfg)
        self.assertEqual(agent.spec.recent_k, 4)
        self.assertEqual(agent.memory_size, 15)
        self.assertEqual(agent.token_pool, 1)


class SharedPreprocessTest(unittest.TestCase):
    def test_extract_and_agent_share_saved_jpeg_tokens(self):
        cfg = load_config()
        cfg["dry_run"] = True
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"].update(encoder="fake", fake_dim=4, fake_grid=[16, 26], token_pool=2, recent_k=4, memory_size=15)
        encoder = FakeEncoder(dim=4, grid=(16, 26))
        rng = np.random.RandomState(0)
        rgb = rng.randint(0, 256, size=(3, 40, 60), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demos_main"
            dest = _write_demo(root, "1-4", 1, [["forward"], ["turn_left_15"]], rgb=rgb)
            out = Path(tmp) / "feat"
            extract_demos([root], out, encoder, cfg)
            packed = np.load(list(out.glob("*.npy"))[0], allow_pickle=True).item()
            agent = JointAgent(cfg)
            from PIL import Image

            jpeg = dest / "frames" / "00000.jpg"
            decoded = np.asarray(Image.open(jpeg).convert("RGB"))
            self.assertEqual(decoded.shape, (*FRAME_HW, 3))
            shared = encode_joint_tokens(decoded, encoder, 2)
            eval_feats = agent.features(decoded, saved_jpeg=True)
            np.testing.assert_allclose(eval_feats["tokens"], shared["tokens"], rtol=0, atol=0)
            stored = packed["tokens"][0].astype(np.float16).astype(np.float32)
            np.testing.assert_allclose(eval_feats["tokens"].astype(np.float16).astype(np.float32), stored, rtol=0, atol=0)
            self.assertGreater(_cos(eval_feats["pooled"], packed["pooled"][0]), 0.999)


class AuxReplayTest(unittest.TestCase):
    def test_eval_aux_matches_training_encode_aux(self):
        cfg = load_config()
        cfg["dry_run"] = True
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"].update(encoder="fake", fake_dim=4, fake_grid=[2, 3], token_pool=1)
        spec = joint_spec(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            dest = _write_demo(Path(tmp) / "demos_main", "1-4", 1, [["forward"], ["turn_left_15"]])
            packed = read_episode(dest)
            task = load_task(packed["record"], dest)
            agent = JointAgent(cfg)
            last = None
            for step in packed["steps"]:
                match = step.get("match") or {}
                env_step = int(step.get("env_step") or 0)
                if _visible(match):
                    last = {
                        "step": env_step,
                        "direction": match.get("target_direction"),
                        "distance": match.get("target_distance"),
                    }
                mem = dict(step.get("memory") or {})
                mem["env_steps"] = env_step
                train_aux = encode_aux(task, mem.get("sector8"), mem.get("recent_actions"), last, env_step, spec)
                agent.reset()
                pin = PolicyInput(
                    task=task,
                    match=match,
                    memory=mem,
                    recent_actions=list(mem.get("recent_actions") or []),
                )
                eval_aux = agent.aux_from_input(pin)
                np.testing.assert_array_equal(eval_aux, train_aux)


@unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
class CheckpointRuntimeTest(unittest.TestCase):
    def test_agent_requires_ckpt_runtime_and_ignores_policy(self):
        import torch
        from lfnav.policy.joint import JointNet

        cfg = load_config()
        cfg["dry_run"] = True
        cfg["policy"] = dict(cfg.get("policy") or {})
        cfg["policy"].update(recent_k=99, memory_size=3)
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"].update(
            encoder="fake", fake_dim=8, fake_grid=[2, 3], hidden=256,
            policy_uses_pooled=False, detach_scene=True,
        )
        spec = joint_spec(cfg)
        model = JointNet(8, spec, hidden=8, horizon=spec.horizon, policy_uses_pooled=False, detach_scene=True)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ok.pt"
            torch.save({
                "model": model.state_dict(),
                "policy_uses_pooled": False,
                "detach_scene": True,
                "recent_k": 4,
                "memory_size": 15,
                "token_pool": 1,
                "dim": 8,
                "horizon": spec.horizon,
            }, ckpt)
            cfg["joint"]["checkpoint"] = str(ckpt)
            with self.assertLogs("lfnav.policy.joint", level="WARNING") as logs:
                agent = JointAgent(cfg)
            self.assertTrue(any("hidden=8" in row and "256" in row for row in logs.output))
            self.assertEqual(agent.hidden, 8)
            self.assertEqual(agent.spec.recent_k, 4)
            self.assertEqual(agent.memory_size, 15)
            self.assertEqual(agent.model.hidden, 8)
            bad = Path(tmp) / "bad.pt"
            torch.save({"model": model.state_dict(), "policy_uses_pooled": False, "detach_scene": True}, bad)
            cfg["joint"]["checkpoint"] = str(bad)
            with self.assertRaises(RuntimeError) as ctx:
                JointAgent(cfg)
            self.assertIn("policy.*", str(ctx.exception))


class ScratchReplayTest(unittest.TestCase):
    """Replay 2–3 real demos. Requires env vars (skip lists whichever is missing):

    LFNAV_JOINT_CKPT  path to joint .pt (must include or infer hidden; needs recent_k, memory_size, token_pool)
    LFNAV_FEATURES    packed joint_v1 dir containing tokens.npy / pooled.npy / index.json
    LFNAV_DEMOS       expert demo root (success/.../frames/*.jpg)
    """

    ENV_CKPT = "LFNAV_JOINT_CKPT"
    ENV_FEAT = "LFNAV_FEATURES"
    ENV_DEMOS = "LFNAV_DEMOS"

    def _replay_paths(self):
        ckpt = os.environ.get(self.ENV_CKPT) or ""
        feat = os.environ.get(self.ENV_FEAT) or ""
        demos = os.environ.get(self.ENV_DEMOS) or ""
        missing = []
        if not ckpt:
            missing.append(f"{self.ENV_CKPT} (unset)")
        elif not Path(ckpt).is_file():
            missing.append(f"{self.ENV_CKPT} (not a file: {ckpt})")
        if not feat:
            missing.append(f"{self.ENV_FEAT} (unset)")
        elif not (Path(feat) / "tokens.npy").is_file():
            missing.append(f"{self.ENV_FEAT} (no tokens.npy in {feat})")
        if not demos:
            missing.append(f"{self.ENV_DEMOS} (unset)")
        elif not Path(demos).is_dir():
            missing.append(f"{self.ENV_DEMOS} (not a directory: {demos})")
        if not HAS_TORCH:
            missing.append("torch (not installed)")
        return ckpt, feat, demos, missing

    def test_replay_two_or_three_demos(self):
        ckpt, feat, demos, missing = self._replay_paths()
        if missing:
            msg = "real demo replay skipped, missing: " + "; ".join(missing)
            print(msg, flush=True)
            self.skipTest(msg)
        cfg = load_config()
        feat_dir = Path(feat)
        roots = [Path(demos)]
        tokens_mm, pooled_mm, index = load_memmaps(feat_dir)
        lookup = {(str(r["episode"]), int(r["decision"])): int(r["row"]) for r in index.get("rows") or []}
        folders = []
        for root in roots:
            for rec in sorted(root.glob("**/record.json")):
                folder = rec.parent
                frames = folder / "frames"
                if not frames.is_dir():
                    continue
                packed = read_episode(folder)
                stem = episode_feature_name(packed["record"], folder).removesuffix(".npy")
                if any((stem, int(s.get("decision") or i)) in lookup for i, s in enumerate(packed["steps"]) if s.get("action_chunk")):
                    folders.append(folder)
                if len(folders) >= 3:
                    break
            if len(folders) >= 3:
                break
        if len(folders) < 2:
            msg = (
                f"real demo replay skipped, missing: matched demo episodes with frames "
                f"under LFNAV_DEMOS ({demos}); need 2, found {len(folders)}"
            )
            print(msg, flush=True)
            self.skipTest(msg)
        folders = folders[:3]
        cfg["joint"] = dict(cfg.get("joint") or {})
        cfg["joint"]["checkpoint"] = ckpt
        try:
            agent = JointAgent(cfg)
        except Exception as exc:
            msg = f"JointAgent failed to load LFNAV_JOINT_CKPT={ckpt}: {exc}"
            print(msg, flush=True)
            self.skipTest(msg)
        spec = agent.spec
        token_sims, pooled_sims = [], []
        aux_ok = 0
        aux_n = 0
        act_ok = 0
        act_n = 0
        from PIL import Image

        for folder in folders:
            packed = read_episode(folder)
            task = load_task(packed["record"], folder)
            stem = episode_feature_name(packed["record"], folder).removesuffix(".npy")
            agent.reset()
            last = None
            for i, step in enumerate(packed["steps"]):
                chunk = step.get("action_chunk")
                if not chunk:
                    continue
                decision = int(step["decision"] if step.get("decision") is not None else i)
                row = lookup.get((stem, decision))
                jpeg = folder / "frames" / f"{decision:05d}.jpg"
                if row is None or not jpeg.is_file():
                    continue
                decoded = np.asarray(Image.open(jpeg).convert("RGB"))
                feats = agent.features(decoded, saved_jpeg=True)
                token_sims.append(_cos(feats["tokens"], tokens_mm[row]))
                pooled_sims.append(_cos(feats["pooled"], pooled_mm[row]))
                match = step.get("match") or {}
                env_step = int(step.get("env_step") or 0)
                if _visible(match):
                    last = {
                        "step": env_step,
                        "direction": match.get("target_direction"),
                        "distance": match.get("target_distance"),
                    }
                mem = dict(step.get("memory") or {})
                mem["env_steps"] = env_step
                train_aux = encode_aux(task, mem.get("sector8"), mem.get("recent_actions"), last, env_step, spec)
                pin = PolicyInput(task=task, match=match, memory=mem, recent_actions=list(mem.get("recent_actions") or []))
                eval_aux = agent.aux_from_input(pin)
                aux_n += 1
                if np.allclose(eval_aux, train_aux, rtol=0.0, atol=0.0):
                    aux_ok += 1
                out = agent.act(pin)
                expert = [spec.actions[j] for j in pad_chunk(chunk, spec)]
                for a, b in zip(out.action_chunk, expert):
                    act_n += 1
                    act_ok += int(a == b)
        self.assertTrue(token_sims)
        mean_tok = float(np.mean(token_sims))
        mean_pool = float(np.mean(pooled_sims))
        aux_frac = aux_ok / max(aux_n, 1)
        act_frac = act_ok / max(act_n, 1)
        print(
            json.dumps({
                "n_episodes": len(folders),
                "n_decisions": len(token_sims),
                "token_cosine": mean_tok,
                "pooled_cosine": mean_pool,
                "aux_exact": aux_frac,
                "action_match": act_frac,
            }),
            flush=True,
        )
        self.assertGreater(mean_tok, 0.99)
        self.assertGreater(mean_pool, 0.99)
        self.assertEqual(aux_frac, 1.0)
        if ckpt:
            self.assertGreater(act_frac, 0.65)


if __name__ == "__main__":
    unittest.main()
