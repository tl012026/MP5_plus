"""Perception dry-run, feature cache, MineLLM parsing, and offline metrics."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lfnav.eval.eval_perception import score_scenes
from lfnav.perception.encoder import FakeEncoder, load_encoder

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
from lfnav.perception.extract_features import extract_shard
from lfnav.perception.minellm import parse_choice, parse_yes, scene_from_answers
from lfnav.perception.preprocess import target_hw
from lfnav.perception.learned import train_step_numpy
from lfnav.perception.train_perception import dry_run, split_name


class PerceptionTest(unittest.TestCase):
    def test_resize_and_split(self):
        self.assertEqual(target_hw(100, 200, 224, 14)[0], 224)
        self.assertEqual(target_hw(100, 200, 224, 14)[1] % 14, 0)
        self.assertEqual(split_name(0), "train")
        self.assertEqual(split_name(8), "val")
        self.assertEqual(split_name(9), "test")

    def test_dry_run_and_cache(self):
        dry_run()
        rgb = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "shard_00000.npz"
            np.savez_compressed(src, rgb=rgb)
            dst = Path(tmp) / "feat.npz"
            extract_shard(src, dst, FakeEncoder())
            with np.load(dst) as data:
                self.assertEqual(data["cls"].dtype, np.float16)
                self.assertEqual(data["cls"].shape[0], 2)

    def test_minellm_parse(self):
        self.assertEqual(parse_yes("Yes, there is."), 1.0)
        self.assertEqual(parse_yes("No."), 0.0)
        labels = {"biomes": ["forest", "plains", "other"], "time": ["day", "night"], "weather": ["clear", "rain"], "brightness": ["sufficient", "insufficient"]}
        scene = scene_from_answers(
            {"present:pig": 1.0, "biome": "plains", "time": "night", "weather": "rain", "brightness": "sufficient"},
            ["pig"],
            labels,
        )
        self.assertEqual(scene.object("pig").present, 1.0)
        self.assertEqual(parse_choice("plains", labels["biomes"])["plains"], 1.0)
        self.assertEqual(scene.time["night"], 1.0)

    def test_metrics(self):
        rows = [{
            "pred": {"objects": {"pig": {"present": 0.9, "direction": {"center": 1}, "distance": {"near": 1}}}},
            "gt": {"objects": {"pig": {"present": 1, "direction": {"center": 1}, "distance": {"near": 1}}}},
        }, {
            "pred": {"objects": {"pig": {"present": 0.8, "direction": {"left": 1}, "distance": {"far": 1}}}},
            "gt": {"objects": {"pig": {"present": 0, "direction": {"center": 0}, "distance": {"far": 0}}}},
        }]
        report = score_scenes(rows)
        self.assertGreater(report["present"]["f1"], 0.0)
        self.assertGreater(report["present"]["hallucination"], 0.0)
        self.assertEqual(report["direction_accuracy"], 1.0)

    @unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
    def test_dinov2_wrap_from_hub_stub(self):
        dim = 8

        class FakeDino(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward_features(self, tensor):
                batch, _, height, width = tensor.shape
                n_h, n_w = height // 14, width // 14
                return {
                    "x_norm_clstoken": torch.zeros(batch, dim),
                    "x_norm_patchtokens": torch.zeros(batch, n_h * n_w, dim),
                }

        with patch("torch.hub.load", return_value=FakeDino()) as hub:
            encoder = load_encoder("dinov2_vits14")
        hub.assert_called_once()
        self.assertEqual(encoder.name, "dinov2_vits14")
        out = encoder(np.zeros((224, 364, 3), dtype=np.uint8))
        self.assertEqual(out["tokens"].shape, (16, 26, dim))
        self.assertEqual(out["cls"].shape, (dim,))
        self.assertEqual(out["pooled"].shape, (dim,))

    @unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
    def test_dinov2_uses_local_hub_without_network(self):
        dim = 8

        class FakeDino(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward_features(self, tensor):
                batch, _, height, width = tensor.shape
                return {
                    "x_norm_clstoken": torch.zeros(batch, dim),
                    "x_norm_patchtokens": torch.zeros(batch, (height // 14) * (width // 14), dim),
                }

        def blocked(*_args, **_kwargs):
            raise RuntimeError("network blocked")

        calls = []

        def hub_load(repo, entry, *args, **kwargs):
            calls.append((repo, entry, kwargs.get("source"), kwargs.get("skip_validation")))
            if kwargs.get("source") == "local":
                return FakeDino()
            raise RuntimeError("network blocked")

        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "hub" / "facebookresearch_dinov2_main"
            local.mkdir(parents=True)
            with patch.dict(os.environ, {"TORCH_HOME": tmp}):
                with patch("torch.hub.load", side_effect=hub_load):
                    with patch("urllib.request.urlopen", side_effect=blocked):
                        with patch("socket.create_connection", side_effect=blocked):
                            encoder = load_encoder("dinov2_vits14")
        self.assertEqual(encoder.name, "dinov2_vits14")
        self.assertEqual(len(calls), 1)
        self.assertEqual(Path(calls[0][0]), local)
        self.assertEqual(calls[0][1], "dinov2_vits14")
        self.assertEqual(calls[0][2], "local")
        out = encoder(np.zeros((224, 364, 3), dtype=np.uint8))
        self.assertEqual(out["tokens"].shape, (16, 26, dim))

    def test_ignore_patches_drop_out_of_loss(self):
        logits = np.array([10.0, -10.0])
        target = np.array([0.0, 0.0])
        full = train_step_numpy(logits, target)
        masked = train_step_numpy(logits, target, ignore=[1.0, 0.0])
        self.assertLess(masked, full)


if __name__ == "__main__":
    unittest.main()
