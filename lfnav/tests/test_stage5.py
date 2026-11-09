"""Expert demos and the C2 interface, without a GPU."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lfnav.config import load_config
from lfnav.data.collect_demos import assert_train_seed, collect
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.policy.c2 import BehaviorCloning
from lfnav.policy.expert import Expert, wide_target
from lfnav.policy.train_c2 import dry_run
from lfnav.types import PolicyInput

ROOT = Path(__file__).resolve().parents[2]


class Stage5Test(unittest.TestCase):
    def test_seed_split_and_c2_chunk(self):
        with self.assertRaises(ValueError):
            assert_train_seed(19)
        assert_train_seed(11)
        policy = BehaviorCloning({"seed": 1})
        output = policy.act(PolicyInput(match={"status": "not_found", "target_direction": "left", "missing": []}))
        self.assertEqual(len(output.action_chunk), 4)
        dry_run()

    def test_expert_saves_success(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        cfg["judge"] = {"max_steps": 30, "radius": 3}
        cfg["match"] = {"stop_mode": "sr_b", "tau_obj": 0.5, "tau_cond": 0.5, "tau_near": 0.5, "smooth_n": 1}
        with tempfile.TemporaryDirectory() as tmp:
            saved = collect(cfg, [str(ROOT / "lfnav/tasks/specs/1-4.json")], [1], Path(tmp))
            self.assertGreaterEqual(saved["success"], 1)
            path = next((Path(tmp) / "success").glob("*/*.json"))
            rec = json.loads(path.read_text())
            self.assertTrue(rec["record"]["success"])

    def test_expert_turns_right_when_target_is_to_the_right(self):
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 4.0, "z": 0.3}]})
        obs = env.reset()
        wide = wide_target(obs, "pig", cfg)
        self.assertTrue(wide.get("visible"))
        self.assertEqual(wide.get("direction"), "right")
        out = Expert(cfg).act(PolicyInput(
            match={"status": "not_found", "satisfied": {}, "missing": []},
            memory={"wide_target": wide, "local_grid": [[0.0]]},
        ))
        self.assertEqual(out.action_chunk[0], "turn_right_15")


if __name__ == "__main__":
    unittest.main()
