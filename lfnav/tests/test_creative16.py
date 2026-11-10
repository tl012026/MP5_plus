"""creative16-v1: run.py episode JSON matches result_schema.json."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from lfnav.config import load_config
from lfnav.env.fake import FakeMineDojoEnv
from lfnav.eval.run import run_episode
from lfnav.types import TaskSpec

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "benchmark" / "creative16" / "result_schema.json"
SPECS = ROOT / "lfnav" / "tasks" / "specs"


def _type_ok(value, types) -> bool:
    if isinstance(types, str):
        types = [types]
    for t in types:
        if t == "null" and value is None:
            return True
        if t == "string" and isinstance(value, str):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "object" and isinstance(value, dict):
            return True
        if t == "array" and isinstance(value, list):
            return True
    return False


def validate_episode_result(record: dict, schema: dict) -> None:
    for key in schema["required"]:
        if key not in record:
            raise AssertionError(f"missing required field {key}")
    for key, spec in (schema.get("properties") or {}).items():
        if key not in record or "type" not in spec:
            continue
        if not _type_ok(record[key], spec["type"]):
            raise AssertionError(f"{key} has type {type(record[key]).__name__}, schema {spec['type']}")
        if "enum" in spec and record[key] not in spec["enum"]:
            raise AssertionError(f"{key}={record[key]!r} not in {spec['enum']}")


class ResultSchemaTest(unittest.TestCase):
    def test_run_episode_output_validates(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        cfg = load_config()
        cfg["image_hw"] = [14, 14]
        cfg["perception_backend"] = "gt"
        cfg["policy_name"] = "c1"
        cfg["backend"] = "fake"
        cfg["difficulty"] = "easy"
        cfg["judge"] = dict(cfg.get("judge") or {})
        cfg["judge"]["max_steps"] = 40
        cfg["match"] = dict(cfg.get("match") or {})
        cfg["match"]["stop_mode"] = "sr_b"
        task = TaskSpec.from_dict(json.loads((SPECS / "1-4.json").read_text()))
        env = FakeMineDojoEnv(cfg, {"entities": [{"name": "pig", "x": 0.0, "z": 6.0}], "biome": "plains"})
        rec, _events = run_episode(env, task, cfg, seed=0)
        validate_episode_result(rec, schema)
        self.assertEqual(rec["task"], rec["task_name"])
        self.assertEqual(rec["policy"], rec["policy_name"])


class ManifestTest(unittest.TestCase):
    def test_listed_paths_exist(self):
        text = (ROOT / "benchmark" / "creative16" / "manifest.yaml").read_text()
        paths = []
        for raw in text.splitlines():
            stripped = raw.split("#")[0].strip()
            if stripped.startswith("- "):
                candidate = stripped[2:].strip()
            elif ": " in stripped:
                candidate = stripped.split(": ", 1)[1].strip()
            else:
                continue
            if "*" in candidate or " " in candidate:
                continue
            if "/" in candidate and candidate.endswith((".json", ".yaml", ".py", ".txt")):
                paths.append(candidate)
        self.assertGreaterEqual(len(paths), 16)
        for line in paths:
            self.assertTrue((ROOT / line).is_file(), line)
