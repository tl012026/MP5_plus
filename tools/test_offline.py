#!/usr/bin/env python3
"""Login-node tests: no MineDojo, no GPU, no network, no model weights."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
AGENT = REPO / "MP5_agent" / "agent"


def _install_stubs():
    for name in (
        "openai",
        "langchain",
        "langchain.chat_models",
        "langchain.embeddings",
        "langchain.embeddings.openai",
        "langchain.schema",
        "langchain.vectorstores",
        "minedojo",
        "minedojo.sim",
        "minedojo.sim.mc_meta",
        "minedojo.sim.mc_meta.mc",
        "PIL",
        "PIL.Image",
    ):
        sys.modules.setdefault(name, MagicMock())


def _arr(xs):
    class _A:
        def __init__(self, values):
            self._values = list(values)

        def tolist(self):
            return list(self._values)

    return _A(xs)


def test_config():
    sys.path.insert(0, str(AGENT))
    from config import load_config, resolve_api_key, validate_config

    default = load_config(REPO / "configs" / "code_default.yaml")
    strict = load_config(REPO / "configs" / "strict.yaml")
    assert default["env"]["break_speed_multiplier"] == 100.0
    assert default["env"]["specified_biome"] == "forest"
    assert len(default["env"]["start_inventory"]) == 3
    assert default["eval"]["max_steps"] == 12000
    assert default["eval"]["max_subgoal_retries"] == 300
    assert default["eval"]["save_frames"] is False
    assert strict["env"]["break_speed_multiplier"] == 1.0
    assert strict["env"]["specified_biome"] is None
    assert strict["env"]["start_inventory"] == []
    validate_config(default)
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        try:
            resolve_api_key()
            raise AssertionError("expected missing-key error")
        except RuntimeError as e:
            assert "OPENAI_API_KEY" in str(e)
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old
    print("ok config")


def test_env_wrapper_and_success():
    sys.path.insert(0, str(AGENT))
    from env_wrapper import EnvWrapper, EpisodeDeath, EpisodeTimeout
    from success import failure_category, inventory_has_item

    class Boom:
        def __init__(self, life):
            self.life = life
            self.n = 0

        def step(self, action):
            self.n += 1
            return {"life_stats": {"life": self.life}}, 0, False, {}

    w = EnvWrapper(Boom(20), max_steps=2)
    w.step([0])
    w.step([0])
    try:
        w.step([0])
        raise AssertionError("expected timeout")
    except EpisodeTimeout:
        pass

    d = EnvWrapper(Boom(0), max_steps=100)
    try:
        d.step([0])
        raise AssertionError("expected death")
    except EpisodeDeath:
        pass

    assert inventory_has_item({"wooden pickaxe": 1}, {"task": "wooden pickaxe", "quantity": 1})
    assert failure_category("timeout", True, False, False) == "timeout_in_find"
    assert failure_category("death", False, False, False) == "death"
    print("ok env_wrapper")


def test_reset_and_frames():
    _install_stubs()
    sys.path.insert(0, str(AGENT))
    import structured_actions as sa
    from runtime import runtime

    sa.explore_steps = 99
    sa.stuck = 3
    sa.save_count = 7
    sa.reset_state()
    assert sa.explore_steps == 0
    assert sa.stuck == 0
    assert sa.save_count == 0

    with tempfile.TemporaryDirectory() as td:
        runtime.bind(td, {"save_frames": False, "save_perception_images": True})
        sa.save_count = 0
        sa.save_rgb_for_video({"rgb": __import__("numpy").zeros((3, 4, 4), dtype="uint8")})
        frames = list(Path(td).joinpath("video").glob("*.jpg"))
        assert frames == [], frames
        runtime.reset()
    print("ok reset_state / save_frames")


def test_run_episode_fake():
    _install_stubs()
    sys.path.insert(0, str(AGENT))
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
    import run_agent
    from config import load_config

    class FakeMemory:
        def __init__(self, **kwargs):
            self.inventory = {}
            self.ckpt_dir = kwargs.get("ckpt_dir")

        def update_inventory(self, d):
            self.inventory = dict(d)

        def reset_current_environment_information(self):
            pass

        def reset_all(self):
            pass

        def add_successful_workflow(self, *a, **k):
            pass

    class FakePlanner:
        def __init__(self, **kwargs):
            self.event_logger = kwargs.get("event_logger")

        def get_workflow(self, task_information, underground, check_result):
            if self.event_logger:
                self.event_logger.emit(
                    "planner_call",
                    subgoal=task_information.get("task"),
                    inventory_summary=dict(self.inventory) if hasattr(self, "inventory") else {},
                    feedback_summary="",
                    raw_output="{}",
                    parse_ok=True,
                    retries_left=5,
                    latency_s=0.0,
                )
            return {"workflow": [{"times": 1, "actions": [{"name": "equip", "args": {"obj": "hand"}}]}]}

    class FakePatroller:
        def __init__(self, **kwargs):
            pass

        def check_action_preparation(self, name, inventory, args):
            return {"success": True, "feedback": "", "suggestion": ""}

        def check_task_success(self, task_information, max_retries=5):
            return {"success": True, "feedback": "done", "suggestion": ""}

    class FakePercipient:
        def __init__(self, **kwargs):
            pass

    class FakePerformer:
        def __init__(self, **kwargs):
            self.event_logger = kwargs.get("event_logger")

        def check_and_execute_workflow(self, env, workflow_dict, task_information, underground):
            if self.event_logger:
                self.event_logger.emit("action", name="equip", args={}, start_step=0, end_step=1, result="ok")
            return {"success": True, "feedback": "", "suggestion": ""}, underground

    class FakeEnv:
        def reset(self):
            return self._obs(20)

        def step(self, action):
            return self._obs(20), 0, False, {}

        def _obs(self, life):
            return {
                "life_stats": {"life": life},
                "inventory": {"name": _arr(["wooden pickaxe"]), "quantity": _arr([1])},
                "location_stats": {"pos": [0, 64, 0], "biome_id": 4, "is_raining": False},
                "rays": {"block_name": ["dirt"], "entity_name": []},
            }

    run_agent.Memory = FakeMemory
    run_agent.Planner = FakePlanner
    run_agent.Patroller = FakePatroller
    run_agent.Percipient = FakePercipient
    run_agent.Performer = FakePerformer

    cfg = load_config(REPO / "configs" / "code_default.yaml")
    task = REPO / "MP5_agent" / "agent" / "tasks" / "process" / "craft_wooden_pickaxe.json"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "craft_wooden_pickaxe" / "10001"
        rec = run_agent.run_episode(str(task), 10001, cfg, out, env=FakeEnv())
        assert rec["outcome"] == "success", rec
        assert rec["success"] is True
        events = out.parent / "10001.events.jsonl"
        assert events.is_file(), events
        lines = [json.loads(x) for x in events.read_text().splitlines() if x.strip()]
        kinds = {x["event"] for x in lines}
        assert "planner_call" in kinds
        assert "episode_end" in kinds
        assert list((out / "video").glob("*.jpg")) == []
        assert rec["failure_category"] is None
    print("ok run_episode fake")


def test_batch_skip_and_summary():
    sys.path.insert(0, str(REPO / "eval"))
    from summarize import write_summary

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "craft_wooden_pickaxe" / "10001.json"
        dest.parent.mkdir(parents=True)
        dest.write_text(json.dumps({
            "task_name": "craft_wooden_pickaxe",
            "seed": 10001,
            "task_kind": "process",
            "outcome": "success",
            "success": True,
            "steps": 3,
            "elapsed_s": 0.1,
            "failure_category": None,
            "self_reported_steps": None,
        }))
        csv_path = write_summary(td)
        text = csv_path.read_text()
        assert "craft_wooden_pickaxe" in text
        assert "failure_category" in text
    print("ok summarize")


def main():
    _install_stubs()
    test_config()
    test_env_wrapper_and_success()
    test_reset_and_frames()
    test_run_episode_fake()
    test_batch_skip_and_summary()
    print("ALL OFFLINE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
