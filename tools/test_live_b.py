#!/usr/bin/env python3
"""Real MP5 chat/Memory with a limited wood-task simulator, NOT Minecraft evaluation."""
import argparse
import json
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AGENT = REPO / "MP5_agent/agent"


class WoodWorld:
    def reset(self):
        self.inventory = {}
        self.visible = False
        self.near = False
        self.trace = []
        return self.obs()

    def obs(self):
        return {"life_stats": {"life": 20}, "inventory": {
            "name": np.array(list(self.inventory)),
            "quantity": np.array(list(self.inventory.values()))}}

    def step(self, action):
        return self.obs(), 0, False, {}

    def tick(self, env, action):
        self.trace.append(action)
        env.step([0])

    @staticmethod
    def check_wood(name):
        if name not in ("wood", "tree", "log"):
            raise RuntimeError(f"Simulator only supports wood targets, received {name!r}")

    def find(self, env, object, percipient, memory, task_information, underground):
        self.check_wood(object)
        self.visible = True
        memory.update_current_environment_information({"wood": "A tree is visible (SIMULATED)."})
        self.tick(env, "find wood")
        return True

    def approach(self, env, object, underground):
        self.check_wood(object)
        if not self.visible:
            raise RuntimeError("Simulator: find wood before move_to")
        self.near = True
        self.tick(env, "move_to wood")

    def mine(self, env, target, equipment, underground):
        self.check_wood(target)
        if not self.near:
            raise RuntimeError("Simulator: find and move_to before mine")
        self.inventory["log"] = self.inventory.get("log", 0) + 1
        self.tick(env, "mine log")
        return list(self.inventory), list(self.inventory.values())

    def craft(self, env, name, table, furnace, craft_num):
        recipes = {"planks": ({"log": 1}, 4), "stick": ({"planks": 2}, 4),
                   "crafting_table": ({"planks": 4}, 1),
                   "wooden_pickaxe": ({"planks": 3, "stick": 2}, 1)}
        if name not in recipes:
            raise RuntimeError(f"Simulator has no recipe for {name!r}")
        if craft_num < 1:
            raise RuntimeError("Simulator requires positive crafting count")
        if name == "wooden_pickaxe" and not (table and self.inventory.get("crafting table", 0)):
            raise RuntimeError("Wooden pickaxe requires a crafting table")
        materials, output_count = recipes[name]
        for item, count in materials.items():
            if self.inventory.get(item, 0) < count * craft_num:
                raise RuntimeError(f"Simulator: insufficient {item}")
        for item, count in materials.items():
            self.inventory[item] -= count * craft_num
        item = name.replace("_", " ")
        self.inventory[item] = self.inventory.get(item, 0) + craft_num * output_count
        self.inventory = {k: v for k, v in self.inventory.items() if v > 0}
        self.tick(env, f"craft {name} x {craft_num} batches")
        return list(self.inventory), list(self.inventory.values())


def self_test():
    world = WoodWorld()
    world.reset()
    try:
        world.mine(world, "wood", "", False)
        raise AssertionError("Mining before approaching should fail")
    except RuntimeError:
        pass
    mem = SimpleNamespace(update_current_environment_information=lambda value: None)
    world.find(world, "wood", None, mem, {}, False)
    world.approach(world, "wood", False)
    for _ in range(3):
        world.mine(world, "wood", "", False)
    for name, count, table in [("planks", 3, False), ("stick", 1, False),
                               ("crafting_table", 1, False), ("wooden_pickaxe", 1, True)]:
        world.craft(world, name, table, False, count)
    assert world.inventory == {"planks": 3, "stick": 2, "crafting table": 1, "wooden pickaxe": 1}
    print("PASS simulator self-test; no API requests")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs/openrouter.yaml"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--task", default=str(AGENT / "tasks/process/craft_wooden_pickaxe.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    sys.path.insert(0, str(AGENT))
    from config import load_config
    from llm_factory import create_chat_openai, create_openai_embeddings
    from task_parser import validate_execution_tasks

    cfg = load_config(args.config)
    task = Path(args.task).expanduser().resolve()
    task_data = json.loads(task.read_text())
    validate_execution_tasks(task_data[::-1])
    if "process" not in task.parts:
        raise ValueError("Use a task under a process/ directory for inventory-based success evaluation")
    cfg["eval"].update(max_subgoal_retries=3, max_steps=100, save_frames=False)
    cfg["env"]["start_inventory"] = []
    out = Path(args.out or REPO / "logs/live_b" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "input_task.json").write_text(json.dumps(task_data, indent=2))
    world = WoodWorld()
    actions = ModuleType("structured_actions")
    actions.reset_state = lambda: None
    actions.explore_above_ground = world.find
    actions.approach = world.approach
    actions.mine = world.mine
    actions.action_craft = world.craft
    # Replace only game imports; NEVER install the offline test's LLM stubs.
    sys.modules["structured_actions"] = actions
    import run_agent
    import performer

    calls = {"chat": 0}

    def chat_factory(model_cfg):
        client = create_chat_openai(model_cfg)
        client.request_timeout = 60
        client.max_retries = 0
        client.max_tokens = 1500

        def call(messages):
            if calls["chat"] >= 40:
                raise RuntimeError("Live B test reached 40 chat-call budget")
            calls["chat"] += 1
            return client(messages)
        return call

    def embedding_factory(embedding_cfg=None):
        client = create_openai_embeddings(embedding_cfg)
        client.request_timeout = 60
        client.max_retries = 1
        return client

    execute = performer.Performer.check_and_execute_workflow

    def checked_execute(self, env, workflow_dict, task_information, underground):
        # Test capability guard, not a production planner validator.
        for step in workflow_dict.get("workflow", []):
            for action in step["actions"]:
                if action["name"] not in ("find", "move_to", "mine", "craft"):
                    raise RuntimeError(f"Simulator does not support action {action['name']!r}")
        return execute(self, env, workflow_dict, task_information, underground)

    with ExitStack() as stack:
        for module in ("planner", "patroller", "memory"):
            stack.enter_context(patch(f"{module}.create_chat_openai", chat_factory))
        stack.enter_context(patch("memory.create_openai_embeddings", embedding_factory))
        stack.enter_context(patch("run_agent.Percipient", lambda **kw: SimpleNamespace()))
        stack.enter_context(patch.object(performer.Performer, "check_and_execute_workflow", checked_execute))
        print("LIVE model/embeddings calls; simulated game and perception. Output:", out, flush=True)
        result = run_agent.run_episode(str(task), 10001, cfg, out / "episode", env=world)
    result.update(test_mode="live_models_simulated_game", api_calls=calls,
                  simulated_inventory=world.inventory, simulated_actions=world.trace)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("Results:", out / "result.json")
    passed = result["success"] is True and world.inventory.get("wooden pickaxe", 0) >= 1
    print("PASS LIVE B (SIMULATED GAME)" if passed else "LIVE B DID NOT PASS; inspect result and episode logs")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
