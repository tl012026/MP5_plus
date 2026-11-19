#!/usr/bin/env python3
"""Run real B orchestration with scripted model and game boundaries; no network."""
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from test_offline import AGENT, REPO, _install_stubs

_install_stubs()
sys.path.insert(0, str(AGENT))
import run_agent
from config import load_config


class LocalVectorStore:
    """Only replace Chroma/embeddings; Memory's bookkeeping remains real."""
    def __init__(self, **kwargs):
        self.rows = {}
        self._collection = self

    def count(self):
        return len(self.rows)

    def add_texts(self, texts, ids, metadatas):
        self.rows.update(zip(ids, metadatas))

    def similarity_search_with_score(self, query, k):
        return [(SimpleNamespace(metadata=m), 0) for m in list(self.rows.values())[:k]]

    def delete(self, ids):
        for key in ids:
            self.rows.pop(key, None)

    def persist(self):
        pass

    def delete_collection(self):
        self.rows.clear()


class GameBoundary:
    def reset(self):
        self.inventory = {}
        return self.obs()

    def obs(self):
        return {"life_stats": {"life": 20}, "inventory": {
            "name": np.array(list(self.inventory)),
            "quantity": np.array(list(self.inventory.values()))}}

    def step(self, action):
        return self.obs(), 0, False, {}

    def inventory_lists(self, env):
        env.step([0])
        return list(self.inventory), list(self.inventory.values())

    def mine(self, env, target, equipment, underground):
        assert target == "wood" and equipment == ""
        self.inventory["log"] = self.inventory.get("log", 0) + 1
        return self.inventory_lists(env)

    def craft(self, env, name, table, furnace, craft_num):
        recipes = {"planks": ({"log": 1}, 4), "stick": ({"planks": 2}, 4),
                   "crafting_table": ({"planks": 4}, 1),
                   "wooden_pickaxe": ({"planks": 3, "stick": 2}, 1)}
        materials, yield_count = recipes[name]
        if name == "wooden_pickaxe":
            assert table and self.inventory["crafting table"] == 1
        for item, quantity in materials.items():
            assert self.inventory.get(item, 0) >= quantity * craft_num
            self.inventory[item] -= quantity * craft_num
        item = name.replace("_", " ")
        self.inventory[item] = self.inventory.get(item, 0) + yield_count * craft_num
        return self.inventory_lists(env)


def main(relative_task=False):
    task = AGENT / "tasks/process/craft_wooden_pickaxe.json"
    goals = json.loads(task.read_text())[::-1]
    plans = []
    for goal in goals:
        if goal["task"] == "wood":
            actions = [{"name": "mine", "args": {"obj": "wood", "tool": None}}]
            times = 3
        else:
            actions = [{"name": "craft", "args": {
                "obj": {goal["task"]: goal["quantity"]},
                "materials": goal["material"], "platform": goal["platform"]}}]
            times = 1
        plans.append({"workflow": [{"times": times, "actions": actions}]})
    responses = iter(plans)
    game = GameBoundary()
    planner_calls = []
    checker_calls = []

    def planner_model(messages):
        planner_calls.append(messages)
        return SimpleNamespace(content=json.dumps(next(responses)))

    def checker_model(messages):
        goal = goals[len(checker_calls)]
        item = "log" if goal["task"] == "wood" else goal["task"]
        assert game.inventory.get(item, 0) >= goal["quantity"]
        checker_calls.append(messages)
        return SimpleNamespace(content=json.dumps({"success": True, "feedback": "", "suggestion": ""}))

    with tempfile.TemporaryDirectory() as td, ExitStack() as stack:
        replacements = {
            "run_agent.get_openai_key": lambda: "offline-placeholder",
            "memory.create_chat_openai": lambda cfg: None,
            "memory.create_openai_embeddings": lambda cfg=None: None,
            "memory.Chroma": LocalVectorStore,
            "planner.create_chat_openai": lambda cfg: planner_model,
            "patroller.create_chat_openai": lambda cfg: checker_model,
            # No find action in this smoke test; visual service is not exercised.
            "run_agent.Percipient": lambda **kwargs: SimpleNamespace(),
            "performer.mine": game.mine,
            "performer.action_craft": game.craft,
        }
        for target, replacement in replacements.items():
            stack.enter_context(patch(target, replacement))
        for module in ("planner", "patroller"):
            for message in ("HumanMessage", "SystemMessage"):
                stack.enter_context(patch(f"{module}.{message}", lambda **kw: SimpleNamespace(**kw)))
        task_arg = os.path.relpath(task) if relative_task else str(task)
        caller_cwd = Path.cwd()
        result = run_agent.run_episode(task_arg, 10001, load_config(), Path(td) / "episode", env=game)
        assert Path.cwd() == caller_cwd
        assert result["outcome"] == "success", result
        assert result["inventory"]["wooden pickaxe"] == 1, result
        assert len(planner_calls) == len(checker_calls) == 5
        events = [json.loads(line) for line in (Path(td) / "episode.events.jsonl").read_text().splitlines()]
        completed = [e["task"] for e in events if e["event"] == "subgoal" and e["phase"] == "end"]
        assert completed == [g["task"] for g in goals], completed
        saved = json.loads((Path(td) / "episode/memory/workflows_0.json").read_text())
        assert len(saved) == 5, saved
        assert result["steps"] == 7, result
    print("PASS: real Planner -> Performer -> Patroller -> Memory, 5 subgoals, 7 simulated steps")
    print("Boundary mocks: model replies, vector store, mining/crafting; vision and real game NOT tested")


if __name__ == "__main__":
    main()
    main(relative_task=True)
    print("PASS: absolute and caller-relative task paths; caller directory restored")
