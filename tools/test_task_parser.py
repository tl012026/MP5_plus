"""Offline tests for decomposition accounting and task serialization."""
import copy
import json
import sys
import unittest
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "MP5_agent/agent"))
from task_parser import parse_task, validate_execution_tasks, load_knowledge, select_knowledge, KnowledgeGap


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.tasks = json.loads((REPO / "MP5_agent/agent/tasks/process/craft_wooden_pickaxe.json").read_text())[::-1]

    def test_baseline(self):
        trace = validate_execution_tasks(self.tasks)
        self.assertEqual(trace[-1]["inventory_after"]["wooden pickaxe"], 1)
        self.assertEqual(trace[-1]["inventory_after"]["planks"], 3)

    def test_batched_output(self):
        self.tasks[1]["quantity"] = 12
        self.tasks[2]["quantity"] = 4
        validate_execution_tasks(self.tasks)

    def test_insufficient_total_materials(self):
        self.tasks[0]["quantity"] = 2
        self.tasks[1]["quantity"] = 8
        self.tasks[1]["material"] = {"log": 2}
        with self.assertRaisesRegex(ValueError, "insufficient planks"):
            validate_execution_tasks(self.tasks)

    def test_wrong_order(self):
        with self.assertRaises(ValueError):
            validate_execution_tasks(self.tasks[::-1])

    def test_wrong_recipe(self):
        self.tasks[2]["material"] = {"planks": 1}
        with self.assertRaisesRegex(ValueError, "material must be"):
            validate_execution_tasks(self.tasks)

    def test_missing_table_before_pickaxe(self):
        self.tasks[-1], self.tasks[-2] = self.tasks[-2], self.tasks[-1]
        with self.assertRaisesRegex(ValueError, "missing preceding"):
            validate_execution_tasks(self.tasks)

    def test_invalid_fields(self):
        for key, value in [("quantity", True), ("task", "diamond"), ("description", "")]:
            tasks = copy.deepcopy(self.tasks)
            tasks[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_execution_tasks(tasks)

    def test_other_goal_and_dependency_selection(self):
        trace = validate_execution_tasks(self.tasks[:2], goals={"planks": 9})
        self.assertEqual(trace[-1]["inventory_after"]["planks"], 12)
        selected = select_knowledge({"planks": 9}, load_knowledge())
        self.assertEqual(set(selected), {"wood", "planks"})

    def test_existing_inventory_and_multiple_goals(self):
        validate_execution_tasks(self.tasks[-1:], goals={"wooden pickaxe": 1, "stick": 2},
            initial_inventory={"planks": 3, "stick": 4, "crafting table": 1})
        with self.assertRaisesRegex(ValueError, "Final goal unsatisfied"):
            validate_execution_tasks(self.tasks[-1:], goals={"wooden pickaxe": 1, "stick": 3},
                initial_inventory={"planks": 3, "stick": 4, "crafting table": 1})
        validate_execution_tasks([], goals={"stick": 2}, initial_inventory={"stick": 2})

    def test_multiple_pickaxes(self):
        tasks = copy.deepcopy(self.tasks)
        tasks[0]["quantity"] = 4
        tasks[1]["quantity"] = 16
        tasks[1]["material"] = {"log": 4}
        tasks[-1]["quantity"] = 2
        tasks[-1]["material"] = {"planks": 6, "stick": 4}
        validate_execution_tasks(tasks, goals={"wooden pickaxe": 2})

    def test_knowledge_extension_without_parser_change(self):
        records = load_knowledge()
        records["test item"] = dict(kind="craft", output_item="test item", output_count=1,
            material={"planks": 1}, tool=None, platform=None, enabled=True, source="test fixture only")
        task = dict(task="test item", quantity=1, material={"planks": 1}, tool=None,
                    platform=None, tips="Test recipe", description="Test fixture")
        validate_execution_tasks([task], goals={"test item": 1},
                                 initial_inventory={"planks": 1}, knowledge=records)

    def test_coverage_failure_before_api_call(self):
        with tempfile.TemporaryDirectory() as td, patch("llm_factory.create_chat_openai") as factory:
            out = Path(td) / "run"
            with self.assertRaises(KnowledgeGap):
                parse_task("Obtain a diamond", {}, out, goals={"diamond": 1})
            factory.assert_not_called()
            self.assertEqual(json.loads((out / "validation.json").read_text())["status"], "needs_knowledge")
            self.assertFalse((out / "process").exists())

    def test_disabled_capability_and_cycle(self):
        records = load_knowledge()
        records["wood"]["enabled"] = False
        with self.assertRaisesRegex(KnowledgeGap, "executor_support"):
            select_knowledge({"wooden pickaxe": 1}, records)
        records = load_knowledge()
        records["wood"]["tool"] = "wooden pickaxe"
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            select_knowledge({"wooden pickaxe": 1}, records)

    def test_nonempty_inventory_exports_plan_only(self):
        schema = ModuleType("langchain.schema")
        schema.SystemMessage = schema.HumanMessage = lambda **kw: SimpleNamespace(**kw)
        client = Mock(return_value=SimpleNamespace(content=json.dumps(self.tasks[-1:])))
        with tempfile.TemporaryDirectory() as td, patch.dict(sys.modules, {"langchain.schema": schema}), patch("llm_factory.create_chat_openai", return_value=client):
            out = Path(td) / "run"
            path = parse_task("Make one wooden pickaxe", {}, out,
                initial_inventory={"planks": 3, "stick": 2, "crafting table": 1})
            self.assertEqual(path.name, "execution_order.json")
            self.assertFalse((out / "process").exists())
            self.assertEqual(json.loads((out / "validation.json").read_text())["status"], "validated_plan_only")

    def test_generation_saves_reversed_task(self):
        schema = ModuleType("langchain.schema")
        schema.SystemMessage = schema.HumanMessage = lambda **kw: SimpleNamespace(**kw)
        client = Mock(return_value=SimpleNamespace(content=json.dumps(self.tasks)))
        with tempfile.TemporaryDirectory() as td, patch.dict(sys.modules, {"langchain.schema": schema}), patch("llm_factory.create_chat_openai", return_value=client):
            out = Path(td) / "run"
            path = parse_task("Make one wooden pickaxe", {"model_name": "mock"}, out)
            self.assertEqual(json.loads(path.read_text()), self.tasks[::-1])
            self.assertEqual(path.parent.name, "process")
            self.assertTrue(json.loads((out / "validation.json").read_text())["passed"])
            client.assert_called_once()

    def test_live_missing_platform_and_shortage_repaired(self):
        # Reproduce the HPC failure: one log, four planks, sticks, no table.
        bad = copy.deepcopy(self.tasks)
        bad[0]["quantity"] = 1
        bad[1]["quantity"] = 4
        bad[1]["material"] = {"log": 1}
        bad[2]["quantity"] = 4
        del bad[3]
        schema = ModuleType("langchain.schema")
        schema.SystemMessage = schema.HumanMessage = lambda **kw: SimpleNamespace(**kw)
        client = Mock(side_effect=[SimpleNamespace(content=json.dumps(bad)),
                                  SimpleNamespace(content=json.dumps(self.tasks))])
        with tempfile.TemporaryDirectory() as td, patch.dict(sys.modules, {"langchain.schema": schema}), patch("llm_factory.create_chat_openai", return_value=client):
            out = Path(td) / "run"
            path = parse_task("Make one wooden pickaxe", {}, out)
            self.assertEqual(json.loads(path.read_text()), self.tasks[::-1])
            self.assertEqual(client.call_count, 2)
            attempts = json.loads((out / "attempts.json").read_text())
            self.assertIn("missing preceding crafting table", attempts[0]["error"])
            self.assertTrue(attempts[1]["passed"])
            self.assertEqual(json.loads((out / "raw_response_1.txt").read_text()), bad)
            self.assertIn("total material demand", client.call_args.args[0][-1].content)
        # Adding the table alone cannot fix the same resource-shortage plan.
        bad.insert(3, copy.deepcopy(self.tasks[3]))
        with self.assertRaisesRegex(ValueError, "insufficient planks"):
            validate_execution_tasks(bad)

    def test_invalid_generation_is_saved_but_not_executed(self):
        schema = ModuleType("langchain.schema")
        schema.SystemMessage = schema.HumanMessage = lambda **kw: SimpleNamespace(**kw)
        client = Mock(return_value=SimpleNamespace(content="not JSON"))
        with tempfile.TemporaryDirectory() as td, patch.dict(sys.modules, {"langchain.schema": schema}), patch("llm_factory.create_chat_openai", return_value=client):
            out = Path(td) / "run"
            with self.assertRaises(ValueError):
                parse_task("Make one wooden pickaxe", {"model_name": "mock"}, out)
            self.assertEqual((out / "raw_response.txt").read_text(), "not JSON")
            self.assertFalse((out / "process").exists())
            self.assertEqual(client.call_count, 2)
            self.assertTrue((out / "raw_response_2.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
