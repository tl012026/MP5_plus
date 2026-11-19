"""Exercise the real Patroller with scripted model responses; no network or game."""
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_offline import AGENT, _install_stubs

_install_stubs()
sys.path.insert(0, str(AGENT))
import patroller


def reply(value):
    return SimpleNamespace(content=json.dumps(value))


class PatrollerTests(unittest.TestCase):
    def setUp(self):
        # Patch external message types/client, retaining the production checker
        # and JSON parser. File loading is isolated from the caller's cwd.
        self.patches = [
            patch.object(patroller, "create_chat_openai", return_value=Mock()),
            patch.object(patroller, "load_prompt", return_value="{task_information} {inventory}"),
            patch.object(patroller, "SystemMessage", side_effect=lambda **kw: kw),
            patch.object(patroller, "HumanMessage", side_effect=lambda **kw: kw),
            patch.object(patroller, "log_info"),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.checker = patroller.Patroller(
            openai_key="unused",
            memory=SimpleNamespace(inventory={"log": 3}, current_environment_information=[]),
        )
        self.task = {"description": "Mine 3 wood."}

    def test_transient_error_then_success(self):
        self.checker.llm.side_effect = [TimeoutError("simulated"), reply({"success": True})]
        result = self.checker.check_task_success(self.task, max_retries=2)
        self.assertEqual(result, {"success": True, "feedback": "", "suggestion": ""})
        self.assertEqual(self.checker.llm.call_count, 2)
        self.assertEqual(self.checker.llm.call_args_list[0], self.checker.llm.call_args_list[1])

    def test_persistent_error_is_bounded(self):
        self.checker.llm.side_effect = TimeoutError("simulated")
        with self.assertRaisesRegex(RuntimeError, "attempt budget"):
            self.checker.check_task_success(self.task, max_retries=3)
        self.assertEqual(self.checker.llm.call_count, 3)

    def test_zero_budget_makes_no_request(self):
        with self.assertRaises(RuntimeError):
            self.checker.check_task_success(self.task, max_retries=0)
        self.checker.llm.assert_not_called()

    def test_valid_negative_result_is_not_retried(self):
        value = {"success": False, "feedback": "Need more wood", "suggestion": "Mine again"}
        self.checker.llm.return_value = reply(value)
        self.assertEqual(self.checker.check_task_success(self.task), value)
        self.checker.llm.assert_called_once()

    def test_invalid_success_values_retry(self):
        for value in (1, "false", None):
            with self.subTest(value=value):
                self.checker.llm.reset_mock()
                self.checker.llm.side_effect = [reply({"success": value}), reply({"success": False})]
                self.assertFalse(self.checker.check_task_success(self.task, max_retries=2)["success"])
                self.assertEqual(self.checker.llm.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
