"""python -m unittest discover -s minedreamer_bridge/tests -t .   (no MineDojo, MineRL, torch, or GPU)"""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minedreamer_bridge import actions as A
from minedreamer_bridge.backends import HttpImaginator, MockImaginator, MockPromptGenerator, MockSteve1
from minedreamer_bridge.fake_world import FakeMinecraft
from minedreamer_bridge.imageio import decode_png, encode_png
from minedreamer_bridge.instructions import instruction_for, mp5_inventory
from minedreamer_bridge.performer import DreamerPerformer
from minedreamer_bridge.schema import validate
from minedreamer_bridge.mc116 import CommandCrafter, task_setup
from minedreamer_bridge.remote import PerformerServer, RemotePerformer
from minedreamer_bridge.simulate import DEFAULT_WORKFLOW, DEMOS, gt_perceive


def make_performer(out=None, seed=0, **kw):
    env = FakeMinecraft(seed=seed)
    perf = DreamerPerformer(env, kw.pop("imaginator", MockImaginator()), MockPromptGenerator(), MockSteve1(seed),
                            out_dir=out, perceive_fn=gt_perceive(env), **kw)
    return env, perf


class InstructionTest(unittest.TestCase):
    def test_mapping(self):
        mine = instruction_for({"name": "mine", "args": {"obj": "wood", "tool": None}})
        self.assertEqual((mine.text_prompt, mine.in_distribution), ("chop a tree", True))
        self.assertEqual(mine.done, {"kind": "inventory_gain", "item": "log", "count": 1})
        self.assertEqual(instruction_for({"name": "dig_down", "args": {"y_level": 11}}).done["y"], 11.0)
        self.assertTrue(instruction_for({"name": "craft", "args": {"obj": {"planks": 4}}}).supported)
        self.assertFalse(instruction_for({"name": "craft", "args": {"obj": {"crafting table": 1}}}).supported)
        self.assertFalse(instruction_for({"name": "fight", "args": {"obj": "pig"}}).supported)
        self.assertFalse(instruction_for({"name": "find", "args": {"obj": "pig"}}).in_distribution)

    def test_inventory_names(self):
        self.assertEqual(mp5_inventory({"oak_log": 2, "birch_log": 1, "crafting_table": 1, "dirt": 0}),
                         {"log": 3, "crafting table": 1})


class ActionTest(unittest.TestCase):
    def test_minedojo_view(self):
        self.assertEqual(A.to_minedojo(A.noop()), [0, 0, 0, 12, 12, 0, 0, 0])
        self.assertEqual(A.to_minedojo(A.make(forward=1, attack=1, camera=[-15.0, 30.0])), [1, 0, 0, 11, 14, 3, 0, 0])
        self.assertEqual(A.compact(A.make(jump=1, camera=[0, 2.5])), {"jump": 1, "camera": [0.0, 2.5]})
        with self.assertRaises(KeyError):
            A.make(fly=1)


class PngTest(unittest.TestCase):
    def test_roundtrip_all_filters(self):
        img = np.random.default_rng(0).integers(0, 256, (9, 7, 3), dtype=np.uint8)
        for filters in ([0], [1], [2], [3], [4], [0, 1, 2, 3, 4]):
            np.testing.assert_array_equal(decode_png(encode_png(img, filters)), img)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _Session:
    def __init__(self, goal):
        self.goal, self.calls = goal, []

    def post(self, url, files=None, data=None, timeout=None):
        self.calls.append((url, files, data))
        return _Resp({"result": 1, "goal_image_list": [base64.b64encode(encode_png(self.goal)).decode()]})


class HttpImaginatorTest(unittest.TestCase):
    def test_request_and_decode(self):
        goal = np.full((256, 256, 3), 7, np.uint8)
        sess = _Session(goal)
        out = HttpImaginator("http://host:25547", session=sess).imagine("chop a tree", np.zeros((160, 256, 3), np.uint8))
        url, files, data = sess.calls[0]
        self.assertEqual(url, "http://host:25547/goal_image")
        self.assertEqual(data, {"text_prompt": "chop a tree", "is_del": 1})
        self.assertEqual(decode_png(files["current_image"][1]).shape, (256, 256, 3))
        np.testing.assert_array_equal(out[0], goal)


class PerformerTest(unittest.TestCase):
    def test_default_workflow_mines_a_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            _env, perf = make_performer(Path(tmp), save_frames=True)
            check, underground = perf.check_and_execute_workflow(perf.env, DEFAULT_WORKFLOW, {}, False)
            self.assertTrue(check["success"], check)
            self.assertFalse(underground)
            self.assertEqual([r["outcome"] for r in perf.records], ["done", "done", "done"])
            self.assertEqual(perf.records[-1]["inventory_delta"], {"log": 1})
            for i, rec in enumerate(perf.records):
                self.assertEqual(validate(rec), [])
                on_disk = json.loads((Path(tmp) / f"{i:04d}_{rec['action']['name']}.json").read_text())
                self.assertEqual(on_disk["outcome"], rec["outcome"])
                rows = (Path(tmp) / rec["trace_file"]).read_text().splitlines()
                self.assertEqual(len(rows), rec["steps"])
            self.assertTrue(any((Path(tmp) / "frames").iterdir()))

    def test_dig_down_sets_underground(self):
        _env, perf = make_performer()
        wf = {"workflow": [{"times": 1, "actions": [{"name": "dig_down", "args": {"y_level": 60, "tool": None}}]}]}
        check, underground = perf.check_and_execute_workflow(perf.env, wf, {}, False)
        self.assertTrue(check["success"])
        self.assertTrue(underground)
        self.assertLessEqual(perf.records[0]["pose_after"]["pos"][1], 60)

    def test_failure_goes_back_to_planner(self):
        _env, perf = make_performer(budgets={"mine": 50})
        wf = {"workflow": [{"times": 1, "actions": [{"name": "mine", "args": {"obj": "iron ore", "tool": None}},
                                                    {"name": "mine", "args": {"obj": "wood", "tool": None}}]}]}
        check, _ = perf.check_and_execute_workflow(perf.env, wf, {}, False)
        self.assertFalse(check["success"])
        self.assertIn("iron ore did not increase", check["feedback"])
        self.assertEqual(len(perf.records), 1)

    def test_unsupported_and_noop(self):
        _env, perf = make_performer()
        craft = perf.execute_action({"name": "craft", "args": {"obj": {"crafting table": 1}, "platform": None}})
        equip = perf.execute_action({"name": "equip", "args": {"obj": "wooden pickaxe"}})
        self.assertEqual((craft["outcome"], craft["check_result"]["success"]), ("unsupported", False))
        self.assertEqual((equip["outcome"], equip["check_result"]["success"]), ("noop", True))

    def test_precheck_uses_patroller(self):
        class Checker:
            def check_action_preparation(self, name, inventory, args):
                return {"success": False, "feedback": "no stone pickaxe", "suggestion": "craft it"}

        class Memory:
            inventory = {}

            def update_inventory(self, inv):
                self.inventory = inv

        _env, perf = make_performer(checker=Checker(), memory=Memory())
        rec = perf.execute_action({"name": "mine", "args": {"obj": "iron ore", "tool": "stone pickaxe"}})
        self.assertEqual(rec["outcome"], "precheck_fail")
        self.assertEqual(rec["steps"], 0)

    def test_backend_error_still_writes_file(self):
        class Broken:
            name = "broken"

            def imagine(self, text, frame):
                raise ConnectionError("imaginator down")

        with tempfile.TemporaryDirectory() as tmp:
            _env, perf = make_performer(Path(tmp), imaginator=Broken())
            rec = perf.execute_action({"name": "mine", "args": {"obj": "wood", "tool": None}})
            self.assertEqual(rec["outcome"], "error")
            self.assertIn("imaginator down", rec["error"])
            self.assertTrue((Path(tmp) / "0000_mine.json").is_file())


class Mc116Test(unittest.TestCase):
    def test_task_setup_from_config_and_episode(self):
        env_cfg = {"specified_biome": "forest", "start_at_night": False, "start_inventory": [
            {"slot": 9, "name": "dirt", "quantity": 6}, {"slot": 16, "name": "coal", "quantity": 3},
            {"slot": 17, "name": "coal", "quantity": 3}]}
        episode = {"specified_biome": "plains", "start_time": "night", "initial_weather": "rain",
                   "spawn": [{"name": "pig", "kind": "mob"}, {"name": "water", "kind": "block"}]}
        setup = task_setup(env_cfg, episode, seed=9)
        self.assertEqual(setup["preferred_spawn_biome"], "plains")
        self.assertEqual(sorted(setup["inventory"], key=lambda r: r["type"]),
                         [{"type": "coal", "quantity": 6}, {"type": "dirt", "quantity": 6}])
        self.assertEqual(setup["init_commands"][0], "/gamerule sendCommandFeedback false")
        cmds = " | ".join(setup["init_commands"])
        for frag in ("/time set 18000", "doDaylightCycle false", "/weather rain", "summon minecraft:pig", "setblock"):
            self.assertIn(frag, cmds)

    def test_crafter_checks_recipe_platform_and_variants(self):
        env = FakeMinecraft(inventory={"birch_planks": 2, "oak_planks": 2, "stick": 2})
        crafter = CommandCrafter(env)
        pick = {"name": "craft", "args": {"obj": {"wooden pickaxe": 1}, "platform": "crafting table"}}
        self.assertIn("crafting table is required", crafter.plan(pick, env.inventory)["reason"])
        env.inventory["crafting_table"] = 1
        plan = crafter.plan(pick, env.inventory)
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["commands"][-1], "/give @p minecraft:wooden_pickaxe 1")
        res = crafter.craft(pick, env.observation() | {"inventory": dict(env.inventory)})
        self.assertTrue(res["ok"], res)
        self.assertEqual(mp5_inventory(env.inventory), {"planks": 1, "crafting table": 1, "wooden pickaxe": 1})
        self.assertIn("need 3 planks", crafter.plan(pick, env.inventory)["reason"])

    def test_pickaxe_chain(self):
        _env, perf = make_performer(crafter=None)
        perf.crafter = CommandCrafter(perf.env)
        check, _ = perf.check_and_execute_workflow(perf.env, DEMOS["pickaxe"], {}, False)
        self.assertTrue(check["success"], check)
        self.assertEqual(perf.records[-1]["inventory_after"].get("wooden pickaxe"), 1)
        self.assertEqual([r["executor"] for r in perf.records][-4:], ["command_craft"] * 4)


class RemoteTest(unittest.TestCase):
    def test_http_split_matches_local(self):
        env, perf = make_performer()
        server = PerformerServer(perf, host="127.0.0.1", port=0)
        server.serve_in_thread()
        seen_images = []
        local = gt_perceive(env)

        def perceive(image, obs, done):
            seen_images.append(None if image is None else image.shape)
            return local(None, done)

        try:
            client = RemotePerformer(f"http://127.0.0.1:{server.port}", perceive_fn=perceive)
            self.assertIn("pov_png_b64", client.observe())
            check, _ = client.check_and_execute_workflow(None, DEFAULT_WORKFLOW, {}, False)
        finally:
            server.shutdown()
        self.assertTrue(check["success"], check)
        self.assertEqual(client.records[-1]["inventory_delta"], {"log": 1})
        self.assertTrue(seen_images and seen_images[0] == (160, 256, 3))


if __name__ == "__main__":
    unittest.main()
