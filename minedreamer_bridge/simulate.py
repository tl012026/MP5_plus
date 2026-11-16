"""Simulate MineDreamer-as-Performer before Minecraft is opened.

    python -m minedreamer_bridge.simulate --out /scratch/$USER/mp5_runs/dreamer_sim/demo
    python -m minedreamer_bridge.simulate --demo pickaxe --remote      # brain and body in two "processes"
    python -m minedreamer_bridge.simulate --action '{"name": "mine", "args": {"obj": "wood", "tool": null}}'
    python -m minedreamer_bridge.simulate --workflow my_workflow.json --save-frames

--demo log      first subgoal of craft_wooden_pickaxe: find(wood) -> move_to(wood) -> mine(wood)
--demo pickaxe  the whole chain: 3 logs, then planks, sticks, crafting table, wooden pickaxe
--remote        runs the body behind remote.PerformerServer on localhost and drives it with
                RemotePerformer, the same split used when MP5 (MC 1.11 stack) and MineRL (1.16) differ
Env and models are mocks unless --imaginator-url points at a running Imaginator server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from minedreamer_bridge.backends import HttpImaginator, MockImaginator, MockPromptGenerator, MockSteve1
from minedreamer_bridge.fake_world import FakeMinecraft
from minedreamer_bridge.mc116 import CommandCrafter
from minedreamer_bridge.performer import DreamerPerformer

_GET_LOG = [
    {"name": "find", "args": {"obj": "wood"}},
    {"name": "move_to", "args": {"obj": "wood"}},
    {"name": "mine", "args": {"obj": "wood", "tool": None}},
]

DEMOS = {
    "log": {"explanation": "", "thoughts": "Find a tree, walk to it, mine one log.",
            "workflow": [{"times": "1", "actions": _GET_LOG}]},
    "pickaxe": {"explanation": "", "thoughts": "Three logs, then the wooden pickaxe chain.",
                "workflow": [
                    {"times": "3", "actions": _GET_LOG},
                    {"times": "1", "actions": [
                        {"name": "craft", "args": {"obj": {"planks": 12}, "materials": {"log": 3}, "platform": None}},
                        {"name": "craft", "args": {"obj": {"stick": 4}, "materials": {"planks": 2}, "platform": None}},
                        {"name": "craft", "args": {"obj": {"crafting table": 1}, "materials": {"planks": 4}, "platform": None}},
                        {"name": "craft", "args": {"obj": {"wooden pickaxe": 1}, "materials": {"planks": 3, "stick": 2},
                                                   "platform": "crafting table"}},
                    ]},
                ]},
}
DEFAULT_WORKFLOW = DEMOS["log"]


def gt_perceive(env):
    """Ground-truth answers from the fake world, standing in for MP5's percipient."""
    def perceive(obs, done):
        if done["kind"] == "visible":
            seen = env.visible(done["obj"])
            return seen, seen
        dist = env.distance(done["obj"])
        return (dist is not None and dist <= float(done["blocks"])), dist
    return perceive


def build(args):
    env = FakeMinecraft(seed=args.seed, n_trees=args.trees)
    env.reset()
    imaginator = HttpImaginator(args.imaginator_url) if args.imaginator_url else MockImaginator()
    performer = DreamerPerformer(
        env, imaginator, MockPromptGenerator(), MockSteve1(seed=args.seed),
        freq=args.freq, out_dir=args.out, save_frames=args.save_frames, seed=args.seed,
        perceive_fn=gt_perceive(env), crafter=None if args.no_crafter else CommandCrafter(env),
        budgets={k: args.max_steps for k in ("find", "move_to", "mine", "dig_down", "craft")} if args.max_steps else None,
    )
    return env, performer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="mp5_runs/dreamer_sim", help="result directory (one JSON per action)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--action", help="one MP5 action as JSON")
    group.add_argument("--workflow", help="a planner workflow JSON file")
    group.add_argument("--demo", choices=sorted(DEMOS), default="log")
    ap.add_argument("--remote", action="store_true", help="drive the body through the HTTP split")
    ap.add_argument("--no-crafter", action="store_true", help="do not craft by commands")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trees", type=int, default=3)
    ap.add_argument("--freq", type=int, default=25, help="imagine every N env steps (MineDreamer default 25)")
    ap.add_argument("--max-steps", type=int, default=None, help="override every per-action budget")
    ap.add_argument("--imaginator-url", default=None, help="e.g. http://10.0.0.5:25547/ ; default is the mock")
    ap.add_argument("--save-frames", action="store_true", help="write current/goal PNGs at each imagination")
    args = ap.parse_args(argv)
    args.out = Path(args.out)

    env, performer = build(args)
    runner, server = performer, None
    if args.remote:
        from minedreamer_bridge.remote import PerformerServer, RemotePerformer
        server = PerformerServer(performer, host="127.0.0.1", port=0)
        server.serve_in_thread()
        perceive = gt_perceive(env)
        runner = RemotePerformer(f"http://127.0.0.1:{server.port}",
                                 perceive_fn=lambda image, obs, done: perceive(None, done))
    try:
        if args.action:
            records = [runner.execute_action(json.loads(args.action))]
            check = records[0]["check_result"]
        else:
            workflow = json.loads(Path(args.workflow).read_text()) if args.workflow else DEMOS[args.demo]
            check, _underground = runner.check_and_execute_workflow(env, workflow, {}, False)
            records = runner.records
    finally:
        if server is not None:
            server.shutdown()

    print(f"{'#':>2}  {'action':<9} {'executor':<13} {'prompt':<16} {'outcome':<16} {'steps':>5} {'imag':>4}  delta")
    for i, rec in enumerate(records):
        ins = rec["instruction"]
        print(f"{i:>2}  {rec['action']['name']:<9} {rec.get('executor', ''):<13} {str(ins['text_prompt']):<16} "
              f"{rec['outcome']:<16} {rec['steps']:>5} {rec['imagine_calls']:>4}  {rec['inventory_delta']}")
    print("final inventory ->", records[-1]["inventory_after"] if records else {})
    print("check_result ->", json.dumps(check, ensure_ascii=False))
    print("files ->", args.out.resolve(), "(remote)" if args.remote else "")
    return 0 if check.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
