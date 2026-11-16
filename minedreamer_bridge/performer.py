"""DreamerPerformer: the Performer role filled by MineDreamer.

Per MP5 action:
  1. instructions.instruction_for(action) -> text prompt, step budget, done-condition
  2. every `freq` env steps (MineDreamer default 25): Imaginator(current frame, prompt) -> goal
     frame, Prompt Generator -> goal embedding; perception done-conditions are checked here
  3. each env step: policy(obs, goal) -> MineRL action -> env.step; cheap done-conditions checked
  4. stop on done or budget; write the result file (schema.py) and return the record
  craft actions go to a CommandCrafter (mc116.py) when one is given, as MineDojo crafts by event.

The action loop is a generator (`action_steps`) that pauses whenever it needs perception
("is the object visible / within reach?"). execute_action() answers with perceive_fn locally;
remote.py answers over HTTP, so MP5's percipient can live in another process and Python env.

check_and_execute_workflow() keeps the MP5 Performer signature. Unlike the scripted performer
it stops at the first failed action and returns that failure to the planner
(stop_on_action_failure=True), because a learned controller fails often and silently.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path

from minedreamer_bridge import actions as A
from minedreamer_bridge.backends import GoalSpec, timed
from minedreamer_bridge.imageio import encode_png
from minedreamer_bridge.instructions import NOOP_ACTIONS, instruction_for, mp5_inventory
from minedreamer_bridge.schema import SCHEMA_ID, validate

PERCEPTION_KINDS = ("visible", "distance_le")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _pose(obs: dict):
    loc = (obs or {}).get("location_stats")
    if not loc:
        return None
    return {"pos": [round(float(v), 3) for v in loc.get("pos", [])],
            "yaw": round(float(loc.get("yaw", 0.0)), 2), "pitch": round(float(loc.get("pitch", 0.0)), 2)}


def _delta(before: dict, after: dict) -> dict:
    keys = set(before) | set(after)
    out = {k: after.get(k, 0) - before.get(k, 0) for k in keys}
    return {k: v for k, v in sorted(out.items()) if v}


def run_workflow(execute_action, workflow_dict, underground, stop_on_action_failure=True):
    """MP5 workflow semantics (times, mine early-exit, dig_down/dig_up) over any execute_action."""
    for step in workflow_dict.get("workflow") or []:
        times = int(step.get("times", 1))
        for _ in range(times):
            mine_done = False
            for action in step.get("actions") or []:
                rec = execute_action(action)
                if action.get("name") == "dig_down" and rec["outcome"] == "done":
                    underground = True
                if action.get("name") == "dig_up":
                    underground = False
                if not rec["check_result"]["success"] and stop_on_action_failure:
                    return rec["check_result"], underground
                if action.get("name") == "mine":
                    item = rec["instruction"]["done"].get("item")
                    if item and rec["inventory_after"].get(item, 0) >= times:
                        mine_done = True
            if mine_done:
                break
    return {"feedback": "", "success": True, "suggestion": ""}, underground


class DreamerPerformer:
    def __init__(self, env, imaginator, prompt_generator, policy, *, freq: int = 25,
                 budgets: dict | None = None, out_dir=None, perceive_fn=None, crafter=None,
                 memory=None, checker=None, event_logger=None, save_frames: bool = False,
                 stop_on_action_failure: bool = True, seed: int = 0, craft_planks_with_dreamer: bool = False):
        self.env = env
        self.imaginator = imaginator
        self.prompt_generator = prompt_generator
        self.policy = policy
        self.freq = int(freq)
        self.budgets = budgets or {}
        self.out_dir = Path(out_dir) if out_dir else None
        # perceive_fn(obs, done) -> (satisfied: bool, value): MP5 percipient in a real run,
        # ground truth in the fake world. done is {"kind": "visible"|"distance_le", "obj": ...}
        self.perceive_fn = perceive_fn
        self.crafter = crafter
        self.memory = memory
        self.checker = checker
        self.event_logger = event_logger
        self.save_frames = bool(save_frames)
        self.stop_on_action_failure = bool(stop_on_action_failure)
        self.seed = seed
        self.craft_planks_with_dreamer = bool(craft_planks_with_dreamer)
        self._index = 0
        self.records: list[dict] = []

    # ------------------------------------------------------------ helpers
    def _obs(self):
        getter = getattr(self.env, "observation", None)
        return getter() if callable(getter) else None

    def _backend(self) -> dict:
        return {
            "env": getattr(self.env, "name", type(self.env).__name__),
            "imaginator": getattr(self.imaginator, "name", type(self.imaginator).__name__),
            "prompt_generator": getattr(self.prompt_generator, "name", type(self.prompt_generator).__name__),
            "policy": getattr(self.policy, "name", type(self.policy).__name__),
            "crafter": getattr(self.crafter, "name", None),
            "freq": self.freq,
            "seed": self.seed,
        }

    def _cheap_done(self, done: dict, obs: dict, inv_before: dict):
        kind = done.get("kind")
        if kind == "inventory_gain":
            value = mp5_inventory(obs.get("inventory")).get(done["item"], 0) - inv_before.get(done["item"], 0)
            return value >= int(done.get("count", 1)), value
        if kind == "y_le":
            y = (_pose(obs) or {}).get("pos", [None, None])[1]
            if done.get("y") is None or y is None:
                return False, y
            return y <= float(done["y"]), y
        return False, None

    def _frame_path(self, kind: str, step: int) -> str | None:
        if not (self.save_frames and self.out_dir):
            return None
        return f"frames/{self._index:04d}_{kind}_{step:04d}.png"

    def _save_png(self, rel: str | None, img) -> None:
        if rel is None:
            return
        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encode_png(img))

    def _new_record(self, action, ins, obs, inv_before) -> dict:
        return {
            "schema": SCHEMA_ID, "created": _now(), "action": action, "instruction": ins.to_dict(),
            "backend": self._backend(), "executor": "minedreamer", "outcome": None, "steps": 0,
            "imagine_calls": 0, "imagine_latency_s": {"total": 0.0, "max": 0.0},
            "inventory_before": inv_before, "inventory_after": inv_before, "inventory_delta": {},
            "pose_before": _pose(obs), "pose_after": _pose(obs),
            "done_check": {"kind": ins.done.get("kind"), "satisfied": False, "value": None, "checked_at_step": None},
            "imaginations": [], "commands": [], "trace_file": None,
            "check_result": {"success": False, "feedback": "", "suggestion": ""}, "error": None,
        }

    # ------------------------------------------------------------ one action
    def execute_action(self, action: dict) -> dict:
        gen = self.action_steps(action)
        try:
            request = next(gen)
            while True:
                _tag, obs, done = request
                answer = self.perceive_fn(obs, done) if self.perceive_fn else (False, None)
                request = gen.send(tuple(answer))
        except StopIteration as stop:
            return stop.value

    def action_steps(self, action: dict):
        """Generator. Yields ("perceive", obs, done) and expects (satisfied, value) back. Returns the record."""
        ins = instruction_for(action, self.budgets)
        name = action.get("name")
        obs = self._obs()
        inv_before = mp5_inventory((obs or {}).get("inventory"))
        record = self._new_record(action, ins, obs, inv_before)

        if self.checker is not None and self.memory is not None:
            prep = self.checker.check_action_preparation(name, self.memory.inventory, action.get("args") or {})
            if not prep["success"]:
                record["outcome"] = "precheck_fail"
                record["check_result"] = {k: prep.get(k, "") for k in ("success", "feedback", "suggestion")}
                return self._finish(record)

        use_crafter = name == "craft" and self.crafter is not None and not (
            self.craft_planks_with_dreamer and ins.supported)
        if use_crafter:
            return self._finish(self._command_craft(action, record, obs, inv_before))

        if not ins.supported:
            noop = name in NOOP_ACTIONS
            record["executor"] = "none"
            record["outcome"] = "noop" if noop else "unsupported"
            record["check_result"] = {
                "success": noop, "feedback": ins.note,
                "suggestion": "" if noop else f"Run {name} with the scripted MP5 performer or give a CommandCrafter.",
            }
            return self._finish(record)

        trace_rows, lat, goal = [], [], None
        try:
            if hasattr(self.policy, "reset"):
                self.policy.reset()
            for t in range(ins.max_steps):
                tick = t % self.freq == 0
                if ins.done.get("kind") in PERCEPTION_KINDS:
                    ok, value = (yield ("perceive", obs, ins.done)) if tick else (False, None)
                else:
                    ok, value = self._cheap_done(ins.done, obs, inv_before)
                if ok:
                    record["done_check"] = {"kind": ins.done["kind"], "satisfied": True, "value": value, "checked_at_step": t}
                    break
                if tick:
                    goals, dt = timed(self.imaginator.imagine, ins.text_prompt, obs["pov"])
                    embed = self.prompt_generator.embed(ins.text_prompt, obs["pov"], goals[0])
                    goal = GoalSpec(ins.text_prompt, t, goals[0], embed, dt)
                    lat.append(dt)
                    cur_rel, goal_rel = self._frame_path("current", t), self._frame_path("goal", t)
                    self._save_png(cur_rel, obs["pov"])
                    self._save_png(goal_rel, goals[0])
                    record["imaginations"].append({"step": t, "current_image": cur_rel, "goal_image": goal_rel,
                                                   "latency_s": round(dt, 4)})
                act = self.policy.act(obs, goal)
                obs, _r, _d, info = self.env.step(act)
                pose = _pose(obs) or {}
                trace_rows.append({"t": t, "minerl": A.compact(act), "minedojo": A.to_minedojo(act),
                                   "pos": pose.get("pos"), "yaw": pose.get("yaw"), "pitch": pose.get("pitch"),
                                   "event": (info or {}).get("event")})
                record["steps"] = t + 1
            else:
                if ins.done.get("kind") in PERCEPTION_KINDS:
                    ok, value = yield ("perceive", obs, ins.done)
                else:
                    ok, value = self._cheap_done(ins.done, obs, inv_before)
                record["done_check"] = {"kind": ins.done["kind"], "satisfied": bool(ok), "value": value,
                                        "checked_at_step": ins.max_steps}
        except Exception as exc:  # keep the file even when a backend dies
            record["outcome"] = "error"
            record["error"] = repr(exc)

        self._close(record, obs, inv_before, lat)
        record["check_result"] = self._feedback(ins, record)
        if self.out_dir:
            record["trace_file"] = f"{self._index:04d}_{name}.trace.jsonl"
            path = self.out_dir / record["trace_file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as fh:
                for row in trace_rows:
                    fh.write(json.dumps(row) + "\n")
        return self._finish(record)

    def _close(self, record, obs, inv_before, lat):
        inv_after = mp5_inventory((obs or {}).get("inventory"))
        record["inventory_after"] = inv_after
        record["inventory_delta"] = _delta(inv_before, inv_after)
        record["pose_after"] = _pose(obs)
        record["imagine_calls"] = len(lat)
        record["imagine_latency_s"] = {"total": round(sum(lat), 4), "max": round(max(lat), 4) if lat else 0.0}
        if record["outcome"] is None:
            record["outcome"] = "done" if record["done_check"]["satisfied"] else "budget_exhausted"
        if self.memory is not None:
            self.memory.update_inventory(inv_after)

    def _command_craft(self, action, record, obs, inv_before):
        record["executor"] = self.crafter.name
        record["instruction"]["note"] = "Crafted by recipe-checked chat commands (MineDojo crafts by event as well)."
        try:
            res = self.crafter.craft(action, obs)
        except Exception as exc:
            record["outcome"], record["error"] = "error", repr(exc)
            res = {"ok": False, "reason": repr(exc), "commands": [], "obs": obs}
        record["commands"] = res["commands"]
        record["done_check"] = {"kind": "inventory_gain", "satisfied": bool(res["ok"]), "value": None, "checked_at_step": 0}
        self._close(record, res.get("obs") or obs, inv_before, [])
        if res["ok"]:
            record["check_result"] = {"success": True, "feedback": f"Crafted {record['inventory_delta']}.", "suggestion": ""}
        else:
            record["check_result"] = {"success": False, "feedback": f"Craft failed: {res['reason']}.",
                                      "suggestion": "Collect the missing materials or platform first."}
        return record

    def _feedback(self, ins, record) -> dict:
        prompt = ins.text_prompt
        if record["outcome"] == "done":
            return {"success": True, "feedback": f"MineDreamer finished '{prompt}' in {record['steps']} steps.",
                    "suggestion": ""}
        if record["outcome"] == "error":
            return {"success": False, "feedback": f"MineDreamer backend error during '{prompt}': {record['error']}",
                    "suggestion": "Check the Imaginator server and the STEVE-1 weights."}
        done = ins.done
        why = {
            "inventory_gain": f"{done.get('item')} did not increase by {done.get('count')}",
            "visible": f"{done.get('obj')} was not seen",
            "distance_le": f"did not get within {done.get('blocks')} blocks of {done.get('obj')}",
            "y_le": f"did not reach y <= {done.get('y')}",
        }.get(done.get("kind"), "done-condition not met")
        hint = {
            "mine": f"Use find({done.get('item')}) and move_to before mine, or plan a different source.",
            "find": "Retry find; MineDreamer explores at random without a find prompt.",
            "move_to": "Retry find first so the object is in view.",
        }.get(ins.action, "")
        return {"success": False,
                "feedback": f"MineDreamer ran '{prompt}' for {record['steps']} steps but {why}.",
                "suggestion": hint}

    def _finish(self, record: dict) -> dict:
        problems = validate(record)
        if problems:
            raise AssertionError(f"result does not match {SCHEMA_ID}: {problems}")
        if self.out_dir:
            name = (record.get("action") or {}).get("name")
            _atomic_write_json(self.out_dir / f"{self._index:04d}_{name}.json", record)
        if self.event_logger is not None:
            self.event_logger.emit("action", name=record["action"].get("name"), args=record["action"].get("args"),
                                   start_step=None, end_step=None, result=record["outcome"], backend="minedreamer",
                                   steps=record["steps"], imagine_calls=record["imagine_calls"])
        self.records.append(record)
        self._index += 1
        return record

    # ------------------------------------------------------------ MP5 API
    def check_and_execute_workflow(self, env, workflow_dict, task_information, underground):
        self.env = env if env is not None else self.env
        return run_workflow(self.execute_action, workflow_dict, underground, self.stop_on_action_failure)
