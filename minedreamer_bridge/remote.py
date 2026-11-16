"""Process split for the version gap: MP5's brain and the 1.16 body talk JSON over HTTP.

  [MP5 process]  MineDojo image, py3.9, langchain 0.0.300      RemotePerformer  ─┐
     planner / patroller / memory / percipient (MineLLM or API)                 │ HTTP
  [body process] MineRL 1.0 env (MC 1.16.5) + STEVE-1 + CVAE    PerformerServer ◄┘
  [imaginator]   MineDreamer Imaginator server :25547 (called by the body process)

The two Python stacks never share an interpreter, so their pinned dependencies cannot clash.

Endpoints (all JSON):
  POST /reset            {"setup": mc116.task_setup(...)}            -> observation
  GET  /observe                                                        -> observation
  POST /action/start     {"action": {...}}                             -> status
  POST /action/continue  {"perception": [satisfied, value]}            -> status
observation = {"inventory": MP5 names, "minerl_inventory": raw, "pose": ..., "pov_png_b64": ...}
status      = {"state": "need_perception", "done": {...}, "observation": ...}
            | {"state": "finished", "record": <schema v1 record>}
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from minedreamer_bridge.imageio import decode_png, encode_png
from minedreamer_bridge.instructions import mp5_inventory
from minedreamer_bridge.performer import _pose, run_workflow


def observation_payload(obs: dict | None, with_image: bool = True) -> dict:
    obs = obs or {}
    out = {"inventory": mp5_inventory(obs.get("inventory")),
           "minerl_inventory": {k: int(v) for k, v in (obs.get("inventory") or {}).items()},
           "pose": _pose(obs)}
    if with_image and obs.get("pov") is not None:
        out["pov_png_b64"] = base64.b64encode(encode_png(obs["pov"])).decode()
    return out


class PerformerServer:
    """Owns the env and a DreamerPerformer; one action at a time."""

    def __init__(self, performer, reset_fn=None, host: str = "0.0.0.0", port: int = 25560):
        self.performer = performer
        self.reset_fn = reset_fn            # reset_fn(setup) -> env ; default: performer.env.reset()
        self._gen = None
        self._lock = threading.Lock()
        self.httpd = ThreadingHTTPServer((host, port), self._handler())

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def _status(self, step):
        try:
            _tag, obs, done = step()
            return {"state": "need_perception", "done": done, "observation": observation_payload(obs)}
        except StopIteration as stop:
            self._gen = None
            return {"state": "finished", "record": stop.value}

    def handle(self, method: str, path: str, body: dict) -> dict:
        with self._lock:
            if method == "POST" and path == "/reset":
                if self.reset_fn is not None:
                    self.performer.env = self.reset_fn(body.get("setup") or {})
                obs = self.performer.env.reset()
                self._gen = None
                return observation_payload(obs)
            if method == "GET" and path == "/observe":
                return observation_payload(self.performer.env.observation())
            if method == "POST" and path == "/action/start":
                if self._gen is not None:
                    raise RuntimeError("an action is still running; finish it with /action/continue")
                self._gen = self.performer.action_steps(body["action"])
                return self._status(lambda: next(self._gen))
            if method == "POST" and path == "/action/continue":
                if self._gen is None:
                    raise RuntimeError("no running action")
                answer = tuple(body.get("perception") or (False, None))
                return self._status(lambda: self._gen.send(answer))
        raise KeyError(f"{method} {path}")

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, code, payload):
                data = json.dumps(payload, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _do(self, method):
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                    self._reply(200, server.handle(method, self.path, body))
                except KeyError as exc:
                    self._reply(404, {"error": str(exc)})
                except Exception as exc:
                    self._reply(500, {"error": repr(exc)})

            def do_GET(self):
                self._do("GET")

            def do_POST(self):
                self._do("POST")

            def log_message(self, *_args):
                pass

        return Handler

    def serve_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        thread.start()
        return thread

    def shutdown(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class RemotePerformer:
    """MP5-side client with the Performer signature. perceive_fn(image_hwc, observation, done) -> (bool, value)."""

    def __init__(self, url: str, perceive_fn=None, memory=None, stop_on_action_failure=True, timeout=3600):
        self.url = url.rstrip("/")
        self.perceive_fn = perceive_fn
        self.memory = memory
        self.stop_on_action_failure = stop_on_action_failure
        self.timeout = timeout
        self.records: list[dict] = []

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def reset(self, setup: dict) -> dict:
        return self._call("POST", "/reset", {"setup": setup})

    def observe(self) -> dict:
        return self._call("GET", "/observe")

    @staticmethod
    def image(observation: dict):
        b64 = observation.get("pov_png_b64")
        return None if b64 is None else decode_png(base64.b64decode(b64))

    def execute_action(self, action: dict) -> dict:
        status = self._call("POST", "/action/start", {"action": action})
        while status["state"] == "need_perception":
            obs = status["observation"]
            answer = self.perceive_fn(self.image(obs), obs, status["done"]) if self.perceive_fn else (False, None)
            status = self._call("POST", "/action/continue", {"perception": list(answer)})
        record = status["record"]
        if self.memory is not None:
            self.memory.update_inventory(record["inventory_after"])
        self.records.append(record)
        return record

    def check_and_execute_workflow(self, env, workflow_dict, task_information, underground):
        return run_workflow(self.execute_action, workflow_dict, underground, self.stop_on_action_failure)
