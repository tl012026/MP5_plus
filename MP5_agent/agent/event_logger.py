"""Per-episode JSONL event logger. Never writes API keys."""

from __future__ import annotations

import json
import time
from pathlib import Path


_REDACT_KEYS = ("openai_key", "api_key", "OPENAI_API_KEY", "authorization", "Authorization")


def _redact(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if any(s.lower() in str(k).lower() for s in _REDACT_KEYS):
                out[k] = "<redacted>"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    if isinstance(obj, str) and obj.startswith("sk-") and len(obj) > 20:
        return "<redacted>"
    return obj


class EventLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.counts = {
            "planner_call": 0,
            "perception_call": 0,
            "action": 0,
            "patroller": 0,
            "subgoal": 0,
        }

    def emit(self, event_type, **fields):
        rec = {"ts": time.time(), "event": event_type}
        rec.update(_redact(fields))
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        if event_type in self.counts:
            self.counts[event_type] += 1

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class NullLogger:
    counts = {}

    def emit(self, event_type, **fields):
        pass

    def close(self):
        pass
