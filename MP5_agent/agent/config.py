"""YAML config loader with defaults and validation. No API keys are stored here."""

from __future__ import annotations

import copy
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "code_default.yaml"

_MODEL_KEYS = ("planner", "patroller", "memory", "percipient_question")
_MODEL_FIELDS = ("model_name", "base_url", "temperature")
_PERCIPIENT_FIELDS = ("answer_method", "answer_model", "mllm_url", "gpt_vision_model", "gpt_vision_base_url")
_EVAL_FIELDS = ("max_steps", "max_subgoal_retries", "save_frames", "save_perception_images")
_ENV_REQUIRED = (
    "task_id", "target_names", "target_quantities", "image_size", "env_seed",
    "specified_biome", "spawn_rate", "break_speed_multiplier",
    "spawn_range_low", "spawn_range_high", "start_at_night", "world_seed",
    "use_voxel", "voxel_radius", "use_lidar", "lidar_pitch", "lidar_yaw",
    "lidar_range", "start_inventory", "warmup_action",
)


def _load_yaml(path):
    path = Path(path)
    text = path.read_text()
    try:
        import yaml
        data = yaml.safe_load(text)
    except ImportError:
        data = parse_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse as a mapping")
    return data


def _parse_scalar(token):
    token = token.strip()
    if token in ("null", "~", ""):
        return None
    if token in ("true", "True"):
        return True
    if token in ("false", "False"):
        return False
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(p) for p in inner.split(",")]
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        if not inner:
            return {}
        out = {}
        for part in inner.split(","):
            k, v = part.split(":", 1)
            out[k.strip()] = _parse_scalar(v)
        return out
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def parse_simple_yaml(text):
    """Indent-based subset: mappings, dashed lists, JSON-ish scalars. No anchors."""
    raw = []
    for line in text.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        if line.strip():
            raw.append(line.rstrip())

    def indent(i):
        return len(raw[i]) - len(raw[i].lstrip())

    def parse_block(i, min_indent):
        mapping, seq = {}, []
        mode = None
        while i < len(raw):
            ind = indent(i)
            if ind < min_indent:
                break
            stripped = raw[i].lstrip()
            if stripped.startswith("- "):
                if mode == "map":
                    raise ValueError(f"mixed list/map at: {raw[i]}")
                mode = "list"
                rest = stripped[2:]
                if ":" in rest and not rest.startswith("{") and not rest.startswith("["):
                    key, _, val = rest.partition(":")
                    item = {key.strip(): _parse_scalar(val)}
                    if i + 1 < len(raw) and indent(i + 1) > ind:
                        child, i = parse_block(i + 1, indent(i + 1))
                        if isinstance(child, dict):
                            item.update(child)
                    else:
                        i += 1
                    seq.append(item)
                else:
                    seq.append(_parse_scalar(rest))
                    i += 1
                continue
            if ":" not in stripped:
                raise ValueError(f"cannot parse: {raw[i]}")
            if mode == "list":
                break
            mode = "map"
            key, _, val = stripped.partition(":")
            key, val = key.strip(), val.strip()
            if val:
                mapping[key] = _parse_scalar(val)
                i += 1
            else:
                if i + 1 >= len(raw) or indent(i + 1) <= ind:
                    mapping[key] = None
                    i += 1
                else:
                    child, i = parse_block(i + 1, indent(i + 1))
                    mapping[key] = child
        return (seq if mode == "list" else mapping), i

    data, _ = parse_block(0, 0)
    return data


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = _load_yaml(path)
    validate_config(cfg)
    cfg["_path"] = str(path.resolve())
    return cfg


def validate_config(cfg):
    if "models" not in cfg:
        raise ValueError("config missing models")
    for role in _MODEL_KEYS:
        if role not in cfg["models"]:
            raise ValueError(f"config.models missing {role}")
        block = cfg["models"][role]
        for f in _MODEL_FIELDS:
            if f not in block:
                raise ValueError(f"config.models.{role} missing {f}")
        if not block["model_name"]:
            raise ValueError(f"config.models.{role}.model_name is empty")
    if "percipient" not in cfg:
        raise ValueError("config missing percipient")
    for f in _PERCIPIENT_FIELDS:
        if f not in cfg["percipient"]:
            raise ValueError(f"config.percipient missing {f}")
    if cfg["percipient"]["answer_method"] not in ("active", "caption"):
        raise ValueError("percipient.answer_method must be active or caption")
    if cfg["percipient"]["answer_model"] not in ("mllm", "gpt-vision"):
        raise ValueError("percipient.answer_model must be mllm or gpt-vision")
    if "env" not in cfg:
        raise ValueError("config missing env")
    for f in _ENV_REQUIRED:
        if f not in cfg["env"]:
            raise ValueError(f"config.env missing {f}")
    if "eval" not in cfg:
        raise ValueError("config missing eval")
    for f in _EVAL_FIELDS:
        if f not in cfg["eval"]:
            raise ValueError(f"config.eval missing {f}")
    ev = cfg["eval"]
    if int(ev["max_steps"]) <= 0:
        raise ValueError("eval.max_steps must be positive")
    if int(ev["max_subgoal_retries"]) < 0:
        raise ValueError("eval.max_subgoal_retries must be >= 0")
    return cfg


def resolve_api_key():
    """API key only from the environment. Raises a clear error if missing."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it (or source a 600-permission file "
            "that exports it). Keys are not read from config or the repo."
        )
    return key


def resolve_base_url(config_value):
    env_url = os.environ.get("OPENAI_BASE_URL")
    if env_url:
        return env_url
    return config_value
