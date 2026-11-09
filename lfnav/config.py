"""YAML config loader. Paths may contain ${SCR}, ${NETID}, ${HOME}."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value


def deep_merge(base: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    inherit = data.pop("inherit", None)
    if inherit:
        parent = Path(os.path.expandvars(str(inherit)))
        if not parent.is_file():
            cand = path.parent / inherit
            parent = cand if cand.is_file() else repo_root() / inherit
        base = load_yaml(parent)
        data = deep_merge(base, data)
    data = expand(data)
    data["_path"] = str(path.resolve())
    return data


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    return repo_root() / "configs" / "lfnav" / "default.yaml"


def load_config(path=None) -> dict:
    cfg = load_yaml(path or default_config_path())
    root = repo_root()
    files = {
        "labels": "labels_file",
        "difficulty_table": "difficulty_file",
        "obs_keys": "obs_keys_file",
        "actions": "actions_file",
    }
    for dest, src in files.items():
        rel = cfg.get(src)
        if not rel:
            continue
        p = Path(rel)
        if not p.is_absolute():
            p = root / p
        cfg[dest] = load_yaml(p)
    return cfg
