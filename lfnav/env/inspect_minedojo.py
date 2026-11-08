"""Print a MineDojo meta-task constructor signature. Does not call make() or reset().

Run this inside the MineDojo image, after the task id in configs/lfnav/default.yaml
is the one you intend to use:

    python lfnav/env/inspect_minedojo.py
    python lfnav/env/inspect_minedojo.py --task-id harvest
"""

from __future__ import annotations

import argparse
import importlib
import inspect


def find_task_class(task_id: str):
    """Return the class MineDojo registers for task_id. Imports minedojo."""
    tasks = importlib.import_module("minedojo.tasks")
    registry = getattr(tasks, "MetaTaskName2Class", None)
    if registry and task_id.lower() in registry:
        return registry[task_id.lower()]
    for name in ("_get_task_cls", "get_task_cls"):
        finder = getattr(tasks, name, None)
        if callable(finder):
            return finder(task_id)
    packages = [tasks]
    for sub in ("meta", "programmatic"):
        try:
            packages.append(importlib.import_module(f"minedojo.tasks.{sub}"))
        except ImportError:
            continue
    for package in packages:
        for obj in vars(package).values():
            if not inspect.isclass(obj):
                continue
            ids = getattr(obj, "task_ids", None)
            if ids is None:
                ids = getattr(obj, "task_id", None)
            if ids is None:
                continue
            if isinstance(ids, str):
                ids = [ids]
            if task_id in list(ids):
                return obj
    raise LookupError(
        f"No MineDojo task class declares task id {task_id!r}. "
        "Check minedojo.tasks for the meta-task registry."
    )


def format_signature(cls) -> str:
    sig = inspect.signature(cls.__init__)
    lines = [f"{cls.__module__}.{cls.__qualname__}{sig}"]
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty:
            lines.append(f"  {name}")
        else:
            lines.append(f"  {name} = {param.default!r}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print a MineDojo task constructor signature.")
    parser.add_argument("--task-id", default=None, help="defaults to minedojo.task_id in configs/lfnav/default.yaml")
    args = parser.parse_args(argv)
    task_id = args.task_id
    if not task_id:
        from lfnav.config import load_config
        task_id = (load_config().get("minedojo") or {}).get("task_id")
    if not task_id:
        raise SystemExit("Pass --task-id or set minedojo.task_id.")
    cls = find_task_class(task_id)
    print(format_signature(cls))


if __name__ == "__main__":
    main()
