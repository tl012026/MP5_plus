"""Map a Slurm array task id onto one (task, difficulty) cell."""

from __future__ import annotations

from pathlib import Path

DIFFICULTIES = ("easy", "medium", "hard")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def task_ids(root: Path | None = None) -> list[str]:
    specs = (root or repo_root()) / "lfnav" / "tasks" / "specs"
    return sorted(path.stem for path in specs.glob("*.json"))


def n_cells(root: Path | None = None) -> int:
    return len(task_ids(root)) * len(DIFFICULTIES)


def cell(index: int, root: Path | None = None) -> tuple[str, str]:
    ids = task_ids(root)
    total = len(ids) * len(DIFFICULTIES)
    index = int(index)
    if index < 0 or index >= total:
        raise IndexError(f"array index {index} is outside 0..{total - 1}")
    return ids[index // len(DIFFICULTIES)], DIFFICULTIES[index % len(DIFFICULTIES)]


def all_cells(root: Path | None = None) -> list[tuple[str, str]]:
    return [cell(i, root) for i in range(n_cells(root))]
