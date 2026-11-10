"""Summarize expert demo folders by task and difficulty."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from lfnav.data.demo_format import read_episode


def _size_bytes(folder: Path) -> int:
    total = 0
    for path in folder.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _numeric(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def collect_rows(root: Path) -> list[dict]:
    rows = []
    for rec_path in sorted(root.glob("**/record.json")):
        folder = rec_path.parent
        data = read_episode(folder)
        rec = data["record"]
        steps = data["steps"]
        n_wide = sum(1 for row in steps if row.get("used_wide"))
        n_act = sum(1 for row in steps if row.get("action_chunk") is not None)
        rows.append({
            "task": rec.get("task_name") or folder.parent.parent.name,
            "difficulty": rec.get("difficulty") or folder.parent.name,
            "success": bool(rec.get("success")),
            "spl": _numeric(rec.get("spl")),
            "steps": rec.get("steps"),
            "used_wide_n": n_wide,
            "used_wide_d": n_act,
            "natural_closer": bool(rec.get("natural_target_closer")),
            "n_frames": len(data["frames"]),
            "bytes": _size_bytes(folder),
        })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["difficulty"])].append(row)
    out = []
    for (task, diff), items in sorted(groups.items()):
        n = len(items)
        suc = sum(1 for r in items if r["success"])
        spls = [r["spl"] for r in items if r["spl"] is not None]
        step_vals = [int(r["steps"]) for r in items if r["steps"] is not None]
        wide_n = sum(r["used_wide_n"] for r in items)
        wide_d = sum(r["used_wide_d"] for r in items)
        closer = sum(1 for r in items if r["natural_closer"])
        out.append({
            "task": task,
            "difficulty": diff,
            "n": n,
            "success_rate": suc / n if n else 0.0,
            "spl": statistics.mean(spls) if spls else None,
            "median_steps": statistics.median(step_vals) if step_vals else None,
            "used_wide": wide_n / wide_d if wide_d else 0.0,
            "natural_target_closer": closer / n if n else 0.0,
            "n_frames": sum(r["n_frames"] for r in items),
            "bytes": sum(r["bytes"] for r in items),
        })
    return out


def render_md(stats: list[dict]) -> str:
    lines = [
        "# Demo stats",
        "",
        "| task | difficulty | n | success | SPL | median steps | used_wide | natural_closer | frames | size_MB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in stats:
        spl = "" if row["spl"] is None else f"{row['spl']:.3f}"
        med = "" if row["median_steps"] is None else f"{row['median_steps']:.0f}"
        lines.append(
            f"| {row['task']} | {row['difficulty']} | {row['n']} | {row['success_rate']:.2f} | {spl} | {med} | "
            f"{row['used_wide']:.2f} | {row['natural_target_closer']:.2f} | {row['n_frames']} | {row['bytes'] / 1e6:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    stats = summarize(collect_rows(Path(args.src)))
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_md(stats))
    dest.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(dest)


if __name__ == "__main__":
    main()
