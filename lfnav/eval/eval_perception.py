"""Offline perception metrics: F1, hallucination, misses, direction, distance, ECE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_div(num, den) -> float:
    return float(num) / float(den) if den else 0.0


def binary_counts(pairs: list[tuple[int, int]]) -> dict:
    tp = fp = fn = tn = 0
    for pred, gt in pairs:
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and gt:
            fn += 1
        else:
            tn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "hallucination": _safe_div(fp, fp + tn),
        "miss": _safe_div(fn, fn + tp),
    }


def accuracy(pairs: list[tuple[str | None, str | None]]) -> float:
    if not pairs:
        return 0.0
    return _safe_div(sum(int(a == b) for a, b in pairs), len(pairs))


def expected_calibration_error(confidences: list[float], correct: list[int], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = 0.0
    n = len(confidences)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(acc - conf)
    return float(total)


def score_scenes(rows: list[dict], threshold: float = 0.5) -> dict:
    present_pairs = []
    direction_pairs = []
    distance_pairs = []
    confidences = []
    correct = []
    for row in rows:
        pred = row["pred"]["objects"]
        gt = row["gt"]["objects"]
        for name in gt:
            p = float(pred.get(name, {}).get("present", 0.0))
            g = int(float(gt[name].get("present", 0.0)) >= 0.5)
            present_pairs.append((int(p >= threshold), g))
            confidences.append(p)
            correct.append(int((p >= threshold) == bool(g)))
            if g:
                direction_pairs.append((_argmax(pred.get(name, {}).get("direction")), _argmax(gt[name].get("direction"))))
                distance_pairs.append((_argmax(pred.get(name, {}).get("distance")), _argmax(gt[name].get("distance"))))
    return {
        "present": binary_counts(present_pairs),
        "direction_accuracy": accuracy(direction_pairs),
        "distance_accuracy": accuracy(distance_pairs),
        "ece": expected_calibration_error(confidences, correct),
    }


def _argmax(table: dict | None) -> str | None:
    if not table:
        return None
    return max(table, key=table.get)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.data).read_text())
    report = score_scenes(rows)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
