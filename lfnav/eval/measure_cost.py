"""Parameter and latency notes for the fake encoder and C2.

GPU memory is reported only when torch.cuda is available. This login node records
CPU latency. The scatter plot is an SVG so it does not need matplotlib.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lfnav.perception.encoder import FakeEncoder
from lfnav.policy.c2 import BehaviorCloning
from lfnav.types import PolicyInput


def _percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[idx])


def time_call(fn, repeats: int = 20) -> dict:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {"mean_s": float(sum(samples) / len(samples)), "p95_s": _percentile(samples, 0.95), "n": repeats}


def svg_scatter(points: list[tuple[float, float]], path: Path) -> None:
    width, height = 480, 320
    if not points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
        return
    xs = [math_log(max(p[0], 1e-6)) for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    span_x = max(max_x - min_x, 1e-6)
    dots = []
    for x, y in zip(xs, ys):
        px = 40 + (x - min_x) / span_x * (width - 80)
        py = height - 40 - y * (height - 80)
        dots.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='4' fill='#8ab4ff'/>")
    path.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='480' height='320'>"
        "<rect width='100%' height='100%' fill='#111'/>"
        "<text x='20' y='24' fill='#eee' font-size='14'>success vs log decision time</text>"
        + "".join(dots) + "</svg>"
    )


def math_log(value: float) -> float:
    import math
    return math.log(value)


def measure() -> dict:
    encoder = FakeEncoder()
    frame = np.zeros((28, 42, 3), dtype=np.uint8)
    policy = BehaviorCloning({"seed": 0})
    pin = PolicyInput()
    report = {
        "fake_encoder": {"parameters": int(encoder.dim), **time_call(lambda: encoder(frame))},
        "c2": {"parameters": int(policy.n_parameters()), **time_call(lambda: policy.act(pin))},
        "gpu": None,
    }
    try:
        import torch
        if torch.cuda.is_available():
            report["gpu"] = {"name": torch.cuda.get_device_name(0)}
    except Exception:
        report["gpu"] = None
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = measure()
    (out / "cost.json").write_text(json.dumps(report, indent=2))
    svg_scatter(
        [
            (report["fake_encoder"]["mean_s"], 0.0),
            (report["c2"]["mean_s"], 0.0),
        ],
        out / "success_latency.svg",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
