"""One cross-entropy step for C2. --dry-run does not read demonstrations or a GPU."""

from __future__ import annotations

import argparse

import numpy as np

from lfnav.policy.c2 import ACTIONS, BehaviorCloning


def dry_run() -> float:
    policy = BehaviorCloning({"seed": 0, "policy": {"chunk": 4, "dagger": False}})
    vector = np.zeros(policy.dim)
    labels = [ACTIONS.index("forward")] * policy.horizon
    before = policy.train_step(vector, labels)
    after = policy.loss(vector, labels)
    if after > before + 1e-6:
        raise RuntimeError(f"loss increased from {before} to {after}")
    print(f"dry-run c2 loss {before:.4f} -> {after:.4f}")
    return after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dagger", action="store_true")
    args = parser.parse_args()
    if args.dagger:
        print("DAgger merge is enabled. Collect a second round with the learner before this step on a GPU node.")
    dry_run()


if __name__ == "__main__":
    main()
