"""Minimal C2Nav login-node self-check: C1 labels, C3 init losses, C8 overfit."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "4")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from lfnav.config import load_config
from lfnav.policy.c2_data import (
    _episode_key,
    _is_legacy_source,
    _is_main_source,
    demo_roots,
    iter_success_episodes,
)
from lfnav.policy.c2nav import C2NavNet, patch_bearings, spec_from_cfg
from lfnav.policy.c2nav_labels import FORWARD_BINS, MODE_NAMES, N_TURN, TURN_BINS, branch_is_mapped, labels_from_step, mode_from_branch
from lfnav.policy.train_c2nav import (
    VAL_SEEDS_DEFAULT,
    class_weights,
    collate,
    heads_out,
    nav_loss,
    samples_from_episode,
    stop_pos_weight,
    NavSample,
    _to_torch,
)


class ItemTimeout(Exception):
    pass


def _alarm(sec: int):
    def deco(fn):
        def wrapped(*args, **kwargs):
            def handler(_s, _f):
                raise ItemTimeout(f"{fn.__name__} exceeded {sec}s")

            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(sec))
            try:
                return fn(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)

        return wrapped

    return deco


def _resolve_features(args, cfg) -> Path | None:
    if args.features:
        return Path(args.features)
    env = os.environ.get("LFNAV_FEATURES")
    if env:
        return Path(env)
    version = str((cfg.get("c2nav") or {}).get("features_version") or "")
    user = os.environ.get("USER", "user")
    root = os.environ.get("LFNAV_FEATURE_ROOT") or os.environ.get("SCR") or str(Path("/scratch") / user)
    if version:
        return Path(root) / "lfnav" / "features" / version
    return None


def _read_record_steps(folder: Path) -> tuple[dict, list[dict]]:
    record = json.loads((folder / "record.json").read_text())
    steps = []
    path = folder / "steps.jsonl"
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return record, steps


def _keep_episode(record: dict, folder: Path, cfg: dict) -> bool:
    c2 = cfg.get("c2") or {}
    nav = cfg.get("c2nav") or {}
    exclude = {str(t) for t in (c2.get("exclude_tasks") or ["4-3"])}
    allowed = nav.get("train_difficulties") or c2.get("train_difficulties")
    allowed_diff = {str(x) for x in allowed} if allowed else None
    legacy_tasks = {str(t) for t in (c2.get("legacy_tasks") or [])}
    if not record.get("success"):
        return False
    if record.get("difficulty_unmet"):
        return False
    task_id, difficulty, _seed = _episode_key(record, folder)
    if allowed_diff is not None and difficulty not in allowed_diff:
        return False
    if task_id in exclude:
        return False
    if _is_main_source(record, folder):
        if int(record.get("hostile_seen") or 0) != 0:
            return False
        return True
    if _is_legacy_source(record) and task_id in legacy_tasks:
        return True
    return False


def _episode_folders(cfg: dict) -> tuple[list[Path], list[Path]]:
    nav = cfg.get("c2nav") or {}
    roots = demo_roots(nav.get("demo_dirs"), {"c2": {"demo_dirs": nav.get("demo_dirs")}}) if nav.get("demo_dirs") else demo_roots(None, cfg)
    folders = []
    for root in roots:
        if Path(root).is_dir():
            folders.extend(iter_success_episodes(root))
    val_seeds = {int(s) for s in (nav.get("val_seeds") or VAL_SEEDS_DEFAULT)}
    train, val = [], []
    for folder in folders:
        rec_path = Path(folder) / "record.json"
        if not rec_path.is_file():
            continue
        try:
            record = json.loads(rec_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not _keep_episode(record, Path(folder), cfg):
            continue
        seed = int(record.get("seed") if record.get("seed") is not None else Path(folder).name)
        (val if seed in val_seeds else train).append(Path(folder))
    return train, val


def _label_rows(folders: list[Path], radius: float) -> list[dict]:
    rows = []
    for folder in folders:
        record, steps = _read_record_steps(folder)
        for step in steps:
            labels = labels_from_step(step, radius=radius)
            branch = step.get("expert_branch")
            branch_s = "" if branch is None else str(branch)
            if labels is None:
                rows.append({
                    "has_c": False,
                    "used_wide": False,
                    "mode": None,
                    "turn": None,
                    "forward": None,
                    "stop": None,
                    "clipped": False,
                    "branch": branch_s,
                })
                continue
            rows.append({
                "has_c": True,
                "used_wide": bool(labels.get("used_wide")),
                "mode": int(labels["mode"]),
                "turn": int(labels["turn_index"]),
                "forward": int(labels["forward_index"]),
                "stop": int(labels["stop"]),
                "clipped": bool(labels.get("forward_clipped")),
                "branch": branch_s,
            })
    return rows


def _split_stats(rows: list[dict]) -> dict:
    usable = [r for r in rows if r["has_c"] and not r["used_wide"] and int(r["stop"] or 0) == 0]
    ignored = sum(1 for r in rows if not r["has_c"])
    wide = sum(1 for r in rows if r["has_c"] and r["used_wide"])
    mode_c = Counter(r["mode"] for r in usable)
    turn_c = Counter(r["turn"] for r in usable)
    fwd_c = Counter(r["forward"] for r in usable)
    branch_c = Counter(r["branch"] if r["branch"] else "(empty)" for r in rows)
    mapped = {}
    unmapped = {}
    for raw, cnt in sorted(branch_c.items()):
        src = None if raw == "(empty)" else raw
        idx, wide_b = mode_from_branch(src, raw == "wide_turn")
        mapped[raw] = {"count": cnt, "mode": MODE_NAMES[idx], "mode_index": idx, "used_wide": wide_b}
        if src is not None and not branch_is_mapped(src):
            unmapped[raw] = cnt
    stop_pos = sum(int(r["stop"] or 0) for r in rows if r["has_c"])
    stop_n = sum(1 for r in rows if r["has_c"])
    return {
        "n": len(rows),
        "ignored_no_c": ignored,
        "used_wide": wide,
        "mode": {MODE_NAMES[i]: int(mode_c.get(i, 0)) for i in range(len(MODE_NAMES))},
        "turn": {str(i): int(turn_c.get(i, 0)) for i in range(N_TURN)},
        "forward": {str(FORWARD_BINS[i]): int(fwd_c.get(i, 0)) for i in range(len(FORWARD_BINS))},
        "forward_clipped": sum(1 for r in usable if r["clipped"]),
        "stop_pos": stop_pos,
        "stop_neg": stop_n - stop_pos,
        "expert_branch": mapped,
        "unmapped_branch": unmapped,
        "usable": usable,
    }


def _print_json(title: str, obj) -> None:
    print(title, flush=True)
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False), flush=True)


@_alarm(60)
def run_c1(cfg: dict) -> str:
    radius = float((cfg.get("judge") or {}).get("radius") or 3.0)
    train_ep, val_ep = _episode_folders(cfg)
    print(f"C1 episodes train={len(train_ep)} val={len(val_ep)}", flush=True)
    train_rows = _label_rows(train_ep, radius)
    val_rows = _label_rows(val_ep, radius)
    train_s = _split_stats(train_rows)
    val_s = _split_stats(val_rows)
    usable = train_s.pop("usable")
    val_s.pop("usable", None)
    mode_w, mode_n = class_weights([r["mode"] for r in usable], len(MODE_NAMES))
    turn_w, turn_n = class_weights([r["turn"] for r in usable], N_TURN)
    fwd_w, fwd_n = class_weights([r["forward"] for r in usable], len(FORWARD_BINS))

    class Dummy:
        def __init__(self, stop, has_c):
            self.stop = stop
            self.has_c = has_c

    dummy = [Dummy(r["stop"], True) for r in train_rows if r["has_c"]]
    pos_w = stop_pos_weight(dummy) if dummy else None
    rare = []
    for name, n in train_s["mode"].items():
        if n < 30:
            rare.append(f"mode:{name}={n}")
    for name, n in train_s["turn"].items():
        if n < 30:
            rare.append(f"turn:{name}={n}")
    for name, n in train_s["forward"].items():
        if n < 30:
            rare.append(f"forward:{name}={n}")
    ok = train_s["stop_pos"] > 0 and train_s["forward"].get("0", 0) > 0
    status = "PASS" if ok else "FAIL"
    _print_json("C1 train", train_s)
    _print_json("C1 val", val_s)
    _print_json("C1 weights", {
        "mode": [{"class": MODE_NAMES[i], "n_c": int(mode_n[i]), "w": float(mode_w[i])} for i in range(len(MODE_NAMES))],
        "turn": [{"class": i, "n_c": int(turn_n[i]), "w": float(turn_w[i])} for i in range(N_TURN)],
        "forward": [{"class": FORWARD_BINS[i], "n_c": int(fwd_n[i]), "w": float(fwd_w[i])} for i in range(len(FORWARD_BINS))],
        "stop_pos_weight": pos_w,
        "rare_n_lt_30": rare,
        "status": status,
        "checks": {
            "stop_pos": train_s["stop_pos"],
            "forward_0": train_s["forward"].get("0", 0),
        },
    })
    print(f"C1 {status}", flush=True)
    return status


def _index_lookup(index: dict) -> dict[tuple[str, int], int]:
    table = {}
    for row in index.get("rows") or []:
        table[(str(row["episode"]), int(row["decision"]))] = int(row["row"])
    return table


def _collect_samples(folders, lookup, spec, cfg, limit: int, min_stop: int = 0) -> list[NavSample]:
    out: list[NavSample] = []
    n_stop = 0
    for folder in folders:
        rows = samples_from_episode(folder, lookup, spec, cfg, None)
        out.extend(rows)
        n_stop = sum(int(r.stop) for r in out if r.has_c)
        if len(out) >= limit and n_stop >= min_stop:
            break
    return out


@_alarm(60)
def run_c3(cfg: dict, feat: Path) -> str:
    import torch

    from lfnav.perception.convert_features import load_memmaps

    spec = spec_from_cfg(cfg)
    spec["dropout"] = 0.0
    tokens, _, index = load_memmaps(feat)
    spec["dim"] = int(index.get("dim") or spec["dim"])
    spec["grid"] = tuple(index.get("grid") or spec["grid"])
    lookup = _index_lookup(index)
    train_ep, _val_ep = _episode_folders(cfg)
    order = list(train_ep)
    random.Random(0).shuffle(order)
    rows = _collect_samples(order, lookup, spec, cfg, 512, min_stop=1)
    rows = rows[:512]
    if len(rows) < 8:
        print(f"C3 FAIL too few samples {len(rows)}", flush=True)
        return "FAIL"
    src = Counter(r.obs_source for r in rows)
    usable = [r for r in rows if r.has_c and (not r.used_wide) and int(r.stop) == 0]
    mode_w, mode_n = class_weights([r.mode for r in usable], len(MODE_NAMES))
    turn_w, turn_n = class_weights([r.turn for r in usable], N_TURN)
    fwd_w, fwd_n = class_weights([r.forward for r in usable], len(FORWARD_BINS))
    pos_w = stop_pos_weight(rows)
    _print_json("C3 label stats", {
        "n": len(rows),
        "usable_policy": len(usable),
        "obs_source": dict(src),
        "mode": [{"class": MODE_NAMES[i], "n_c": int(mode_n[i]), "w": float(mode_w[i])} for i in range(len(MODE_NAMES))],
        "turn": [{"class": int(TURN_BINS[i]), "n_c": int(turn_n[i]), "w": float(turn_w[i])} for i in range(N_TURN)],
        "forward": [{"class": int(FORWARD_BINS[i]), "n_c": int(fwd_n[i]), "w": float(fwd_w[i])} for i in range(len(FORWARD_BINS))],
        "stop_pos_weight": pos_w,
        "stop_pos": int(sum(int(r.stop) for r in rows if r.has_c)),
        "stop_neg": int(sum(1 for r in rows if r.has_c and int(r.stop) == 0)),
    })
    print(f"C3 n={len(rows)} obs_source={dict(src)}", flush=True)
    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = C2NavNet(spec).to(device)
    model.eval()
    bearings = patch_bearings(spec["grid"][0], spec["grid"][1], cfg)
    batch = _to_torch(collate(rows, tokens, False, random.Random(0)), device, bearings=bearings)
    k = len(spec["objects"])
    with torch.no_grad():
        out = heads_out(model, batch)
        _t, parts = nav_loss(
            out, batch, {"a": 1, "mode": 1, "wp": 1, "stop": 1},
            torch.ones(k, device=device),
            torch.ones(len(MODE_NAMES), device=device),
            torch.ones(N_TURN, device=device),
            torch.ones(len(FORWARD_BINS), device=device),
            torch.tensor(1.0, device=device),
            0.0,
        )
    mean4 = (parts["biome"] + parts["time"] + parts["weather"] + parts["brightness"]) / 4.0
    refs = {
        "mode": math.log(len(MODE_NAMES)),
        "turn": math.log(N_TURN),
        "forward": math.log(len(FORWARD_BINS)),
        "present": math.log(2.0),
    }
    glob_ok = abs(mean4 - parts["global"]) < 1e-6
    total_ok = parts["total"] < 20
    ce_ok = True
    for name in ("mode", "turn", "forward"):
        ref = refs[name]
        val = parts[name]
        if not (0.5 * ref <= val <= 2.0 * ref):
            ce_ok = False
    src_ok = list(src.keys()) == ["gt_noisy"]
    status = "PASS" if total_ok and glob_ok and ce_ok and src_ok else "FAIL"
    _print_json("C3 losses", {
        "present_bce": parts["present"],
        "log_dist": parts["log_dist"],
        "L_global": parts["global"],
        "biome": parts["biome"],
        "time": parts["time"],
        "weather": parts["weather"],
        "brightness": parts["brightness"],
        "mean4": mean4,
        "L_mode": parts["mode"],
        "turn_ce": parts["turn"],
        "forward_ce": parts["forward"],
        "L_stop": parts["stop"],
        "L": parts["total"],
        "ref": refs,
        "L_global_eq_mean4": glob_ok,
        "status": status,
    })
    print(f"C3 {status}", flush=True)
    return status


def _pick64(rows: list[NavSample]) -> list[NavSample]:
    pos = [r for r in rows if r.has_c and int(r.stop) == 1]
    neg = [r for r in rows if r.has_c and int(r.stop) == 0]
    if len(pos) >= 8:
        take_pos = pos[: max(8, min(len(pos), 16))]
        take_neg = neg[: 64 - len(take_pos)]
        out = (take_pos + take_neg)[:64]
        if len(out) < 64:
            rest = [r for r in rows if r not in out]
            out = (out + rest)[:64]
        return out
    return rows[:64]


@_alarm(180)
def run_c8(cfg: dict, feat: Path) -> str:
    import torch

    from lfnav.perception.convert_features import load_memmaps

    spec = spec_from_cfg(cfg)
    spec["dropout"] = 0.0
    tokens, _, index = load_memmaps(feat)
    spec["dim"] = int(index.get("dim") or spec["dim"])
    spec["grid"] = tuple(index.get("grid") or spec["grid"])
    lookup = _index_lookup(index)
    train_ep, _ = _episode_folders(cfg)
    pool = _collect_samples(train_ep, lookup, spec, cfg, 400, min_stop=8)
    rows = _pick64(pool)
    n_stop = sum(int(r.stop) for r in rows if r.has_c)
    print(f"C8 n={len(rows)} stop_pos={n_stop}", flush=True)
    device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = C2NavNet(spec).to(device)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    bearings = patch_bearings(spec["grid"][0], spec["grid"][1], cfg)
    batch = _to_torch(collate(rows, tokens, False, random.Random(0)), device, bearings=bearings)
    k = len(spec["objects"])
    wts = {"a": 1.0, "mode": 1.0, "wp": 1.0, "stop": 1.0}
    pos_w = torch.ones(k, device=device)
    mode_w = torch.ones(len(MODE_NAMES), device=device)
    turn_w = torch.ones(N_TURN, device=device)
    fwd_w = torch.ones(len(FORWARD_BINS), device=device)
    stop_pos = torch.tensor(1.0, device=device)
    model.train()
    loss0 = None
    last = None
    for step in range(201):
        opt.zero_grad()
        out = heads_out(model, batch)
        loss, _parts = nav_loss(out, batch, wts, pos_w, mode_w, turn_w, fwd_w, stop_pos, 0.0)
        if step == 0:
            loss0 = float(loss.item())
        if step % 20 == 0:
            with torch.no_grad():
                c_mask = (batch["has_c"] > 0.5) & (batch["used_wide"] < 0.5) & (batch["stop"] < 0.5)
                stop_mask = batch["has_c"] > 0.5

                def acc(pred, y):
                    if not bool(c_mask.any()):
                        return None
                    return float((pred[c_mask] == y[c_mask]).float().mean().item())

                pred_stop = (torch.sigmoid(out["stop"][stop_mask]) >= 0.5).float()
                gold = batch["stop"][stop_mask]
                tp = float(((pred_stop > 0.5) & (gold > 0.5)).sum().item())
                fp = float(((pred_stop > 0.5) & (gold < 0.5)).sum().item())
                fn = float(((pred_stop < 0.5) & (gold > 0.5)).sum().item())
                prec = tp / max(tp + fp, 1.0)
                rec = tp / max(tp + fn, 1.0)
                stop_f1 = 2.0 * prec * rec / max(prec + rec, 1e-9)
                rec = {
                    "step": step,
                    "loss": float(loss.item()),
                    "mode_acc": acc(out["mode"].argmax(-1), batch["mode"]),
                    "turn_acc": acc(out["turn"].argmax(-1), batch["turn"]),
                    "forward_acc": acc(out["forward"].argmax(-1), batch["forward"]),
                    "stop_f1": stop_f1,
                }
                last = rec
                print(f"C8 {json.dumps(rec)}", flush=True)
        if step == 200:
            break
        loss.backward()
        opt.step()
    reasons = []
    if last is None or loss0 is None:
        reasons.append("no steps")
    else:
        if last["loss"] >= 0.05 * loss0:
            reasons.append(f"loss {last['loss']} >= 5% of {loss0}")
        for head, bar in (("mode_acc", 0.95), ("turn_acc", 0.89), ("forward_acc", 0.95)):
            if last[head] is None or last[head] < bar:
                reasons.append(f"{head}={last[head]} < {bar}")
        if last.get("stop_f1") is None or last["stop_f1"] < 0.9:
            reasons.append(f"stop_f1={last.get('stop_f1')} < 0.9")
    status = "PASS" if not reasons else "FAIL"
    print(f"C8 {status} reasons={reasons}", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--only", default=None, choices=["C3", "C8"])
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    cfg = load_config(Path(args.config))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {}
    want = {args.only} if args.only else {"C1", "C3", "C8"}
    if "C1" in want:
        try:
            results["C1"] = run_c1(cfg)
        except ItemTimeout as exc:
            print(f"C1 FAIL timeout {exc}", flush=True)
            results["C1"] = "FAIL"
    feat = _resolve_features(args, cfg)
    if feat is None or not Path(feat).exists():
        print("C3 FAIL missing --features or LFNAV_FEATURES / SCR", flush=True)
        print("C8 FAIL missing features", flush=True)
        if "C3" in want:
            results["C3"] = "FAIL"
        if "C8" in want:
            results["C8"] = "FAIL"
    else:
        if "C3" in want:
            try:
                results["C3"] = run_c3(cfg, Path(feat))
            except ItemTimeout as exc:
                print(f"C3 FAIL timeout {exc}", flush=True)
                results["C3"] = "FAIL"
        if "C8" in want:
            try:
                results["C8"] = run_c8(cfg, Path(feat))
            except ItemTimeout as exc:
                print(f"C8 FAIL timeout {exc}", flush=True)
                results["C8"] = "FAIL"
    elapsed = time.time() - t0
    payload = {"elapsed_s": elapsed, "results": results}
    (out_dir / "selfcheck.json").write_text(json.dumps(payload, indent=2))
    print(f"done elapsed_s={elapsed:.1f} {results}", flush=True)


if __name__ == "__main__":
    main()
