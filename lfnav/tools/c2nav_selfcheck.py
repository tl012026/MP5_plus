"""Login-node C2Nav training self-check. No jobs, no MineDojo, no full train."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "4")

from lfnav.config import load_config
from lfnav.policy.c2nav import C2NavNet, patch_bearings, spec_from_cfg
from lfnav.policy.c2nav_labels import (
    FORWARD_BINS,
    MODE_NAMES,
    N_TURN,
    TURN_WIDTH,
    branch_is_mapped,
    clip_forward_blocks,
    compose_waypoint,
    forward_index,
    invert_waypoint,
    mode_from_branch,
    turn_center,
    turn_index,
)
from lfnav.policy.train_c2nav import (
    class_weights,
    collate,
    heads_out,
    load_split,
    nav_loss,
    offline_metrics,
    stop_pos_weight,
    _to_torch,
)


def _resolve_features(args, cfg) -> Path | None:
    if args.features:
        return Path(args.features)
    env = os.environ.get("LFNAV_FEATURES")
    if env:
        return Path(env)
    version = str((cfg.get("c2nav") or {}).get("features_version") or "")
    root = os.environ.get("LFNAV_FEATURE_ROOT") or os.environ.get("SCR")
    if root and version:
        return Path(root) / "lfnav" / "features" / version
    return None


def _item(name: str, status: str, detail) -> dict:
    return {"name": name, "status": status, "detail": detail}


def _fmt(item: dict) -> str:
    body = json.dumps(item["detail"], default=str, ensure_ascii=False)
    return f"## {item['name']}: {item['status']}\n\n```json\n{body}\n```\n"


def s1_labels(train, val, spec) -> dict:
    objects = list(spec["objects"])

    def stats(rows):
        usable = [r for r in rows if r.has_c and not r.used_wide]
        ignored = sum(1 for r in rows if not r.has_c)
        wide = sum(1 for r in rows if r.has_c and r.used_wide)
        mode_c = Counter(r.mode for r in usable)
        turn_c = Counter(r.turn for r in usable)
        fwd_c = Counter(r.forward for r in usable)
        branch_c = Counter(r.branch if r.branch else "(empty)" for r in rows)
        mapped = {}
        unmapped = {}
        for raw, cnt in sorted(branch_c.items()):
            src = None if raw == "(empty)" else raw
            idx, wide_b = mode_from_branch(src, raw == "wide_turn")
            mapped[raw] = {
                "count": cnt,
                "mode": MODE_NAMES[idx],
                "mode_index": idx,
                "used_wide": wide_b,
            }
            if src is not None and not branch_is_mapped(src):
                unmapped[raw] = cnt
        stop_pos = sum(int(r.stop) for r in rows if r.has_c)
        stop_neg = sum(1 for r in rows if r.has_c) - stop_pos
        present = {}
        for i, name in enumerate(objects):
            pos = 0
            tot = 0
            for r in rows:
                m = r.ignore < 0.5
                tot += int(m.sum())
                pos += int((r.present[i][m] > 0.5).sum())
            present[name] = None if tot == 0 else pos / tot
        return {
            "n": len(rows),
            "ignored_no_c": ignored,
            "used_wide": wide,
            "mode": {MODE_NAMES[i]: int(mode_c.get(i, 0)) for i in range(len(MODE_NAMES))},
            "turn": {str(i): int(turn_c.get(i, 0)) for i in range(N_TURN)},
            "forward": {str(FORWARD_BINS[i]): int(fwd_c.get(i, 0)) for i in range(len(FORWARD_BINS))},
            "forward_clipped": sum(1 for r in usable if r.forward_clipped),
            "stop_pos": stop_pos,
            "stop_neg": stop_neg,
            "stop_pos_frac": None if (stop_pos + stop_neg) == 0 else stop_pos / (stop_pos + stop_neg),
            "present_pos_frac": present,
            "expert_branch": mapped,
            "unmapped_branch": unmapped,
        }

    train_s = stats(train)
    val_s = stats(val)
    n_mode_train = sum(1 for v in train_s["mode"].values() if v > 0)
    reasons = []
    if train_s["stop_pos"] <= 0:
        reasons.append("train stop positives == 0")
    if train_s["forward"].get("0", 0) <= 0:
        reasons.append("train forward 0-block count == 0")
    if train_s["unmapped_branch"]:
        reasons.append(f"unmapped expert_branch {train_s['unmapped_branch']}")
    if n_mode_train < 3:
        reasons.append(f"train mode classes with samples={n_mode_train} < 3")
    status = "PASS" if not reasons else "FAIL"
    return _item("S1", status, {"train": train_s, "val": val_s, "reasons": reasons})


def s2_weights(train) -> dict:
    usable = [r for r in train if r.has_c and not r.used_wide]
    mode_w, mode_n = class_weights([r.mode for r in usable], len(MODE_NAMES))
    turn_w, turn_n = class_weights([r.turn for r in usable], N_TURN)
    fwd_w, fwd_n = class_weights([r.forward for r in usable], len(FORWARD_BINS))
    try:
        pos_w = stop_pos_weight(train)
        stop_err = None
    except Exception as exc:
        pos_w = None
        stop_err = str(exc)

    def pack(names, w, n):
        rows = []
        for i, name in enumerate(names):
            rows.append({"class": name, "n_c": int(n[i]), "weight": float(w[i])})
        valid = w[n > 0]
        mean = float(valid.mean()) if valid.size else None
        return {"rows": rows, "valid_mean": mean, "max": float(w.max()) if w.size else None}

    detail = {
        "mode": pack(list(MODE_NAMES), mode_w, mode_n),
        "turn": pack([str(i) for i in range(N_TURN)], turn_w, turn_n),
        "forward": pack([str(b) for b in FORWARD_BINS], fwd_w, fwd_n),
        "stop_pos_weight": pos_w,
        "stop_error": stop_err,
    }
    reasons = []
    for head in ("mode", "turn", "forward"):
        block = detail[head]
        if block["max"] is not None and block["max"] > 10.0 + 1e-6:
            reasons.append(f"{head} max weight {block['max']} > 10")
        for row in block["rows"]:
            if row["n_c"] == 0 and abs(row["weight"]) > 1e-8:
                reasons.append(f"{head} class {row['class']} n_c=0 weight={row['weight']}")
        if block["valid_mean"] is not None and abs(block["valid_mean"] - 1.0) > 1e-4:
            reasons.append(f"{head} valid mean {block['valid_mean']} != 1")
    if pos_w is not None and pos_w > 10.0 + 1e-6:
        reasons.append(f"stop pos_weight {pos_w} > 10")
    if stop_err:
        reasons.append(stop_err)
    return _item("S2", "PASS" if not reasons else "FAIL", {**detail, "reasons": reasons})


def s3_losses(model, rows, tokens, device, spec, cfg, bearings) -> dict:
    import torch

    rng = random.Random(0)
    n = min(512, len(rows))
    if n <= 0:
        return _item("S3", "FAIL", {"reasons": ["no train samples"]})
    idx = rng.sample(range(len(rows)), k=n)
    subset = [rows[i] for i in idx]
    batch = _to_torch(collate(subset, tokens, False, random.Random(0)), device, bearings=bearings)
    k = len(spec["objects"])
    pos_w = torch.ones(k, device=device)
    mode_w = torch.ones(len(MODE_NAMES), device=device)
    turn_w = torch.ones(N_TURN, device=device)
    fwd_w = torch.ones(len(FORWARD_BINS), device=device)
    usable = [r for r in rows if r.has_c and not r.used_wide]
    mw, mn = class_weights([r.mode for r in usable], len(MODE_NAMES))
    tw, tn = class_weights([r.turn for r in usable], N_TURN)
    fw, fn = class_weights([r.forward for r in usable], len(FORWARD_BINS))
    mode_w = torch.as_tensor(mw, device=device)
    turn_w = torch.as_tensor(tw, device=device)
    fwd_w = torch.as_tensor(fw, device=device)
    k_eff = int((mn > 0).sum())
    refs = {
        "present": math.log(2.0),
        "global": None,
        "mode": math.log(max(k_eff, 1)),
        "turn": math.log(max(int((tn > 0).sum()), 1)),
        "forward": math.log(max(int((fn > 0).sum()), 1)),
        "stop": math.log(2.0),
    }
    model.eval()
    with torch.no_grad():
        out = heads_out(model, batch)
        _total, parts = nav_loss(
            out, batch, {"a": 1, "mode": 1, "wp": 1, "stop": 1},
            pos_w, mode_w, turn_w, fwd_w, torch.tensor(1.0, device=device), 0.0,
        )
    reasons = []
    checked = {
        "present": (parts["present"], refs["present"]),
        "mode": (parts["mode"], refs["mode"]),
        "turn": (parts["turn"], refs["turn"]),
        "forward": (parts["forward"], refs["forward"]),
        "stop": (parts["stop"], math.log(2.0)),
    }
    for name, (val, ref) in checked.items():
        if not math.isfinite(val):
            reasons.append(f"{name} not finite: {val}")
            continue
        if not (0.5 * ref <= val <= 2.0 * ref):
            reasons.append(f"{name}={val:.4f} not in [{0.5 * ref:.4f}, {2.0 * ref:.4f}] (ref {ref:.4f})")
    if not math.isfinite(parts["log_dist"]):
        reasons.append(f"log_dist not finite: {parts['log_dist']}")
    if parts["total"] >= 20:
        reasons.append(f"total L={parts['total']} >= 20")
    return _item("S3", "PASS" if not reasons else "FAIL", {
        "n": n,
        "parts": parts,
        "ref": {k: v for k, v in refs.items()},
        "reasons": reasons,
    })


def s4_global(model, rows, tokens, device, spec, bearings) -> dict:
    import torch

    if not rows:
        return _item("S4", "FAIL", {"reasons": ["no samples"]})
    subset = rows[: min(32, len(rows))]
    batch = _to_torch(collate(subset, tokens, False, random.Random(0)), device, bearings=bearings)
    k = len(spec["objects"])
    model.eval()
    with torch.no_grad():
        out = heads_out(model, batch)
        _t, parts = nav_loss(
            out, batch, {"a": 1, "mode": 1, "wp": 1, "stop": 1},
            torch.ones(k, device=device),
            torch.ones(len(MODE_NAMES), device=device),
            torch.ones(N_TURN, device=device),
            torch.ones(len(FORWARD_BINS), device=device),
            torch.tensor(1.0, device=device), 0.0,
        )
    mean4 = (parts["biome"] + parts["time"] + parts["weather"] + parts["brightness"]) / 4.0
    err = abs(mean4 - parts["global"])
    status = "PASS" if err < 1e-6 else "FAIL"
    return _item("S4", status, {
        "biome": parts["biome"],
        "time": parts["time"],
        "weather": parts["weather"],
        "brightness": parts["brightness"],
        "L_global": parts["global"],
        "mean4": mean4,
        "abs_err": err,
    })


def s5_obs_source(train, tokens) -> dict:
    if len(train) < 3:
        return _item("S5", "FAIL", {"reasons": ["need >=3 train samples"]})
    rng = random.Random(1)
    counts = Counter()
    example = None
    bs = min(16, len(train))
    for start in (0, bs, min(2 * bs, max(len(train) - bs, 0))):
        chunk = train[start : start + bs]
        if not chunk:
            continue
        batch = collate(chunk, tokens, False, rng)
        for src in batch["obs_source"]:
            counts[str(src)] += 1
        if example is None:
            row = chunk[0]
            noisy = [float(x) for x in row.geom]
            clean = [float(x) for x in (row.geom if row.geom_clean is None else row.geom_clean)]
            example = {
                "obs_source": row.obs_source,
                "visible_clean": clean[0],
                "visible_noisy": noisy[0],
                "sin_clean": clean[1],
                "sin_noisy": noisy[1],
                "cos_clean": clean[2],
                "cos_noisy": noisy[2],
                "log_dist_clean": clean[3],
                "log_dist_noisy": noisy[3],
            }
    bad = {k: v for k, v in counts.items() if k != "gt_noisy"}
    status = "PASS" if counts and not bad else "FAIL"
    return _item("S5", status, {"obs_source_counts": dict(counts), "example": example, "non_gt_noisy": bad})


def s6_roundtrip() -> dict:
    cases = []
    ok = True
    for blocks in (-1, 0, 1, 4, 6):
        idx = forward_index(blocks)
        actions = invert_waypoint(0, idx)
        again = compose_waypoint(actions)
        want = clip_forward_blocks(blocks)[0]
        hit = again["forward"] == want and again["forward_index"] == forward_index(want)
        cases.append({"kind": "forward", "blocks": blocks, "index": idx, "decoded": again["forward"], "ok": hit})
        ok = ok and hit
    for yaw in (0.0, 15.0, -15.0, 45.0, -180.0, 180.0):
        idx = turn_index(yaw)
        actions = invert_waypoint(idx, 1)
        again = compose_waypoint(actions)
        hit = again["turn_index"] == idx
        cases.append({"kind": "turn", "yaw": yaw, "index": idx, "decoded": again["turn_index"], "ok": hit})
        ok = ok and hit
    for i in range(N_TURN):
        center = turn_center(i)
        lo = center - TURN_WIDTH / 2.0 + 1e-3
        hi = center + TURN_WIDTH / 2.0 - 1e-3
        for yaw in (center, lo, hi):
            hit = turn_index(yaw) == i
            cases.append({"kind": "turn_boundary", "i": i, "yaw": yaw, "ok": hit})
            ok = ok and hit
        actions = invert_waypoint(i, 1)
        again = compose_waypoint(actions)
        hit = again["turn_index"] == i
        cases.append({"kind": "turn_invert", "i": i, "ok": hit})
        ok = ok and hit
    return _item("S6", "PASS" if ok else "FAIL", {"cases": cases, "n": len(cases)})


def s7_metrics(model, val, tokens, device, spec, cfg, bearings) -> dict:
    metrics = offline_metrics(model, val, tokens, device, spec, cfg, bearings, warn_empty_stop=True)
    reasons = []

    def value_of(key):
        raw = metrics.get(key)
        if isinstance(raw, dict) and "value" in raw:
            return raw.get("value"), raw.get("n")
        return raw, metrics.get("n_" + key, metrics.get("n"))

    for key in ("mode_acc", "turn_acc", "forward_acc", "stop_f1", "stop_auc", "present_auc", "log_dist_mae"):
        raw = metrics.get(key)
        if not (isinstance(raw, dict) and "n" in raw):
            reasons.append(f"{key} missing n")
    by = metrics.get("mode_acc_by_name") or {}
    for name, raw in by.items():
        n = raw.get("n") if isinstance(raw, dict) else None
        val_m = raw.get("value") if isinstance(raw, dict) else raw
        if n == 0 and val_m is not None:
            reasons.append(f"mode {name} n=0 but value={val_m}")
    mode_v, mode_n = value_of("mode_acc")
    if mode_v is None:
        reasons.append("mode_acc is null")
    elif not (0.05 <= float(mode_v) <= 0.6):
        reasons.append(f"mode_acc={mode_v} not in [0.05, 0.6]")
    if mode_v == 0.0:
        reasons.append("mode_acc is exactly 0.0")
    stop_f1, n_pos = value_of("stop_f1")
    if n_pos == 0 and stop_f1 is not None:
        reasons.append("stop_f1 should be null when n_pos=0")
    return _item("S7", "PASS" if not reasons else "FAIL", {"metrics": metrics, "reasons": reasons})


def _pick_overfit_rows(train, n: int = 64, min_stop: int = 8):
    usable = [r for r in train if r.has_c]
    if len(usable) < n:
        usable = list(train)
    window = usable[:n]
    if sum(int(r.stop) for r in window) > 0:
        return window
    pos = [r for r in usable if r.stop]
    neg = [r for r in usable if not r.stop]
    if len(pos) >= min_stop:
        take_pos = pos[:max(min_stop, min(len(pos), n // 4))]
        take_neg = neg[: n - len(take_pos)]
        return (take_pos + take_neg)[:n]
    return window


def s8_overfit(cfg, train, tokens, spec, device, bearings) -> dict:
    import torch

    spec = dict(spec)
    spec["dropout"] = 0.0
    rows = _pick_overfit_rows(train, 64, 8)
    n_stop = sum(int(r.stop) for r in rows if r.has_c)
    torch.manual_seed(0)
    model = C2NavNet(spec).to(device)
    for mod in model.modules():
        if hasattr(mod, "dropout"):
            try:
                mod.dropout = 0.0
            except Exception:
                pass
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    k = len(spec["objects"])
    wts = {"a": 1.0, "mode": 1.0, "wp": 1.0, "stop": 1.0}
    pos_w = torch.ones(k, device=device)
    mode_w = torch.ones(len(MODE_NAMES), device=device)
    turn_w = torch.ones(N_TURN, device=device)
    fwd_w = torch.ones(len(FORWARD_BINS), device=device)
    stop_pos = torch.tensor(1.0, device=device)
    batch = _to_torch(collate(rows, tokens, False, random.Random(0)), device, bearings=bearings)
    model.train()
    history = []
    loss0 = None
    last = None
    for step in range(201):
        opt.zero_grad()
        out = heads_out(model, batch)
        loss, parts = nav_loss(out, batch, wts, pos_w, mode_w, turn_w, fwd_w, stop_pos, 0.0)
        if step == 0:
            loss0 = float(loss.item())
        if step % 20 == 0:
            with torch.no_grad():
                pred_mode = out["mode"].argmax(-1)
                pred_turn = out["turn"].argmax(-1)
                pred_fwd = out["forward"].argmax(-1)
                c_mask = (batch["has_c"] > 0.5) & (batch["used_wide"] < 0.5)
                def acc(pred, y):
                    if not bool(c_mask.any()):
                        return None
                    return float((pred[c_mask] == y[c_mask]).float().mean().item())
                stop_t = batch["stop"]
                stop_p = (torch.sigmoid(out["stop"]) >= 0.5).float()
                sm = batch["has_c"] > 0.5
                f1 = None
                if bool(sm.any()) and float(stop_t[sm].sum()) > 0:
                    y = stop_t[sm]
                    p = stop_p[sm]
                    tp = float(((p == 1) & (y == 1)).sum().item())
                    fp = float(((p == 1) & (y == 0)).sum().item())
                    fn = float(((p == 0) & (y == 1)).sum().item())
                    prec = tp / max(tp + fp, 1.0)
                    rec = tp / max(tp + fn, 1.0)
                    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                rec = {
                    "step": step,
                    "loss": float(loss.item()),
                    "mode_acc": acc(pred_mode, batch["mode"]),
                    "turn_acc": acc(pred_turn, batch["turn"]),
                    "forward_acc": acc(pred_fwd, batch["forward"]),
                    "stop_f1": f1,
                }
                history.append(rec)
                print(f"S8 step={step} {json.dumps(rec)}", flush=True)
        if step == 200:
            last = history[-1]
            break
        loss.backward()
        opt.step()
    reasons = []
    failed_heads = []
    if last is None or loss0 is None:
        reasons.append("did not run")
    else:
        if last["loss"] >= 0.05 * loss0:
            reasons.append(f"loss {last['loss']} >= 5% of {loss0}")
        for head in ("mode_acc", "turn_acc", "forward_acc"):
            if last[head] is None or last[head] < 0.95:
                failed_heads.append(head)
                reasons.append(f"{head}={last[head]} < 0.95")
        if n_stop > 0 and (last["stop_f1"] is None or last["stop_f1"] < 0.9):
            failed_heads.append("stop")
            reasons.append(f"stop_f1={last['stop_f1']} < 0.9 (n_stop_pos={n_stop})")
    return _item("S8", "PASS" if not reasons else "FAIL", {
        "n": len(rows),
        "n_stop_pos": n_stop,
        "loss0": loss0,
        "history": history,
        "failed_heads": failed_heads,
        "reasons": reasons,
    })


def s9_no_submit() -> dict:
    files = [_ROOT / "lfnav" / "policy" / "train_c2nav.py"]
    imported = [
        _ROOT / "lfnav" / "policy" / "c2nav.py",
        _ROOT / "lfnav" / "policy" / "c2nav_labels.py",
        _ROOT / "lfnav" / "policy" / "state_tracker.py",
        _ROOT / "lfnav" / "policy" / "c2_data.py",
        _ROOT / "lfnav" / "perception" / "convert_features.py",
        _ROOT / "lfnav" / "perception" / "extract_features.py",
        _ROOT / "lfnav" / "data" / "demo_format.py",
        _ROOT / "lfnav" / "io.py",
        _ROOT / "lfnav" / "config.py",
    ]
    hits = []
    needles = ("sbatch", "subprocess", "submitted val")
    for path in files + imported:
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for needle in needles:
                if needle in line or needle in low:
                    hits.append({"file": str(path.relative_to(_ROOT)), "line": i, "needle": needle, "text": line.strip()})
    return _item("S9", "PASS" if not hits else "FAIL", {"hits": hits})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--features", default=None)
    args = parser.parse_args()
    t0 = time.time()
    import torch

    torch.set_num_threads(4)
    device = torch.device("cpu")
    cfg = load_config(Path(args.config))
    spec = spec_from_cfg(cfg)
    spec["dropout"] = 0.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    items.append(s6_roundtrip())
    items.append(s9_no_submit())
    feat = _resolve_features(args, cfg)
    train = val = []
    tokens = None
    bearings = None
    if feat is None or not Path(feat).exists():
        miss = _item("data", "FAIL", {"reason": "missing --features or LFNAV_FEATURES / SCR feature dir"})
        items.append(miss)
        for name in ("S1", "S2", "S3", "S4", "S5", "S7", "S8"):
            items.append(_item(name, "FAIL", {"reasons": ["features unavailable"]}))
    else:
        from lfnav.perception.convert_features import load_memmaps

        train, val, stats = load_split(cfg, Path(feat), apply_noise=True)
        tokens, _, index = load_memmaps(Path(feat))
        spec["dim"] = int(index.get("dim") or spec["dim"])
        spec["grid"] = tuple(index.get("grid") or spec["grid"])
        bearings = patch_bearings(spec["grid"][0], spec["grid"][1], cfg)
        print(f"selfcheck loaded train={len(train)} val={len(val)} grid={spec['grid']}", flush=True)
        items.append(s1_labels(train, val, spec))
        items.append(s2_weights(train))
        torch.manual_seed(0)
        model = C2NavNet(spec).to(device)
        items.append(s3_losses(model, train, tokens, device, spec, cfg, bearings))
        items.append(s4_global(model, train, tokens, device, spec, bearings))
        items.append(s5_obs_source(train, tokens))
        items.append(s7_metrics(model, val, tokens, device, spec, cfg, bearings))
        from dataclasses import replace

        train_clean = [
            replace(r, geom=r.geom if r.geom_clean is None else r.geom_clean)
            for r in train
        ]
        items.append(s8_overfit(cfg, train_clean, tokens, spec, device, bearings))
    order = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
    by = {it["name"]: it for it in items}
    ordered = [by[name] for name in order if name in by]
    payload = {
        "elapsed_s": time.time() - t0,
        "items": ordered,
        "pass": all(it["status"] == "PASS" for it in ordered),
    }
    (out_dir / "selfcheck.json").write_text(json.dumps(payload, indent=2, default=str))
    md = ["# C2Nav self-check", "", f"elapsed_s: {payload['elapsed_s']:.1f}", ""]
    for it in ordered:
        md.append(_fmt(it))
        print(f"{it['name']} {it['status']}", flush=True)
    (out_dir / "selfcheck.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out_dir / 'selfcheck.md'}", flush=True)


if __name__ == "__main__":
    main()
