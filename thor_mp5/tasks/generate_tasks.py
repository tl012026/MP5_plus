"""Generate conditional object-search tasks for AI2-THOR.

Run on a GPU node, from any directory:

    python generate_tasks.py --per-level 50 --seed 0 --out tasks_v1.jsonl
    python generate_tasks.py --dry-run 5 --seed 0
    python generate_tasks.py --verify tasks_v1.jsonl

The first and third commands start AI2-THOR with CloudRendering.
--dry-run only prints sampled tasks. It does not start a simulator and does not write files.
A full run writes the jsonl and a sibling summary, tasks_v1_summary.md next to --out.

InitialRandomSpawn's count is the number of extra copies, on top of the objects already in the scene.
A scene that already has more copies than the task needs is abandoned; a new seed cannot remove them.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_task import (  # noqa: E402
    RoleError,
    check_roles,
    find_by_name,
    object_poses,
    on_open_surface,
    receptacle_ids,
    same_type,
)

LEVELS = ("L1", "L2", "L3", "L4")
ROOM_ORDER = ("kitchen", "bedroom", "living", "bathroom")
YAWS = (0, 90, 180, 270)
MAX_AGENT_POSES = 40


def load_vocab(path):
    with open(path) as handle:
        vocab = yaml.safe_load(handle)
    missing = [name for name, room in vocab["rooms"].items() if "scene_ids" not in room]
    if missing:
        raise SystemExit(f"vocab rooms missing scene_ids: {missing}")
    return vocab


def room_quotas(per_level, vocab):
    base = vocab["level_room_counts"]
    total = sum(int(base[room]) for room in ROOM_ORDER)
    if per_level == total:
        return {room: int(base[room]) for room in ROOM_ORDER}
    raw = {room: per_level * int(base[room]) / total for room in ROOM_ORDER}
    floors = {room: int(raw[room]) for room in ROOM_ORDER}
    left = per_level - sum(floors.values())
    ranked = sorted(ROOM_ORDER, key=lambda room: (raw[room] - floors[room], int(base[room])), reverse=True)
    for room in ranked[:left]:
        floors[room] += 1
    return floors


def slots_for_level(level, quotas):
    slots = []
    for room in ROOM_ORDER:
        slots.extend((level, room) for _ in range(quotas[room]))
    return slots


def phrase(name, vocab):
    del vocab
    special = {
        "CounterTop": "countertop",
        "Bread": "loaf of bread",
        "ToiletPaper": "roll of toilet paper",
    }
    if name in special:
        return special[name]
    parts = []
    chunk = ""
    for char in name:
        if char.isupper() and chunk:
            parts.append(chunk)
            chunk = char
        else:
            chunk += char
    if chunk:
        parts.append(chunk)
    return " ".join(part.lower() for part in parts)


def article(text):
    return "an" if text[:1].lower() in "aeiou" else "a"


def task_text(room, target_type, conditions, vocab):
    head = phrase(target_type, vocab)
    state = conditions.get("state")
    if state:
        head = f"{state['text']} {head}"
    sentence = f"Find {article(head)} {head}"
    if conditions.get("placement"):
        sentence += f" on the {phrase(conditions['placement'], vocab)}"
    if conditions.get("next_to"):
        sentence += f" next to the {phrase(conditions['next_to'], vocab)}"
    sentence += f" in the {vocab['rooms'][room]['name']}"
    if conditions.get("lights_off"):
        sentence += " with the lights off"
    return sentence + "."


def _anchors_for(room_info, target_type, placement):
    skip = {target_type}
    if placement:
        skip.add(placement)
    return [name for name in room_info["anchors"] if name not in skip]


def sample_conditions(rng, level, room, target_type, vocab):
    room_info = vocab["rooms"][room]
    has_state = target_type in (vocab.get("states") or {})
    placement = None
    nxt = None
    state = None
    lights_off = False
    if level == "L1":
        chosen = []
    elif level == "L2":
        options = ["placement", "next_to"] + (["state"] if has_state else [])
        chosen = [rng.choice(options)]
    elif level == "L3":
        options = ["placement", "next_to"] + (["state"] if has_state else [])
        if len(options) < 2:
            raise RuntimeError(f"{room} {target_type} cannot fill two object conditions")
        chosen = rng.sample(options, 2) if len(options) > 2 else list(options)
    else:
        chosen = ["state", "placement", "next_to"] if has_state else ["placement", "next_to"]
        lights_off = rng.random() < float(vocab["lights_off_probability"])
    if "placement" in chosen:
        placement = rng.choice(list(room_info["open_surfaces"]))
    if "next_to" in chosen:
        anchors = _anchors_for(room_info, target_type, placement)
        if not anchors:
            raise RuntimeError(f"no next-to anchor for {target_type} in {room}")
        nxt = rng.choice(anchors)
    if "state" in chosen:
        spec = vocab["states"][target_type]
        state = {
            "property": spec["property"],
            "value": True,
            "text": spec["text"],
            "true_action": spec["true_action"],
            "false_action": spec["false_action"],
        }
    return {
        "state": state,
        "placement": placement,
        "next_to": nxt,
        "lights_off": lights_off,
    }


def sample_spec(rng, level, room, vocab):
    room_info = vocab["rooms"][room]
    target_type = rng.choice(list(room_info["targets"]))
    conditions = sample_conditions(rng, level, room, target_type, vocab)
    scene = rng.choice(list(room_info["scene_ids"]))
    return {
        "level": level,
        "room": room,
        "scene": scene,
        "target_type": target_type,
        "conditions": conditions,
        "text": task_text(room, target_type, conditions, vocab),
        "open_surfaces": list(room_info["open_surfaces"]),
        "near_threshold_m": float(vocab["near_threshold_m"]),
        "closed_containers": list(vocab["closed_containers"]),
    }


def needed_copies(conditions):
    """1 target, plus one distractor for each object condition. Lights are not an object condition."""
    n = 1
    if conditions.get("state"):
        n += 1
    if conditions.get("placement"):
        n += 1
    if conditions.get("next_to"):
        n += 1
    return n


def instances_of(objects, target_type):
    return [obj for obj in objects if same_type(obj, target_type)]


def plan_state_actions(objects, task):
    """Pick which instance plays which role, then the actions that make the state column true or false.

    Placement and next-to are left as the spawn put them. Returns (actions, reason).
    actions is None when this layout cannot host the roles.
    """
    conditions = task["conditions"]
    state = conditions.get("state")
    group = instances_of(objects, task["target_type"])
    closed = task["closed_containers"]
    surfaces = task["open_surfaces"]
    if any(not on_open_surface(obj, objects, surfaces, closed) for obj in group):
        return None, "an instance is inside a closed container or off an open surface"
    if len(group) != needed_copies(conditions):
        return None, f"{len(group)} instances, want {needed_copies(conditions)}"
    # Match on placement and next-to only. State is applied afterwards.
    probe = dict(conditions)
    probe["state"] = None
    from load_task import condition_flags, expected_roles, _role_flags

    unused = sorted(group, key=lambda obj: obj.get("objectId") or "")
    chosen = {}
    for role in expected_roles(conditions):
        need = _role_flags(role, probe)
        match = None
        for obj in unused:
            flags = condition_flags(obj, objects, probe, task["near_threshold_m"])
            if flags == need:
                match = obj
                break
        if match is None:
            return None, f"no layout for {role}"
        unused.remove(match)
        chosen[role] = match
    actions = []
    if state:
        for role, obj in chosen.items():
            want = role != "distractor_for_state"
            have = bool(obj.get(state["property"]))
            if have == want:
                continue
            action = state["true_action"] if want else state["false_action"]
            if not action:
                return None, f"cannot set {state['property']}={want} on {obj.get('objectId')}"
            actions.append({"objectName": obj.get("name"), "action": action, "role": role})
    return actions, ""


def apply_actions(controller, actions, task_id):
    for act in actions:
        obj = find_by_name(controller.last_event.metadata["objects"], act["objectName"])
        if obj is None:
            raise RoleError(f"{task_id}: lost {act['objectName']} before {act['action']}")
        event = controller.step(action=act["action"], objectId=obj["objectId"], forceAction=True)
        if not event.metadata.get("lastActionSuccess", False):
            message = event.metadata.get("errorMessage") or "action failed"
            raise RoleError(f"{task_id}: {act['action']} failed: {message}")


def set_lights(controller, lights_off):
    switches = [o for o in controller.last_event.metadata["objects"] if o.get("objectType") == "LightSwitch"]
    if lights_off:
        if not switches:
            raise RoleError("scene has no LightSwitch")
        for obj in switches:
            if obj.get("isToggled"):
                event = controller.step(action="ToggleObjectOff", objectId=obj["objectId"], forceAction=True)
                if not event.metadata.get("lastActionSuccess", False):
                    raise RoleError(event.metadata.get("errorMessage") or "ToggleObjectOff failed")
    lights = []
    for obj in controller.last_event.metadata["objects"]:
        if obj.get("objectType") == "LightSwitch":
            lights.append({"objectName": obj.get("name"), "isToggled": bool(obj.get("isToggled"))})
    return lights


def target_visible(objects, target_type):
    return any(same_type(obj, target_type) and obj.get("visible") for obj in objects)


def choose_agent_pose(controller, rng, target_type):
    event = controller.step(action="GetReachablePositions")
    positions = list(event.metadata.get("actionReturn") or [])
    if not positions:
        raise RoleError("no reachable positions")
    rng.shuffle(positions)
    yaws = list(YAWS)
    tried = 0
    for pos in positions:
        rng.shuffle(yaws)
        for yaw in yaws:
            if tried >= MAX_AGENT_POSES:
                raise RoleError("no agent pose hides every target instance")
            tried += 1
            rotation = {"x": 0, "y": int(yaw), "z": 0}
            event = controller.step(
                action="Teleport",
                position=pos,
                rotation=rotation,
                horizon=30,
                standing=True,
            )
            if not event.metadata.get("lastActionSuccess", False):
                continue
            if not target_visible(event.metadata["objects"], target_type):
                return {
                    "position": pos,
                    "rotation": rotation,
                    "horizon": 30,
                    "standing": True,
                }
    raise RoleError("no agent pose hides every target instance")


def spawn_kwargs(spec, objects, seed):
    need = needed_copies(spec["conditions"])
    have = len(instances_of(objects, spec["target_type"]))
    if have > need:
        raise RoleError(f"scene already has {have} {spec['target_type']}, task needs {need}")
    extra = need - have
    duplicates = [{"objectType": spec["target_type"], "count": extra}] if extra else []
    exclude = receptacle_ids(objects, spec["closed_containers"])
    return {
        "randomSeed": int(seed),
        "numDuplicatesOfType": duplicates,
        "excludedReceptacleTypes": list(spec["closed_containers"]),
        "excludedReceptacles": exclude,
        "numPlacementAttempts": 5,
    }


def try_once(controller, spec, seed, rng):
    """Return a task dict, or raise RoleError."""
    controller.reset(scene=spec["scene"])
    before = controller.last_event.metadata["objects"]
    spawn = spawn_kwargs(spec, before, seed)
    event = controller.step(
        action="InitialRandomSpawn",
        randomSeed=spawn["randomSeed"],
        forceVisible=False,
        numPlacementAttempts=spawn["numPlacementAttempts"],
        placeStationary=True,
        numDuplicatesOfType=spawn["numDuplicatesOfType"],
        excludedReceptacles=spawn["excludedReceptacles"],
    )
    if not event.metadata.get("lastActionSuccess", False):
        raise RoleError(event.metadata.get("errorMessage") or "InitialRandomSpawn failed")
    actions, reason = plan_state_actions(controller.last_event.metadata["objects"], spec)
    if actions is None:
        raise RoleError(reason)
    apply_actions(controller, actions, spec.get("id", spec["target_type"]))
    lights = set_lights(controller, spec["conditions"]["lights_off"])
    labeled = check_roles(controller.last_event.metadata["objects"], spec)
    pose_event = controller.step(action="GetObjectPoses")
    poses = object_poses(pose_event if pose_event.metadata.get("lastActionSuccess") else controller.last_event)
    agent = choose_agent_pose(controller, rng, spec["target_type"])
    task = {
        "id": spec["id"],
        "level": spec["level"],
        "room": spec["room"],
        "scene": spec["scene"],
        "text": spec["text"],
        "target_type": spec["target_type"],
        "conditions": spec["conditions"],
        "open_surfaces": spec["open_surfaces"],
        "closed_containers": spec["closed_containers"],
        "near_threshold_m": spec["near_threshold_m"],
        "spawn": {
            "randomSeed": spawn["randomSeed"],
            "numDuplicatesOfType": spawn["numDuplicatesOfType"],
            "excludedReceptacleTypes": spawn["excludedReceptacleTypes"],
            "numPlacementAttempts": spawn["numPlacementAttempts"],
        },
        "state_actions": [{"objectName": act["objectName"], "action": act["action"]} for act in actions],
        "object_poses": poses,
        "instances": labeled,
        "lights": lights,
        "agent": agent,
    }
    check_roles(controller.last_event.metadata["objects"], task)
    return task


def realize(controller, spec, rng, max_retries):
    """Try up to max_retries seeds. Returns (task or None, retries, last_error)."""
    retries = 0
    last = ""
    for _ in range(max_retries):
        seed = rng.randrange(1, 2 ** 31 - 1)
        try:
            return try_once(controller, spec, seed, rng), retries, ""
        except RoleError as exc:
            retries += 1
            last = str(exc)
            # A new seed only reshuffles. It cannot delete objects already in the scene.
            if "already has" in last:
                break
    return None, retries, last


def write_summary(path, tasks, per_level, seed, retries, dropped, drop_notes):
    by_level_room = Counter((task["level"], task["room"]) for task in tasks)
    by_target = Counter(task["target_type"] for task in tasks)
    by_condition = Counter()
    for task in tasks:
        cond = task["conditions"]
        if cond.get("state"):
            by_condition["state"] += 1
        if cond.get("placement"):
            by_condition["placement"] += 1
        if cond.get("next_to"):
            by_condition["next_to"] += 1
        if cond.get("lights_off"):
            by_condition["lights_off"] += 1
    lines = [
        "# Task generation summary",
        "",
        f"Seed {seed}. Requested {per_level} tasks per level. Wrote {len(tasks)}.",
        f"Retries (failed scene attempts): {retries}.",
        f"Dropped task specs (abandoned after the retry cap, then replaced or left empty): {dropped}.",
        "",
        "## Level by room",
        "",
        "| Level | kitchen | bedroom | living | bathroom | total |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for level in LEVELS:
        cells = [by_level_room[(level, room)] for room in ROOM_ORDER]
        lines.append(f"| {level} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {sum(cells)} |")
    lines += ["", "## Condition type", "", "| Condition | Tasks |", "| --- | ---: |"]
    for name in ("state", "placement", "next_to", "lights_off"):
        lines.append(f"| {name} | {by_condition[name]} |")
    lines += ["", "## Target type", "", "| Target | Tasks |", "| --- | ---: |"]
    for name, count in sorted(by_target.items()):
        lines.append(f"| {name} | {count} |")
    if drop_notes:
        lines += ["", "## Why specs were dropped", ""]
        for note, count in sorted(drop_notes.items(), key=lambda item: -item[1]):
            lines.append(f"- {count}: {note}")
    lines.append("")
    path.write_text("\n".join(lines))


def generate(args, vocab):
    from load_task import make_controller

    quotas = room_quotas(args.per_level, vocab)
    rng = random.Random(args.seed)
    controller = make_controller(platform=args.platform, width=args.width, height=args.height)
    tasks = []
    retries = 0
    dropped = 0
    drop_notes = Counter()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for level in LEVELS:
            counts = {room: 0 for room in ROOM_ORDER}
            for room in ROOM_ORDER:
                while counts[room] < quotas[room]:
                    spec = sample_spec(rng, level, room, vocab)
                    spec["id"] = f"{level}-{room}-{counts[room]:02d}"
                    task, used, last = realize(controller, spec, rng, args.max_retries)
                    retries += used
                    if task is None:
                        dropped += 1
                        drop_notes[last or "unknown"] += 1
                        print(f"drop {spec['id']} after {used} tries: {last}", flush=True)
                        if dropped > args.max_drops:
                            raise SystemExit(f"stopped after {dropped} dropped specs")
                        continue
                    tasks.append(task)
                    counts[room] += 1
                    print(f"ok {task['id']} {task['scene']} retries_this_spec={used} {task['text']}", flush=True)
    finally:
        controller.stop()
    with out.open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(task) + "\n")
    summary = out.with_name(out.stem + "_summary.md")
    write_summary(summary, tasks, args.per_level, args.seed, retries, dropped, drop_notes)
    print(f"wrote {out} ({len(tasks)} tasks) and {summary}", flush=True)
    if len(tasks) != args.per_level * len(LEVELS):
        raise SystemExit(f"incomplete: {len(tasks)} tasks, wanted {args.per_level * len(LEVELS)}")


def dry_run(args, vocab):
    offline_checks(vocab)
    rng = random.Random(args.seed)
    quotas = room_quotas(args.per_level, vocab)
    printed = 0
    for level in LEVELS:
        for room in ROOM_ORDER:
            spec = sample_spec(rng, level, room, vocab)
            spec["id"] = f"{level}-{room}-dry"
            spec["copies"] = needed_copies(spec["conditions"])
            print(json.dumps({
                "id": spec["id"],
                "level": level,
                "room": room,
                "scene": spec["scene"],
                "target_type": spec["target_type"],
                "text": spec["text"],
                "conditions": spec["conditions"],
                "copies": spec["copies"],
                "room_quota": quotas[room],
            }))
            printed += 1
            if printed >= args.dry_run:
                return


def verify(args):
    from load_task import load_task, make_controller

    path = Path(args.verify)
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not tasks:
        raise SystemExit(f"{path} has no tasks")
    controller = make_controller(scene=tasks[0]["scene"], platform=args.platform, width=args.width, height=args.height)
    try:
        for index, task in enumerate(tasks, start=1):
            load_task(controller, task)
            print(f"ok {index}/{len(tasks)} {task['id']}", flush=True)
    finally:
        controller.stop()


def offline_checks(vocab):
    quotas = room_quotas(50, vocab)
    assert quotas == {"kitchen": 15, "bedroom": 15, "living": 10, "bathroom": 10}
    conditions = {
        "state": {"property": "isOpen", "value": True, "text": "open"},
        "placement": "Desk",
        "next_to": "DeskLamp",
        "lights_off": False,
    }
    text = task_text("bedroom", "Laptop", conditions, vocab)
    assert text == "Find an open laptop on the desk next to the desk lamp in the bedroom.", text
    mug_on = {
        "objectType": "Mug", "objectId": "mug-a", "name": "mug-a",
        "position": {"x": 0, "y": 1, "z": 0}, "rotation": {"x": 0, "y": 0, "z": 0},
        "parentReceptacles": ["counter"], "visible": False,
        "isOpen": False, "isSliced": False, "isDirty": False, "isToggled": False,
    }
    mug_off = dict(mug_on, objectId="mug-b", name="mug-b", parentReceptacles=["table"],
                   position={"x": 2, "y": 1, "z": 0})
    counter = {"objectType": "CounterTop", "objectId": "counter", "name": "counter", "isOpen": True,
               "position": {"x": 0, "y": 0.9, "z": 0}}
    table = {"objectType": "DiningTable", "objectId": "table", "name": "table", "isOpen": True,
             "position": {"x": 2, "y": 0.8, "z": 0}}
    task = {
        "id": "test",
        "target_type": "Mug",
        "open_surfaces": ["CounterTop", "DiningTable"],
        "closed_containers": ["Fridge", "Cabinet", "Drawer", "Microwave", "Box", "Safe"],
        "near_threshold_m": 0.6,
        "conditions": {"state": None, "placement": "CounterTop", "next_to": None, "lights_off": False},
    }
    labeled = check_roles([mug_on, mug_off, counter, table], task)
    assert [item["role"] for item in labeled] == ["target", "distractor_for_placement"]
    try:
        check_roles([mug_on, dict(mug_on, objectId="mug-c", name="mug-c"), counter], task)
        raise AssertionError("two mugs on the counter should fail")
    except RoleError:
        pass
    laptop_open = dict(mug_on, objectType="Laptop", objectId="lap-a", name="lap-a", isOpen=True,
                       parentReceptacles=["desk"])
    laptop_shut = dict(laptop_open, objectId="lap-b", name="lap-b", isOpen=False,
                       position={"x": 0.3, "y": 1, "z": 0})
    desk = {"objectType": "Desk", "objectId": "desk", "name": "desk", "isOpen": True,
            "position": {"x": 0, "y": 0.7, "z": 0}}
    state_task = {
        "id": "test-state",
        "target_type": "Laptop",
        "open_surfaces": ["Desk"],
        "closed_containers": task["closed_containers"],
        "near_threshold_m": 0.6,
        "conditions": {
            "state": {"property": "isOpen", "value": True, "text": "open"},
            "placement": None,
            "next_to": None,
            "lights_off": False,
        },
    }
    labeled = check_roles([laptop_open, laptop_shut, desk], state_task)
    assert [item["role"] for item in labeled] == ["target", "distractor_for_state"]
    lamp = {"objectType": "DeskLamp", "objectId": "lamp", "name": "lamp", "isOpen": False,
            "position": {"x": 0, "y": 1, "z": 0}}
    shelf = {"objectType": "Shelf", "objectId": "shelf", "name": "shelf", "isOpen": True,
             "position": {"x": 0, "y": 1, "z": 0}}
    def lap(oid, x, parent, is_open):
        return dict(laptop_open, objectId=oid, name=oid, isOpen=is_open, parentReceptacles=[parent],
                    position={"x": x, "y": 1, "z": 0})
    four = {
        "id": "test-l4",
        "target_type": "Laptop",
        "open_surfaces": ["Desk", "Shelf"],
        "closed_containers": task["closed_containers"],
        "near_threshold_m": 0.6,
        "conditions": {
            "state": {"property": "isOpen", "value": True, "text": "open"},
            "placement": "Desk",
            "next_to": "DeskLamp",
            "lights_off": False,
        },
    }
    labeled = check_roles([
        lap("t", 0.1, "desk", True),
        lap("s", 0.2, "desk", False),
        lap("p", 0.1, "shelf", True),
        lap("n", 3.0, "desk", True),
        desk, lamp, shelf,
    ], four)
    assert [item["role"] for item in labeled] == [
        "target", "distractor_for_state", "distractor_for_placement", "distractor_for_next_to"
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate or verify AI2-THOR conditional search tasks.")
    parser.add_argument("--per-level", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="tasks_v1.jsonl")
    parser.add_argument("--vocab", default=str(Path(__file__).with_name("vocab.yaml")))
    parser.add_argument("--max-retries", type=int, default=30)
    parser.add_argument("--max-drops", type=int, default=200)
    parser.add_argument("--platform", default=None, help="AI2-THOR platform. Default: CloudRendering.")
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--dry-run", type=int, default=0, metavar="N",
                        help="Print N sampled tasks and exit. Does not start AI2-THOR.")
    parser.add_argument("--verify", default="", metavar="FILE",
                        help="Load every task in FILE and re-check its roles.")
    return parser.parse_args()


def main():
    args = parse_args()
    vocab = load_vocab(args.vocab)
    if args.dry_run:
        dry_run(args, vocab)
        return
    if args.verify:
        verify(args)
        return
    generate(args, vocab)


if __name__ == "__main__":
    main()
