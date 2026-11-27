"""Restore one generated task into a running AI2-THOR controller and re-check roles.

    from load_task import make_controller, load_task
    controller = make_controller()
    load_task(controller, task)
"""

import os

CLOSED_FALLBACK = ("Fridge", "Cabinet", "Drawer", "Microwave", "Box", "Safe")


class RoleError(RuntimeError):
    pass


def make_controller(scene="FloorPlan1", platform=None, width=300, height=300):
    """Headless AI2-THOR. Default platform is CloudRendering (override with AI2THOR_PLATFORM)."""
    from ai2thor.controller import Controller

    platform = platform or os.environ.get("AI2THOR_PLATFORM", "CloudRendering")
    return Controller(
        platform=platform,
        scene=scene,
        width=int(width),
        height=int(height),
        fieldOfView=90,
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,
        visibilityDistance=1.5,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )


def reset_scene(controller, scene):
    controller.reset(scene=scene)
    return controller.last_event


def same_type(obj, target_type):
    kind = obj.get("objectType") or ""
    return kind == target_type or kind == f"{target_type}Sliced"


def _by_id(objects):
    return {o["objectId"]: o for o in objects}


def parent_objects(obj, by_id):
    return [by_id[i] for i in (obj.get("parentReceptacles") or []) if i in by_id]


def on_open_surface(obj, objects, open_surfaces, closed_containers):
    """True when a parent is an open surface and no parent is a shut container."""
    by_id = _by_id(objects)
    parents = parent_objects(obj, by_id)
    if any(p.get("objectType") in closed_containers and not p.get("isOpen") for p in parents):
        return False
    return any(p.get("objectType") in open_surfaces for p in parents)


def has_parent_type(obj, objects, receptacle_type):
    by_id = _by_id(objects)
    return any(p.get("objectType") == receptacle_type for p in parent_objects(obj, by_id))


def horizontal_distance(a, b):
    dx = float(a["x"]) - float(b["x"])
    dz = float(a["z"]) - float(b["z"])
    return (dx * dx + dz * dz) ** 0.5


def nearest_anchor_distance(obj, objects, anchor_type):
    anchors = [o for o in objects if o.get("objectType") == anchor_type]
    if not anchors:
        return None
    return min(horizontal_distance(obj["position"], a["position"]) for a in anchors)


def condition_flags(obj, objects, conditions, threshold):
    """Fixed placement / next-to flags, plus the state flag when the task has a state."""
    flags = {}
    placement = conditions.get("placement")
    nxt = conditions.get("next_to")
    state = conditions.get("state")
    if placement:
        flags["placement"] = has_parent_type(obj, objects, placement)
    if nxt:
        dist = nearest_anchor_distance(obj, objects, nxt)
        flags["next_to"] = dist is not None and dist <= float(threshold)
    if state:
        flags["state"] = bool(obj.get(state["property"])) == bool(state.get("value", True))
    return flags


def expected_roles(conditions):
    roles = ["target"]
    if conditions.get("state"):
        roles.append("distractor_for_state")
    if conditions.get("placement"):
        roles.append("distractor_for_placement")
    if conditions.get("next_to"):
        roles.append("distractor_for_next_to")
    return roles


def _role_flags(role, conditions):
    """What each condition must be for this role. A distractor flips only its own condition."""
    flags = {}
    if conditions.get("placement"):
        flags["placement"] = role != "distractor_for_placement"
    if conditions.get("next_to"):
        flags["next_to"] = role != "distractor_for_next_to"
    if conditions.get("state"):
        flags["state"] = role != "distractor_for_state"
    return flags


def check_roles(objects, task):
    """Label every target-type instance. Raise RoleError if the labels are not exactly the intended roles.

    Returns the labeled instance dicts (target first).
    """
    conditions = task["conditions"]
    target_type = task["target_type"]
    open_surfaces = task["open_surfaces"]
    closed = task.get("closed_containers") or list(CLOSED_FALLBACK)
    threshold = task["near_threshold_m"]
    wanted = expected_roles(conditions)
    group = [o for o in objects if same_type(o, target_type)]
    if len(group) != len(wanted):
        raise RoleError(
            f"{task.get('id', target_type)}: found {len(group)} {target_type} instances, want {len(wanted)}"
        )
    for obj in group:
        if not on_open_surface(obj, objects, open_surfaces, closed):
            raise RoleError(
                f"{task.get('id', target_type)}: {obj.get('objectId')} is not on an open surface"
            )
    unused = list(group)
    labeled = []
    for role in wanted:
        need = _role_flags(role, conditions)
        match = None
        for obj in unused:
            if condition_flags(obj, objects, conditions, threshold) == need:
                match = obj
                break
        if match is None:
            raise RoleError(f"{task.get('id', target_type)}: no instance for role {role}")
        unused.remove(match)
        labeled.append(_instance_record(match, role))
    if unused:
        raise RoleError(f"{task.get('id', target_type)}: extra instance {unused[0].get('objectId')}")
    if conditions.get("lights_off"):
        switches = [o for o in objects if o.get("objectType") == "LightSwitch"]
        if not switches or any(o.get("isToggled") for o in switches):
            raise RoleError(f"{task.get('id', target_type)}: lights are not off")
    return labeled


def _instance_record(obj, role):
    return {
        "objectName": obj.get("name"),
        "objectId": obj.get("objectId"),
        "objectType": obj.get("objectType"),
        "role": role,
        "position": obj.get("position"),
        "rotation": obj.get("rotation"),
        "states": {
            "isOpen": obj.get("isOpen"),
            "isSliced": obj.get("isSliced"),
            "isDirty": obj.get("isDirty"),
            "isToggled": obj.get("isToggled"),
        },
    }


def object_poses(event):
    """Prefer GetObjectPoses. Fall back to name/position/rotation from object metadata."""
    returned = event.metadata.get("actionReturn")
    if event.metadata.get("lastActionSuccess") and isinstance(returned, list) and returned:
        if isinstance(returned[0], dict) and "objectName" in returned[0]:
            return returned
    poses = []
    for obj in event.metadata.get("objects") or []:
        poses.append({
            "objectName": obj.get("name"),
            "position": obj.get("position"),
            "rotation": obj.get("rotation"),
        })
    return poses


def find_by_name(objects, object_name):
    for obj in objects:
        if obj.get("name") == object_name or obj.get("objectId") == object_name:
            return obj
    return None


def receptacle_ids(objects, types):
    return sorted(o["objectId"] for o in objects if o.get("objectType") in types)


def _require(event, what):
    if not event.metadata.get("lastActionSuccess", False):
        message = event.metadata.get("errorMessage") or "action failed"
        raise RuntimeError(f"{what}: {message}")
    return event


def load_task(controller, task):
    """Reset the scene, replay the spawn and state actions, restore poses, lights, and the agent.

    Raises RoleError if the metadata roles do not match the task.
    """
    reset_scene(controller, task["scene"])
    spawn = task["spawn"]
    exclude = receptacle_ids(controller.last_event.metadata["objects"], spawn["excludedReceptacleTypes"])
    event = controller.step(
        action="InitialRandomSpawn",
        randomSeed=int(spawn["randomSeed"]),
        forceVisible=False,
        numPlacementAttempts=int(spawn.get("numPlacementAttempts", 5)),
        placeStationary=True,
        numDuplicatesOfType=list(spawn.get("numDuplicatesOfType") or []),
        excludedReceptacles=exclude,
    )
    _require(event, f"{task['id']} InitialRandomSpawn")
    for act in task.get("state_actions") or []:
        obj = find_by_name(controller.last_event.metadata["objects"], act["objectName"])
        if obj is None:
            raise RuntimeError(f"{task['id']}: {act['objectName']} is missing before {act['action']}")
        event = controller.step(action=act["action"], objectId=obj["objectId"], forceAction=True)
        _require(event, f"{task['id']} {act['action']}")
    event = controller.step(
        action="SetObjectPoses",
        objectPoses=task["object_poses"],
        placeStationary=True,
    )
    _require(event, f"{task['id']} SetObjectPoses")
    for light in task.get("lights") or []:
        obj = find_by_name(controller.last_event.metadata["objects"], light["objectName"])
        if obj is None:
            raise RuntimeError(f"{task['id']}: light {light['objectName']} is missing")
        want = bool(light["isToggled"])
        if bool(obj.get("isToggled")) == want:
            continue
        action = "ToggleObjectOn" if want else "ToggleObjectOff"
        event = controller.step(action=action, objectId=obj["objectId"], forceAction=True)
        _require(event, f"{task['id']} {action}")
    agent = task["agent"]
    event = controller.step(
        action="Teleport",
        position=agent["position"],
        rotation=agent["rotation"],
        horizon=agent["horizon"],
        standing=agent.get("standing", True),
    )
    _require(event, f"{task['id']} Teleport")
    check_roles(controller.last_event.metadata["objects"], task)
    return controller.last_event
