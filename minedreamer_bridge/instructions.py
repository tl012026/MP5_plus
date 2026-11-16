"""MP5 workflow action -> MineDreamer instruction, step budget, and done-condition.

Prompts marked in_distribution=True are the exact text prompts of MineDreamer's own
programmatic evaluation (minedreamer/play/config/programmatic/*.yaml). Anything else
is free text that the Imaginator was not evaluated on, so the result file flags it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Exact prompts from the MineDreamer programmatic configs.
MINEDREAMER_EVAL_PROMPTS = {
    "chop a tree", "collect dirt", "break grass", "craft wooden planks",
    "build a tower", "dig down", "go explore", "look at the sky", "swim", "stay in water",
}

# MP5 object name (after update_inventory_obj_name) -> MineDreamer prompt for mine().
_MINE_PROMPTS = {
    "log": "chop a tree",
    "dirt": "collect dirt",
    "grass": "break grass",
}

# MineRL 1.16 item ids -> MP5 inventory names (utils/common_utils.py spelling).
_MINERL_TO_MP5 = {
    "oak_log": "log", "birch_log": "log", "spruce_log": "log", "jungle_log": "log",
    "acacia_log": "log", "dark_oak_log": "log",
    "oak_planks": "planks", "birch_planks": "planks", "spruce_planks": "planks",
    "jungle_planks": "planks", "acacia_planks": "planks", "dark_oak_planks": "planks",
    "cobblestone": "cobblestone", "dirt": "dirt", "wheat_seeds": "wheat seeds",
    "crafting_table": "crafting table", "stick": "stick", "iron_ore": "iron ore",
    "coal": "coal", "diamond": "diamond",
}

# Actions MP5 leaves as no-ops (see README "do not change the game").
NOOP_ACTIONS = {"fight", "equip", "apply", "dig_up"}

DEFAULT_BUDGETS = {"find": 1200, "move_to": 600, "mine": 1200, "dig_down": 1200, "craft": 600}


def mp5_item_name(minerl_name: str) -> str:
    return _MINERL_TO_MP5.get(minerl_name, minerl_name.replace("_", " "))


def mp5_inventory(minerl_inventory: dict) -> dict:
    out: dict = {}
    for name, count in (minerl_inventory or {}).items():
        if not count:
            continue
        key = mp5_item_name(name)
        out[key] = out.get(key, 0) + int(count)
    return out


def _inventory_obj_name(obj: str) -> str:
    # Same rule as utils.common_utils.update_inventory_obj_name (copied so this imports without langchain).
    return {"wood": "log", "tree": "log", "stone": "cobblestone", "diamond ore": "diamond"}.get(obj, obj)


def _find_obj_name(obj: str) -> str:
    return {"log": "wood", "tree": "wood", "cobblestone": "stone", "diamond": "diamond ore"}.get(obj, obj)


@dataclass
class Instruction:
    action: str
    text_prompt: str | None
    in_distribution: bool
    max_steps: int
    done: dict = field(default_factory=dict)   # {"kind": ..., ...} evaluated by the performer
    supported: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "text_prompt": self.text_prompt,
            "in_distribution": self.in_distribution,
            "max_steps": self.max_steps,
            "done": self.done,
            "supported": self.supported,
            "note": self.note,
        }


def instruction_for(action: dict, budgets: dict | None = None) -> Instruction:
    """Map one MP5 workflow action {"name": ..., "args": {...}} to a MineDreamer instruction."""
    budgets = {**DEFAULT_BUDGETS, **(budgets or {})}
    name = action.get("name")
    args = action.get("args") or {}

    if name == "find":
        obj = _find_obj_name(str(args.get("obj")))
        target = "tree" if obj == "wood" else obj
        prompt = f"find a {target}"
        return Instruction("find", prompt, prompt in MINEDREAMER_EVAL_PROMPTS, budgets["find"],
                           {"kind": "visible", "obj": obj},
                           note="MineDreamer has no 'find' prompt; stop comes from perception, not from the controller.")
    if name == "move_to":
        obj = _find_obj_name(str(args.get("obj")))
        target = "tree" if obj == "wood" else obj
        prompt = f"go to the {target}"
        return Instruction("move_to", prompt, False, budgets["move_to"],
                           {"kind": "distance_le", "obj": obj, "blocks": 3.0})
    if name == "mine":
        item = _inventory_obj_name(str(args.get("obj")))
        prompt = _MINE_PROMPTS.get(item, f"mine {args.get('obj')}")
        return Instruction("mine", prompt, prompt in MINEDREAMER_EVAL_PROMPTS, budgets["mine"],
                           {"kind": "inventory_gain", "item": item, "count": 1})
    if name == "dig_down":
        y = args.get("y_level")
        return Instruction("dig_down", "dig down", True, budgets["dig_down"],
                           {"kind": "y_le", "y": None if y is None else float(y)})
    if name == "craft":
        obj = args.get("obj") or {}
        item = next(iter(obj), None)
        count = int(next(iter(obj.values()), 1)) if obj else 1
        if item == "planks":
            return Instruction("craft", "craft wooden planks", True, budgets["craft"],
                               {"kind": "inventory_gain", "item": "planks", "count": count})
        return Instruction("craft", None, False, 0, {}, supported=False,
                           note=f"STEVE-1 is not evaluated on crafting {item!r}; hand this action to the scripted MP5 performer.")
    if name in NOOP_ACTIONS:
        return Instruction(name, None, False, 0, {}, supported=False,
                           note="No-op in the original MP5 performer; kept as a no-op.")
    return Instruction(str(name), None, False, 0, {}, supported=False, note="Unknown action name.")
