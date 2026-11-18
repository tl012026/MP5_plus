"""Minecraft 1.16.5 (MineRL 1.0) side of the version split.

MP5's baseline runs in MineDojo (MC 1.11.2); STEVE-1/VPT only work in MineRL (MC 1.16.5).
The MineDreamer track therefore keeps the world in 1.16 and translates at the edges:

* item names        MP5 text names <-> 1.16 ids (oak_log, oak_planks, crafting_table ...)
* task setup        MP5 env config + lfnav episode conditions -> HumanSurvival kwargs + chat commands
* crafting          MineDojo crafts with an event-level action (instant if the recipe holds).
                    STEVE-1 cannot craft reliably, so CommandCrafter does the same thing with
                    recipe-checked /clear + /give commands. Same semantics as the baseline.
* env adapter       MineRLEnv wraps HumanSurvival with observation()/execute_cmd() (needs minerl)

Differences that cannot be translated are listed in UNTRANSLATABLE and copied into run metadata.
"""

from __future__ import annotations

import math

# MP5 name -> canonical 1.16 id used for /give
MP5_TO_ID = {
    "log": "oak_log", "planks": "oak_planks", "stick": "stick", "crafting table": "crafting_table",
    "wooden pickaxe": "wooden_pickaxe", "stone pickaxe": "stone_pickaxe", "iron pickaxe": "iron_pickaxe",
    "cobblestone": "cobblestone", "furnace": "furnace", "iron ore": "iron_ore", "iron ingot": "iron_ingot",
    "coal": "coal", "diamond": "diamond", "dirt": "dirt",
}

_WOODS = ("oak", "birch", "spruce", "jungle", "acacia", "dark_oak")
# MP5 name -> every 1.16 id that counts as that item when consuming
ANY_OF = {
    "log": [f"{w}_log" for w in _WOODS],
    "planks": [f"{w}_planks" for w in _WOODS],
    "coal": ["coal", "charcoal"],
}

# (output MP5 name) -> (yield per craft, {material: count per craft}, platform)
RECIPES = {
    "planks": (4, {"log": 1}, None),
    "stick": (4, {"planks": 2}, None),
    "crafting table": (1, {"planks": 4}, None),
    "wooden pickaxe": (1, {"planks": 3, "stick": 2}, "crafting table"),
    "stone pickaxe": (1, {"cobblestone": 3, "stick": 2}, "crafting table"),
    "furnace": (1, {"cobblestone": 8}, "crafting table"),
    "iron pickaxe": (1, {"iron ingot": 3, "stick": 2}, "crafting table"),
    "iron ingot": (1, {"iron ore": 1, "coal": 1}, "furnace"),   # smelting, one coal per ingot (simplified)
}

UNTRANSLATABLE = [
    "break_speed_multiplier: MineDojo x100 has no MineRL equivalent; the MineDreamer track uses vanilla speed (configs/strict.yaml semantics).",
    "voxels / lidar: not in MineRL observations, so the scripted MP5 move_to/mine cannot run; perception decides find/move_to.",
    "world generation: same seed gives different worlds in 1.11 and 1.16; seeds are not comparable across tracks.",
    "smelting fuel is simplified to one coal per ingot.",
]


def ids_for(mp5_name: str) -> list[str]:
    return ANY_OF.get(mp5_name, [MP5_TO_ID.get(mp5_name, mp5_name.replace(" ", "_"))])


def count_mp5(minerl_inventory: dict, mp5_name: str) -> int:
    return sum(int(minerl_inventory.get(i, 0) or 0) for i in ids_for(mp5_name))


# ------------------------------------------------------------------ task setup
TIME_TICKS = {"day": 6000, "night": 18000}


def task_setup(env_cfg: dict | None = None, episode: dict | None = None, seed: int | None = None) -> dict:
    """MP5 config `env` block + optional lfnav-style `episode` (specified_biome/start_time/
    initial_weather/spawn) -> what MineRLEnv needs."""
    env_cfg = env_cfg or {}
    episode = episode or {}
    inventory = []
    for item in env_cfg.get("start_inventory") or []:
        name = item["name"].replace("_", " ")
        inventory.append({"type": MP5_TO_ID.get(name, item["name"]), "quantity": int(item["quantity"])})
    merged: dict = {}
    for row in inventory:                       # MineDojo uses slots; MineRL wants one row per type
        merged[row["type"]] = merged.get(row["type"], 0) + row["quantity"]
    biome = episode.get("specified_biome") or env_cfg.get("specified_biome")
    # Command feedback is drawn as chat text over the frame (seen in the 2026-11-16 MineRL smoke test);
    # turn it off first so STEVE-1, MineCLIP and the percipient see a clean picture.
    cmds = ["/gamerule sendCommandFeedback false"]
    start_time = episode.get("start_time") or ("night" if env_cfg.get("start_at_night") else None)
    if start_time:
        cmds += [f"/time set {TIME_TICKS[start_time]}", "/gamerule doDaylightCycle false"]
    weather = episode.get("initial_weather")
    if weather:
        cmds += [f"/weather {'clear' if weather == 'clear' else 'rain'} 1000000", "/gamerule doWeatherCycle false"]
    for i, item in enumerate(episode.get("spawn") or []):
        dz = 6 + 2 * i
        if item.get("kind") == "mob":
            cmds.append(f"/execute as @p at @p run summon minecraft:{item['name']} ~0 ~ ~{dz} {{Age:0}}")
        else:
            cmds.append(f"/execute as @p at @p run setblock ~2 ~-1 ~{dz} minecraft:{item['name']}")
    return {
        "inventory": [{"type": k, "quantity": v} for k, v in merged.items()] or None,
        "preferred_spawn_biome": biome,
        "init_commands": cmds,
        "seed": seed,
        "untranslatable": list(UNTRANSLATABLE),
    }


# ------------------------------------------------------------------ crafting
class CommandCrafter:
    """Recipe-checked crafting through chat commands (the 1.16 stand-in for MineDojo's craft action)."""

    name = "command_craft"

    def __init__(self, env, max_settle_steps: int = 40):
        self.env = env
        self.max_settle_steps = int(max_settle_steps)

    def plan(self, action: dict, inventory: dict) -> dict:
        args = action.get("args") or {}
        obj = args.get("obj") or {}
        if not obj:
            return {"ok": False, "reason": "craft action has no obj"}
        item, want = next(iter(obj.items()))
        if item not in RECIPES:
            return {"ok": False, "reason": f"no recipe for {item!r} in mc116.RECIPES"}
        per, materials, platform = RECIPES[item]
        times = max(1, math.ceil(int(want) / per))
        if platform and count_mp5(inventory, platform) < 1:
            return {"ok": False, "reason": f"{platform} is required and not in the inventory"}
        clears = []
        for mat, n in materials.items():
            need = n * times
            have = count_mp5(inventory, mat)
            if have < need:
                return {"ok": False, "reason": f"need {need} {mat}, have {have}"}
            for item_id in ids_for(mat):           # consume variants in order
                take = min(need, int(inventory.get(item_id, 0) or 0))
                if take:
                    clears.append(f"/clear @p minecraft:{item_id} {take}")
                    need -= take
                if need == 0:
                    break
        give = f"/give @p minecraft:{MP5_TO_ID.get(item, item.replace(' ', '_'))} {per * times}"
        return {"ok": True, "item": item, "count": per * times, "commands": clears + [give]}

    def craft(self, action: dict, obs: dict) -> dict:
        inventory = (obs or {}).get("inventory") or {}
        plan = self.plan(action, inventory)
        if not plan["ok"]:
            return {"ok": False, "reason": plan["reason"], "commands": [], "obs": obs}
        before = count_mp5(inventory, plan["item"])
        for cmd in plan["commands"]:
            self.env.execute_cmd(cmd)
        # MineRL applies chat commands on later ticks; poll a few no-op steps (smoke test 2026-11-16)
        for _ in range(self.max_settle_steps):
            obs = self.env.settle()
            after = count_mp5(obs.get("inventory") or {}, plan["item"])
            if after - before >= plan["count"]:
                break
        ok = after - before >= plan["count"]
        return {"ok": ok, "reason": "" if ok else f"{plan['item']} did not appear after the commands",
                "commands": plan["commands"], "obs": obs}


# ------------------------------------------------------------------ env adapter
class MineRLEnv:
    """HumanSurvival behind the interface DreamerPerformer uses. Import-time safe; needs minerl to run."""

    name = "minerl_1.0_humansurvival"

    def __init__(self, raw_env=None, setup: dict | None = None):
        self.setup = setup or {}
        if raw_env is None:
            from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
            try:
                from minedreamer.VPT.agent import ENV_KWARGS
            except ImportError:
                ENV_KWARGS = {}
            raw_env = HumanSurvival(**ENV_KWARGS, inventory=self.setup.get("inventory"),
                                    preferred_spawn_biome=self.setup.get("preferred_spawn_biome")).make()
            raw_env.reset()
            if self.setup.get("seed") is not None:
                raw_env.seed(int(self.setup["seed"]))
        self._env = raw_env
        self._obs = None

    def reset(self, **_kwargs):
        for cmd in self.setup.get("init_commands") or []:
            self._env.execute_cmd(cmd)
        return self.settle(5)

    def settle(self, n: int = 1):
        for _ in range(n):
            self._obs, _r, _d, _i = self._env.step(self._env.action_space.no_op())
        return self._obs

    def execute_cmd(self, cmd: str):
        return self._env.execute_cmd(cmd)

    def step(self, action):
        self._obs, reward, done, info = self._env.step(action)
        return self._obs, reward, done, info

    def observation(self):
        return self._obs

    def close(self):
        self._env.close()
