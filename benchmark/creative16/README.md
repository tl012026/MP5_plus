# creative16-v2

Shared evaluation protocol for the 16 MP5 creative find-tasks. lfnav and the MP5 group use the same specs, seeds, scene builder, judge, and episode JSON fields.

This directory is an index. Task JSON, difficulty, seeds, labels, and the judge stay at the paths in [`manifest.yaml`](manifest.yaml). Do not copy those files.

Any change that alters the scene, the success rule, or the seed lists must bump the version (`creative16-v3`, …), retag, and add a changelog entry below.

## 16 tasks

Primary input: `lfnav/tasks/specs/*.json`. Original MP5 sentences in `MP5_agent/agent/tasks/creative/*.json` are contrast only.

`MP5_agent/agent/tasks/creative/creature.json` and `MP5_agent/agent/tasks/process/` are **out of scope**.

| Id | Original description | Target | Conditions (structured) | Episode biome / time / weather |
| --- | --- | --- | --- | --- |
| 1-1 | Find a tree | tree | — | forest / day / clear |
| 1-2 | Find a grass | grass | — | plains / day / clear |
| 1-3 | Find a cow | cow | — | plains / day / clear |
| 1-4 | Find a pig | pig | — | plains / day / clear |
| 2-1 | Find a tree in the forest | tree | biome=forest | forest / day / clear |
| 2-2 | Find a grass near the pig | grass | near=pig | plains / day / clear |
| 2-3 | Find a cow in the desert | cow | biome=desert | desert / day / clear |
| 2-4 | Find a pig during the night time | pig | time=night | plains / night / clear |
| 3-1 | Find a tree in the forest during the night time | tree | biome=forest, time=night | forest / night / clear |
| 3-2 | Find a grass near the pig in the plains | grass | near=pig, biome=plains | plains / day / clear |
| 3-3 | Find a cow in the desert during the daytime | cow | biome=desert, time=day | desert / day / clear |
| 3-4 | Find a pig during the night time, the weather is rainyday | pig | time=night, weather=rain | plains / night / rain |
| 4-1 | Find a tree in the forest during the night time, the weather is sunnyday | tree | biome=forest, time=night, weather=clear | forest / night / clear |
| 4-2 | Find a grass near the pig in the plains during the daytime | grass | near=pig, biome=plains, time=day | plains / day / clear |
| 4-3 | Find a cow near the water in the desert during the daytime, the weather is sunnyday | cow | near=water, biome=desert, time=day, weather=clear | desert / day / clear |
| 4-4 | Find a pig during the daytime on the plains with a grass next to it, the weather is sunnyday and the brightness is sufficient | pig | near=grass, biome=plains, time=day, weather=clear, brightness=sufficient | plains / day / clear |

Transcription notes (also in `lfnav/tasks/README.md`): sunnyday → `weather: clear`, rainyday → `rain`, daytime → `time: day`, night time → `night`. `near` objects must share the camera view with the target. 2-2 is “find grass with a pig in view”, not “find a pig near grass”.

## Difficulty and seeds

Source: `configs/lfnav/difficulty.yaml`. Spec files do not change with difficulty.

| | Mob spawn ring | Block-target min. start distance | Distractors |
| --- | --- | --- | --- |
| easy | 4–8, any yaw | 0 | 0 |
| medium | 8–16, any yaw | 8 | 2 |
| hard | 16–32, behind the agent ±45° | 16 | 4 |

Eval seeds: `configs/lfnav/eval_seeds_baseline.txt` — **9, 19, 29, 39, 49**. Training / demo seeds (`configs/lfnav/train_seeds_demo.txt`) are for lfnav C2 only; the MP5 group does not use them for this protocol.

Main env: `configs/lfnav/eval_dynamic.yaml` (inherits `baseline_eval.yaml` → `default.yaml`). Animals walk and natural mobs are kept. Static robustness: `configs/lfnav/baseline_eval.yaml` (`freeze_mobs` / `clear_entities` still true).

## Scene construction

Public entry: `lfnav.env.scene.make_scene_env(task_spec, difficulty, seed, cfg) -> (env, obs, initial_metrics)`.

Do **not** rely on MineDojo `initial_mobs`. HarvestMeta issues `/summon` from a reset hook, and that command does not actually spawn (observed 2026-11). lfnav summons **after** `reset()`.

Sequence inside `MineDojoEnv.reset` → `spawn_task_mobs` (flags from the loaded config):

1. Settle with real no-op ticks.
2. Clear field: `/kill @e[type=!player]` only if `clear_entities` is true. **v2 main eval leaves this off.**
3. Clear tall grass / double plants with chunked `/fill` (`clear_vegetation`), unless the task mentions grass.
4. One `/summon` per required mob (and distractors) on the difficulty ring, dropped from `spawn_drop_height`.
5. Freeze (slowness, gravity kept) only if `freeze_mobs` is true. **v2 main eval leaves this off**, so animals walk.
6. Settle after spawn.

When `clear_entities` is false, `initial_target_distance` (and SPL) is the nearest aligned/wide ray on that species. Eval and demo collection both call `apply_spawn_offset_distance` for this. The summon offset is still logged as `spawn_target_distance`. Static robustness still uses the offset as `initial_target_distance`.

Block targets (tree, grass, water) are not summoned. If a block target is closer than `min_initial_distance`, the world is remade with seed `seed*1000+k` up to `max_reroll` times. If the band still fails, the episode is `difficulty_unmet` and still recorded.

**Near-X approximation:** the target and each `conditions.near` name must both be present in the same camera view (`SceneState` belief ≥ 0.5). This is not a Euclidean “next to” test.

## Success (judge)

Source: `lfnav/gt/judge.py`. Radius default 3 blocks (`judge.radius`).

- **SR-b (primary):** SR-a holds **and** the nearest **aligned-ray** hit on the target is ≤ radius. The target must be in the camera frame. Wide lidar is not used for this check.
- **SR-a:** target present in view, each `near` object present in that same view, and biome / time / weather / brightness match when the spec sets them.

Other recorded flags: `false_stop` (agent stopped and SR-a failed), `timeout` (hit `max_steps` without stopping). SPL uses `l = max(initial_target_distance - radius, 0)` and is empty when the start distance is beyond lidar range (`"> N"`).

Episode JSON must include the fields in [`result_schema.json`](result_schema.json). `task` / `policy` are the protocol names; `run.py` also writes `task_name` / `policy_name`.

## Known limits

- **4-3 water is not placed.** The spec lists water as a block spawn item; the biome is expected to provide it. Desert often has none, so SR-a `near:water` may never hold.
- **Block targets on medium/hard** are `difficulty_unmet` when rerolls cannot push `initial_target_distance` to the band (trees/grass/water already in the generated world).

## MP5 hookup

Do not duplicate spawn commands. In this version, do not edit `MP5_agent`; later, point `env_wrapper` at the same function.

1. Load a spec from `lfnav/tasks/specs/` and `configs/lfnav/eval_dynamic.yaml` (or `baseline_eval.yaml` for the static control).
2. `env, obs, initial_metrics = make_scene_env(task_spec, difficulty, seed, cfg)`.
3. Run your agent on that `env`.
4. Score the stop with `lfnav.gt.judge.judge_episode` (or `judge_stop`).
5. Write one JSON per episode with every required key in `result_schema.json` (include `initial_metrics["initial_target_distance"]` and `difficulty_unmet`).

## Changelog

### creative16-v2

- Main evaluation is the **dynamic** scene: `eval_dynamic.yaml` (`freeze_mobs: false`, `clear_entities: false`, `clear_vegetation: true`). Animals walk; natural mobs stay.
- Static scene (`baseline_eval.yaml`) is kept as a robustness control. Old result directories are unchanged.
- With `clear_entities: false`, `initial_target_distance` and SPL use the nearest same-species opening-ray distance (same function as demo collection).

### creative16-v1

- First shared snapshot: 16 specs, difficulty yaml, eval seeds 9/19/29/39/49, `make_scene_env`, judge SR-b/SR-a, result schema.
- Tag: `creative16-v1`.
