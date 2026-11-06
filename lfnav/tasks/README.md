# Track A task specs

Each file in `specs/` transcribes one subgoal from `MP5_agent/agent/tasks/creative/<id>.json`. The original files are not modified.

## Transcription rules

- `target` is the thing the sentence asks the agent to find.
- `conditions.near` lists other objects that must be in the same camera view as the target. In the MP5 wording, "grass" is the grass block, "water" is a water block, and "tree" is wood or leaves (see `lfnav/labels.yaml` aliases).
- `sunnyday` is `weather: clear`. `rainyday` is `weather: rain`. `daytime` is `time: day`. `night time` is `time: night`.
- `brightness: sufficient` is copied only when the sentence says the brightness is sufficient.
- A null condition means the sentence does not constrain it.

## Episode biome

`episode` is the scene we build for evaluation, not a MineDojo observation key.

When the sentence names a biome, that biome is `specified_biome`. When it does not, the default is `forest` for a tree target and `plains` otherwise, so the target can exist. Animals named as the target or as a `near` condition are listed in `spawn`. Blocks (tree, grass, water, sand) are listed with `"kind": "block"`. Mobs use `"kind": "mob"`.

`start_time` is the time named in the sentence, otherwise `day`. `initial_weather` is the weather named in the sentence, otherwise `clear`.

Difficulty (spawn distance, distractors, whether the agent must leave the starting biome) lives in `configs/lfnav/difficulty.yaml` and does not change these files.

## Open questions

- 2-2 says "Find a grass near the pig". The target is grass and the condition is a pig in view. It is not "find a pig near grass".
- 4-2 in the task statement example (pig in a forest) does not match `4-2.json`. The spec follows the file: grass near a pig, on the plains, during the day.
- Sentences that omit the biome do not say where to start. The forest/plains default above is a scene choice.
