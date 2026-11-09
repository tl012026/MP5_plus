# lfnav

LLM-free conditional goal navigation for the 16 creative tasks in MP5. No language model is called on the control path. MineLLM appears only as a perception baseline, queried with fixed templates.

## Modules

```
rgb frame --> A perceive --> SceneState
TaskSpec + SceneState --> B match --> MatchResult
SceneState + MatchResult + visit grid + recent actions --> C act --> action chunk or waypoints
action chunk --> controller --> environment
stop --> judge (SR-b, SR-a)
```

| New module | Replaces in MP5 |
| --- | --- |
| A `lfnav/perception` | Percipient (MineLLM), without GPT-4 asking the questions |
| B `lfnav/matcher.py` | The part of the percipient prompt that decides the object is found |
| C0 / C1 / C2 | The scripted explore branch inside the find action |
| controller | `structured_actions.py` movement, reimplemented on the MineDojo voxel crop (`voxel_radius` 5 → 11³) |
| judge | A new automatic check. MP5's original success rule is SR-a |

C3 is `lfnav/policy/c3.py`. Its `act` raises `NotImplementedError`. The controller already accepts `(dyaw_deg, forward_blocks)` waypoints.

## Privileged information

- The expert, the judge and initial-difficulty logging use the wide lidar and the ground-truth scene state. Evaluation policies never receive them, except when GT perception is the backend under test.
- `initial_target_visible` / `initial_target_distance` are logged only and never enter `PolicyInput`.
- The voxel crop is a contact sensor for the controller. C0, C1 and C2 do not read it.
- Position is odometry from the environment position field.
- When the target is not in the camera frame, B reports `direction = None` and `distance = None`.

`near X` means X and the target appear in the same camera frame. This approximates the original sentence.

SR-b distance is the nearest aligned-ray hit on the target at stop, so the target must be in the camera frame. The fake env keeps a Euclidean `distance_to` for tests.

SPL is `success * l / max(path, l)` with `l = max(initial_target_distance - judge.radius, 0)`. It ignores terrain detours. When the start distance is beyond lidar range (`"> 32"`), SPL is empty and counted separately.

### Difficulty (`configs/lfnav/difficulty.yaml`)

| | Mob spawn ring | Block target min. start distance | Distractors |
| --- | --- | --- | --- |
| easy | 4–8 | 0 | 0 |
| medium | 8–16 | 8 | 2 |
| hard | 16–32, sector behind the agent | 16 | 4 |

- Block targets reroll the world seed (`seed*1000+k`, up to `max_reroll`) until the minimum distance holds. Otherwise the episode is marked `difficulty_unmet`.
- `specified_biome` worlds are assumed to be single-biome, so `leave_biome` is false. Confirm this with step 0 of the baseline.
- Optional mixed-biome hard seeds come from `lfnav/env/search_biome_seeds.py`.

### Camera

- Aligned rays use one ray per DINOv2 patch (224×364