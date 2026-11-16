# Expert demonstrations

The expert is C1 plus ground-truth perception plus the 360° wide lidar (`lfnav/policy/expert.py`). It is not used at evaluation time.

Three demo sets, all collected with `configs/lfnav/main.yaml` (`clear_vegetation: false`):

| Set | Where | Role |
|---|---|---|
| `demos_main` | `$SCR/lfnav/data/demos_main` | Old data, used for A only |
| `demos_main_v3` | `$SCR/lfnav/data/demos_main_v3` | Easy, used to train C. 44 training seeds, easy cells only |
| `demos_v4` | `$SCR/lfnav/data/demos_v4` | Medium, max 500 steps |

Hard collection is planned with a max of 1500 steps. It is not in `demos_v4`.

Training on C keeps successful v3 episodes. `hostile_seen == 0` is the filter used for the older main-scene set (and `exclude_tasks`, default 4-3). `peaceful_sent` (legacy record field `peaceful_ok`) only means `/difficulty peaceful` was sent; it is not the training filter.

Wide lidar is a **search** cue only. The expert uses it when `match.satisfied.target` is false (the target is not on an aligned ray). If the target is already in the camera frame, C1 runs even when a condition still fails. The previous gate (`status != found`) also fired on missing `near` conditions, so the expert kept turning toward the wide hit and never looked for the other object.

Tasks that need this: **2-2, 3-2, 4-2, 4-3, 4-4** (grass near pig, pig near grass, cow near water). Demos collected before this fix on those tasks should be recaptured.

## Vegetation is not cleared

`configs/lfnav/main.yaml` sets `clear_vegetation: false`. Tall grass, plants and flowers stay in the world.

Rays stop at the first block, and entity rays are also blocked by blocks. An animal behind tall grass is invisible to both the GT and the judge. MineDojo ray names: `grass` = tallgrass, `plant` = double_plant, `flower` = small flower, `grass block` = ground. The occluder list is `occluder_names` in `lfnav/labels.yaml`. `grass block` is not an occluder.

`minedojo.clear_entities: false` leaves natural mobs in the world. When entities are not cleared, a natural animal of the target species can stand closer than the summoned offset. The logged `initial_target_distance` is then the nearest aligned or wide ray on that species. `spawn_target_distance` is the offset, and `natural_target_closer` is true when the ray is clearly shorter.

## Scene fields

Each decision's `scene` is the GT `SceneState`. Besides the per-class beliefs, v3 and v4 scenes store:

| Field | Meaning |
|---|---|
| `occluder` | Cells whose ray hit a name in `occluder_names` |
| `patch_dist` | Closest hit distance in each cell. Null if that ray missed |
| `patch_dist_by_class` | Per class, closest **visible** hit only. The visible hit is the first hit on the ray |
| `occluded_by_class` | Per class, cells where that class is behind only occluders. Always empty in MineDojo, because entity rays do not pass through blocks |
| `hit_names` | `{name: count}` of `ray_raw` names over all aligned rays, including subrays. Blocks and entities both count |

`demos_main` is the old set and does not have these fields.

## Run

Login-node check:

```bash
python3 -m unittest discover -s lfnav/tests
```

Array job (16 tasks × 3 difficulties; tree/grass medium/hard exit immediately):

```bash
source $SCR/mp5_env.sh
export MP5=<run copy>
git -C "$MP5" log -n 1
cd "$MP5"
mkdir -p $SCR/lfnav/logs
export CFG=$MP5/configs/lfnav/main.yaml
export SEEDS=$MP5/configs/lfnav/train_seeds_demo.txt
export OUT=$SCR/lfnav/data/demos_main_v3
test -f "$CFG" && test -f "$SEEDS"
sbatch --export=ALL lfnav/slurm/collect_demos_array.sbatch
```

Overrides: `CFG`, `SEEDS`, `OUT`. For the medium set, point `OUT` at `$SCR/lfnav/data/demos_v4`. Failures keep `record.json` and `steps.jsonl` only. Legacy dynamic demos stay under `demos_dynamic`.

Seeds are 1–48 except those ending in 9 (44 seeds). Finished `record.json` files are skipped.

Summarize:

```bash
python3 -m lfnav.data.demo_stats --in $SCR/lfnav/data/demos_main_v3 --out results/demos/stats_v3.md
```

## Layout

`<OUT>/{success|failure}/<task>/<difficulty>/<seed>/`

| File | Contents |
| --- | --- |
| `record.json` | Same fields as an eval episode, plus `config_name` |
| `steps.jsonl` | One decision per line: `decision`, `env_step`, `match`, `scene` (GT, including the fields above), 8-sector visit counts, `recent_actions`, expert `action_chunk`, `used_wide` |
| `frames/NNNNN.jpg` | RGB at that decision, resized to 224×364, JPEG quality 90 |

`used_wide` is true when the expert steered from the wide lidar because aligned rays did not yet see the target.
