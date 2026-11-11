# Expert demonstrations for C2

The expert is C1 plus ground-truth perception plus the 360° wide lidar (`lfnav/policy/expert.py`). It is not used at evaluation time. Dynamic-scene demos keep natural animals moving; they are the C2 training set.

C2 keeps successful main-scene demos with `hostile_seen == 0` (and drops `exclude_tasks`, default 4-3). `peaceful_sent` (legacy record field `peaceful_ok`) only means `/difficulty peaceful` was sent; it is not the training filter.

Wide lidar is a **search** cue only. The expert uses it when `match.satisfied.target` is false (the target is not on an aligned ray). If the target is already in the camera frame, C1 runs even when a condition still fails. The previous gate (`status != found`) also fired on missing `near` conditions, so the expert kept turning toward the wide hit and never looked for the other object.

Tasks that need this: **2-2, 3-2, 4-2, 4-3, 4-4** (grass near pig, pig near grass, cow near water). Demos collected before this fix on those tasks should be recaptured.

## Why vegetation is still cleared

`minedojo.clear_entities: false` leaves natural mobs in the world. Tall grass is different: a pig is about 0.9 blocks tall and plains tall grass is about 0.8, so eye-height GT rays miss the animal. Training labels would then be worse than the RGB. `demo_dynamic.yaml` therefore keeps `clear_vegetation: true` and only skips the fill on grass tasks (same rule as evaluation).

When entities are not cleared, a natural pig can stand closer than the summoned offset. The logged `initial_target_distance` is then the nearest aligned or wide ray on that species. `spawn_target_distance` is the offset, and `natural_target_closer` is true when the ray is clearly shorter.

## Run

Login-node check:

```bash
python3 -m unittest discover -s lfnav/tests
```

Array job (16 tasks × 3 difficulties; tree/grass medium/hard exit immediately):

```bash
source /scratch/$USER/mp5_env.sh
cd $MP5 && git pull
mkdir -p $SCR/lfnav/logs
sbatch -A "$SBATCH_ACCOUNT" -p n2c48m24 --export=ALL lfnav/slurm/collect_demos_array.sbatch
```

Overrides: `CFG`, `SEEDS`, `OUT` (defaults: `configs/lfnav/main.yaml`, `configs/lfnav/train_seeds_demo.txt`, `$SCR/lfnav/data/demos_main`). Failures keep `record.json` and `steps.jsonl` only. Legacy dynamic demos stay under `demos_dynamic`.

Seeds are 1–48 except those ending in 9 (44 seeds). Finished `record.json` files are skipped.

Summarize:

```bash
python3 -m lfnav.data.demo_stats --in $SCR/lfnav/data/demos_dynamic --out results/demos/stats_dynamic.md
```

## Layout

`<OUT>/{success|failure}/<task>/<difficulty>/<seed>/`

| File | Contents |
| --- | --- |
| `record.json` | Same fields as an eval episode, plus `config_name` |
| `steps.jsonl` | One decision per line: `decision`, `env_step`, `match`, `scene` (GT), 8-sector visit counts, `recent_actions`, expert `action_chunk`, `used_wide` |
| `frames/NNNNN.jpg` | RGB at that decision, resized to 224×364, JPEG quality 90 |

`used_wide` is true when the expert steered from the wide lidar because aligned rays did not yet see the target.
