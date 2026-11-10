# Expert demonstrations for C2

The expert is C1 plus ground-truth perception plus the 360° wide lidar (`lfnav/policy/expert.py`). It is not used at evaluation time. Dynamic-scene demos keep natural animals moving; they are the C2 training set.

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

Overrides: `CFG`, `SEEDS`, `OUT` (defaults: `configs/lfnav/demo_dynamic.yaml`, `configs/lfnav/train_seeds_demo.txt`, `$SCR/lfnav/data/demos_dynamic`). Frozen-mob demos use `CFG=$MP5/configs/lfnav/demo_collect.yaml`.

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

`used_wide` is true when the expert steered from the wide lidar because the matcher had not yet found the target.
