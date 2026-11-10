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
| easy | 4–8, any yaw | 0 | 0 |
| medium | 8–16, any yaw | 8 | 2 |
| hard | 16–32, behind the agent ±45° | 16 | 4 |

Animals are summoned by lfnav *after* `reset()`, not by MineDojo `initial_mobs`. HarvestMeta's reset-hook `/summon` does not spawn. Main eval (`eval_dynamic.yaml`) does not `/kill` natural mobs and does not freeze them; `initial_target_distance` and SPL use the nearest same-species opening ray (`apply_spawn_offset_distance`). Static robustness (`baseline_eval.yaml`) still clears, freezes, and uses `hypot(dx, dz)` from the summon offset. Ray range is always logged as `initial_target_distance_rays`. Ray-check retries stay behind `spawn_verify_retry` (off). Episodes with no GT lidar hit on the target are `target_never_seen` and still count in SR. Only a spawn exception is `spawn_failed` and is dropped from SR. `natural_mobs_cleared` marks the `/kill` path.

Block targets (tree, grass, water) and block `near` conditions still reroll the world seed (`seed*1000+k`, up to `max_reroll`) until `min_initial_distance` holds. Otherwise the episode is `difficulty_unmet`. `specified_biome` worlds are treated as single-biome, so `leave_biome` is false. Optional mixed-biome seeds: `lfnav/env/search_biome_seeds.py`.

### Camera

- Aligned rays use one ray per DINOv2 patch (224×364 → 16×26).
- `lfnav/gt/capture_rotations.py` turns in place. `fit_intrinsics.py` fits `K` from the pure-rotation homography `x2 ~ K R K^{-1} x1`, with the HUD masked and RANSAC dropping moving pixels.
- Result: vertical FOV ≈ 70° (fx 368.0, fy 368.6). `default.yaml` keeps `fov_deg: 70` (vertical).
- MineDojo lidar yaw and pitch are negated at a single conversion point to match the camera convention.
- `verify_alignment.py` accepts a median IoU above 0.5.

## Run order

Login-node tests need neither MineDojo nor a GPU:

```bash
python3 -m unittest discover -s lfnav/tests
python3 lfnav/perception/train_perception.py --dry-run
python3 lfnav/policy/train_c2.py --dry-run
```

Real episodes follow `lfnav/slurm/README.md`. Data, features, checkpoints and runs default to `/scratch/$USER/lfnav/` (`configs/lfnav/default.yaml`).

### Frames

- Eval saves one JPG every `frames.every` env steps under `OUT/<task>/<seed>/frames/`. The default is 20; the baseline config uses 10.
- Start, stop and end also keep a full-size PNG.
- Override from the command line with `--frames-every N`, or turn frames off with `--no-frames`.
- `python3 lfnav/eval/make_contact_sheet.py --in OUT` stitches each episode into one sheet.

### Environment reuse

- Each episode closes its environment, including when the episode raises. `reuse_env` defaults to false.
- With `reuse_env` on, consecutive episodes of the same task and difficulty share one MineDojo process and start with `reset`.
- In that mode, `make()` uses the first episode's seed as `world_seed`. Later seeds are kept on the result file and in agent-side randomness, but they do not build a new world.

## Baseline: C0 / C1 × GT perception

Purpose: get an upper bound (C1 + GT) and a lower bound (C0 + GT, close to MP5's blind exploration) before any learned module exists. Both arms use perfect perception, so any difference comes from the navigation policy alone. Smoke-run problems and the fixes that followed are in [`C0C1_BASELINE.md`](C0C1_BASELINE.md).

| | |
| --- | --- |
| Tasks | 16 specs in `lfnav/tasks/specs/` |
| Difficulties | easy / medium / hard |
| Seeds | `configs/lfnav/eval_seeds_baseline.txt` (9, 19, 29, 39, 49; no overlap with training) |
| Episodes | 16 × 3 × 5 = 240 per policy, 480 in total |
| Step limit | 3000 env steps (`judge.max_steps`) |
| Config | `configs/lfnav/eval_dynamic.yaml` (static robustness: `configs/lfnav/baseline_eval.yaml`) |
| Job | `lfnav/slurm/baseline_array.sbatch` → `$SCR/lfnav/runs/baseline_dynamic`. Array index `i` maps to task `i // 3`, difficulty `i % 3` (`lfnav/eval/baseline_index.py`). Each index runs 5 seeds |
| Compute | CPU only (no GPU, no API) |

### Before submitting

```bash
source /scratch/$USER/mp5_env.sh            # sets SCR, MP5, SBATCH_ACCOUNT; unsets OOD Slurm vars
cd $MP5 && git pull && git log -1 --oneline
mkdir -p $SCR/lfnav/logs                    # Slurm will not create the log dir
du -sh $SCR/.gradle                         # every job copies this onto node /tmp
```

- The sbatch files leave `--account` and `--partition` blank. Pass `-A` and `-p` on the command line.
- Pass `--export=ALL` so that `MP5`, `SCR`, `SEEDS`, `ROOT_OUT` and `POLICY` reach the job.

### Step 0: is a `specified_biome` world single-biome? (≈10 min)

```bash
mkdir -p $SCR/lfnav/biome_check
sbatch -A "$SBATCH_ACCOUNT" -p n2c48m24 -c 4 --mem=16G -t 01:00:00 --export=ALL \
  -o $SCR/lfnav/logs/biome-%j.out --wrap '
  source "$MP5/lfnav/env/gradle_home.sh"
  for b in forest desert jungle taiga plains; do
    lfnav_sif_exec "$SCR/env/minedojo.sif" xvfb-run -a \
      python -m lfnav.env.check_biome_world --biome $b --seed 9 --out "$SCR/lfnav/biome_check/$b.json"
  done'
grep -h single_biome $SCR/lfnav/biome_check/*.json
```

If any biome comes back `false`, revisit the biome conditions and `leave_biome` before running the full baseline.

### Step 1: smoke run (2 episodes per policy)

Array index 2 is task 1-1 on hard (block target, reroll path). Index 9 is task 1-4 on easy (mob target).

```bash
echo 9 > $SCR/lfnav/smoke_seed.txt
for P in c1 c0; do
  POLICY=$P SEEDS=$SCR/lfnav/smoke_seed.txt ROOT_OUT=$SCR/lfnav/runs/baseline_smoke \
  sbatch -A "$SBATCH_ACCOUNT" -p n2c48m24 --export=ALL --array=2,9 lfnav/slurm/baseline_array.sbatch
done
squeue -u $USER
```

Check the results:

```bash
python3 - <<'EOF'
import json, glob, os
root = os.path.expandvars("$SCR/lfnav/runs/baseline_smoke")
for f in sorted(glob.glob(f"{root}/*/*/*/*/*.json")):
    if f.endswith("run_meta.json"): continue
    r = json.load(open(f)); d = f[:-5] + "/frames"
    n = len(os.listdir(d)) if os.path.isdir(d) else 0
    mb = sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d)) / 1e6 if n else 0
    print(f.replace(root, ""), {k: r.get(k) for k in ("sr_b","sr_a","steps","initial_target_distance","difficulty_unmet","elapsed_s")}, f"frames={n} {mb:.1f}MB")
EOF
```

Pass criteria:

- frames ≈ steps / 10 + 3 keyframes, at well under 1 MB per episode;
- `initial_target_distance` ≥ 16 on hard, and `difficulty_unmet` is false;
- steps/s (steps ÷ `elapsed_s`) is within 10% of a `--no-frames` run;
- the log shows `GRADLE_USER_HOME` under `/tmp/gradle_<jobid>`, not `$SCR/.gradle`.

### Step 2: full run

```bash
for P in c1 c0; do
  POLICY=$P sbatch -A "$SBATCH_ACCOUNT" -p n2c48m24 --export=ALL --array=0-47%12 lfnav/slurm/baseline_array.sbatch
done
```

- `%12` caps concurrent jobs, which limits Gradle copies on node `/tmp`.
- Each index takes about 5 × (1 min build + up to 2.5 min of steps), well under the 12 h limit.
- `--requeue` with `--skip-done` resumes jobs cancelled by the idle sweep (`CANCELLED by 0`).
- Requeue failed indices with `--array=<list>`.

### Step 3: aggregate

```bash
python3 -m lfnav.aggregate --baseline $SCR/lfnav/runs/baseline_dynamic --out $SCR/lfnav/runs/baseline_dynamic/summary.json
python3 lfnav/eval/make_contact_sheet.py --in $SCR/lfnav/runs/baseline_dynamic/c1/hard/1-1   # spot-check any cell
```

`summary.json` holds:

- one row per (task, difficulty, policy): SR-b, SR-a, false-stop rate, timeout rate, median steps, SPL, initial distance and `difficulty_unmet` rate;
- a paired McNemar test of C1 vs C0.

### Outputs

```
$SCR/lfnav/runs/baseline_dynamic/<policy>/<difficulty>/<task>/
  run_meta.json
  <task>/<seed>.json              per-episode result
  <task>/<seed>/events.jsonl      per-decision log
  <task>/<seed>/frames/           JPG every 10 steps + start/stop/end PNG
```

Frames take about 1–2 GB per policy. They stay on `/scratch`.

The only baseline file committed to the repo is `summary.json`, under `results/baseline/`, together with the commit hash it came from.

### Expected

- C1 has clearly higher SR-b and SPL and fewer steps than C0, and the gap grows from easy to hard.
- Easy block targets are near 100% for both policies.
- False stops are near 0, because perception is ground truth.
- Timeout is the main failure: far targets, terrain occlusion, mobs wandering off.
- C1 can loop on hard when the target is visible but a condition is missing.
- Things to watch:
  - conditions that may never trigger (night, rain, near water);
  - small mobs leaving the frame at close range.
- Remaining limits and a partial score table after the fixes: [`C0C1_BASELINE.md`](docs/C0C1_BASELINE.md).

## Matrix

`configs/lfnav/matrix.yaml` names the primary pairs: MineLLM+C0, DINOv2+C0, DINOv2+C1, DINOv2+C2, GT+C1 and GT+C2. Perception options also include ViT-B/14 and MineCLIP. MineCLIP is not connected until a local checkout exists under the configured directory.

The original MP5 loop is not executed here. `lfnav.eval.run.make_policy` rejects `mp5` and points to `eval/run_batch.py` for a later API run with the same specs, seeds and judge.

## Limits

- Camera FOV, time / brightness thresholds (`time_day_min` 0.42, `brightness_min_light` 13) and biome ids are calibrated on real MineDojo. See `lfnav/REPORT.md`.
- Action vectors in `configs/lfnav/actions.yaml` are confirmed against the harvest_milk probe.
- Tree labels match the name fragments `log`, `leaves` and `wood`.
- A learned head without a checkpoint reports a token statistic, not a trained probability.
- `near X` is a same-frame approximation, and SPL ignores detours (see above).