# lfnav implementation report

## Stage 1

Status: done on the login node.

Added dataclasses, the 16 creative-task specs, the rule matcher, the SR-a / SR-b judge, patch-ray geometry, tile aggregation, the fake grid environment, and threshold sweep. `python3 -m unittest lfnav.tests.test_stage1` passes without MineDojo or a GPU.

No existing MP5 file was modified in that commit. Observation keys that the real client must confirm stay null in `configs/lfnav/obs_keys.yaml`. Action vectors are a local contract marked `confirmed: false`.

A follow-up commit tracks `lfnav/env`: the root `.gitignore` rule `env/` also matched that package. Bytecode that the exception re-included was removed in a third commit.

## Stage 2

Status: done on the login node.

The controller executes the named actions, jumps a one-block solid, refuses water, and turns waypoints into yaw then forward. The 3x3x3 voxel crop stays inside the controller. C0 walks and turns when blocked. C1 faces a visible pig in the fake world and stops inside the success radius. The eval shell writes one JSON, an events JSONL, `run_meta.json`, and `summary.csv`, and `--skip-done` keeps a finished episode. `python3 -m unittest lfnav.tests.test_stage2` passes.

## Stage 3

Status: scripts are ready. They have not been run inside MineDojo.

`probe_obs.py` writes the real observation keys. `calibrate_camera.py` turns a measured pixel and angle into a field of view (`fov.py` is tested). `collect_perception.py` writes resumable shards; the pilot size is 1000 frames in `configs/lfnav/collect_pilot.yaml`. `vis_labels.py` paints patch hits. Slurm files: `lfnav/slurm/probe_obs.sbatch`, `calibrate_camera.sbatch`, `collect_perception.sbatch`. `python3 -m unittest lfnav.tests.test_stage3` passes on the login node.

Checkpoint for a real client, in order:

1. `sbatch lfnav/slurm/probe_obs.sbatch` then copy keys into `configs/lfnav/obs_keys.yaml` and set `actions.confirmed` only after the action vectors match.
2. Measure one block's pixel and `PIXEL=... SIZE=... ANGLE=... AXIS=horizontal sbatch lfnav/slurm/calibrate_camera.sbatch`. Put `fov_deg` and `fov_axis` into `configs/lfnav/default.yaml`.
3. Run a fake or short MineDojo collection and open the PPM from `vis_labels.py`. The patch tint should sit on the object.
4. Only then raise `FRAMES` from 1000 toward 30000–50000.

## Stage 4

Status: done on the login node for the fake encoder. DINOv2 weights are not downloaded here.

`LearnedPerception` and `extract_features.py` cache fp16 features. `train_perception.py --dry-run` runs one batch; this machine has no torch, so the numpy BCE path ran (`dry-run numpy loss`). MineLLM is an HTTP client with fixed yes/no and biome templates; tests cover the parser and do not open a port. MineCLIP is not loaded: there is no local checkout wired up, and `load_encoder("mineclip")` raises with that reason. Offline metrics cover F1, hallucination, miss rate, direction, distance, and ECE.

## Stage 5

Status: done on the login node.

The expert is C1 plus ground truth and the wide lidar, and `collect_demos.py` keeps seeds ending in 9 out of the training pool. A fake episode of task 1-4 is saved under `success/`. C2 returns a chunk of 4 actions from a small linear model. `--dry-run` lowers the cross-entropy. DAgger is the `--dagger` switch and is not launched here.

## Stage 6

Status: done on the login node. No Slurm job was submitted.

`configs/lfnav/matrix.yaml` lists the primary comparisons. `lfnav/eval/metrics.py` reports Wilson intervals and McNemar. `measure_cost.py` times the fake encoder and C2 on CPU and writes an SVG. `lfnav/README.md` and `lfnav/slurm/README.md` describe the run order. The root README link is a separate commit.

## Transcription

The 16 files follow the creative-task sentences, not the pig/forest example in the task statement.

- 1-1 Find a tree → tree, no conditions, episode forest.
- 1-2 Find a grass → grass, plains.
- 1-3 Find a cow → cow, plains.
- 1-4 Find a pig → pig, plains.
- 2-1 tree in the forest → tree, biome forest.
- 2-2 grass near the pig → grass, near pig. The target is the grass.
- 2-3 cow in the desert → cow, biome desert.
- 2-4 pig at night → pig, time night.
- 3-1 tree in the forest at night → tree, forest, night.
- 3-2 grass near the pig in the plains → grass, near pig, plains.
- 3-3 cow in the desert during the day → cow, desert, day.
- 3-4 pig at night in rain → pig, night, rain.
- 4-1 tree in the forest at night, sunny → tree, forest, night, clear.
- 4-2 grass near the pig in the plains during the day → grass, near pig, plains, day. This is not the pig-in-forest example.
- 4-3 cow near water in the desert during the day, sunny → cow, near water, desert, day, clear.
- 4-4 pig during the day on the plains with grass, sunny, brightness sufficient → pig, day, plains, near grass, clear, sufficient.

Sentences that omit the biome use forest for a tree and plains otherwise. That is a scene choice, written in `lfnav/tasks/README.md`.

## Existing files

`MP5_agent/agent/` was not modified. `.gitignore` gained an exception so `lfnav/env/` is not hidden by the conda `env/` rule, plus a rule that keeps that package's bytecode out. The root README link is its own commit.

## What the user should run

1. `sbatch lfnav/slurm/probe_obs.sbatch`
2. Fill `configs/lfnav/obs_keys.yaml` from `/scratch/$USER/lfnav/probe/obs.json`. Confirm action vectors before setting `actions.confirmed: true`.
3. `PIXEL=... SIZE=... ANGLE=... AXIS=horizontal sbatch lfnav/slurm/calibrate_camera.sbatch` and copy the fit into `configs/lfnav/default.yaml`.
4. `FRAMES=1000 sbatch lfnav/slurm/collect_perception.sbatch`, then `python lfnav/data/vis_labels.py --shard <shard> --out /tmp/frame.ppm`.
5. After the tint matches the object, raise `FRAMES` and run extract, train, demos, C2, and eval as in `lfnav/slurm/README.md`.
6. `MINELLM_START='<server command>' sbatch lfnav/slurm/minellm_baseline.sbatch` when the server command is known.

## Risks

- Patch rays and the image can disagree until the field of view is fit. The calibration script and the label picture are the check. Do not train on a pilot whose tint misses the object.
- Mobs move, so a ray label from one frame is noisy. The pilot should be inspected, and the matcher's optional vote (`smooth_n`) can be turned on if stops flicker.
- The expert only looks perfect in the fake grid. On a real map it can miss a target that the wide lidar does not reach. Keep failed demos, and do not train C2 until the success bucket is large enough to inspect.
- Action indices and biome ids are still unconfirmed. The code refuses MineDojo named actions until `actions.confirmed` is true, and it will not invent a biome id map.
