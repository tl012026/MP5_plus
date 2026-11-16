# Slurm order

Account and partition are left blank in the scripts. Fill them in locally before submitting. Every script sets `--requeue`. Logs go to `$SCR/lfnav/logs/`. MineDojo jobs use `$SCR/env/minedojo.sif` and source `$SCR/mp5_env.sh` with absolute paths, because the container user is `user`.

GPU jobs must request `#SBATCH --gres=gpu:1`. `#SBATCH --gpus=1` is rejected.

MineDojo sbatch templates source `lfnav/env/gradle_home.sh`. That copies `$SCR/.gradle` onto `$SLURM_TMPDIR/gradle_$SLURM_JOB_ID` (or `/tmp/...`), skipping `daemon/` and `*.lock`, exports `GRADLE_USER_HOME` to the copy, and deletes the copy on EXIT. Jobs must not write the master. To build or refresh the master, run one job with `GRADLE_USE_MASTER=1` so it uses `$SCR/.gradle` directly. Several workers in one job share the copy and serialize `minedojo.make` (default). `GRADLE_WORKER_COPIES=1` gives each worker `gradle_<job>_w<k>` instead. `init.d/mixingradle-local.gradle` keeps its absolute `$SCR/env/m2` path.

Create `$SCR/lfnav/logs` before the first `sbatch`. Slurm opens `--output` before the job script can `mkdir`.

Before every submit: `export MP5=<run copy>`, confirm `git -C $MP5 log -n 1`, and confirm that every config file the job reads exists. The scripts default to `MP5=${MP5:-$HOME/MP5_plus}`.

## Current workflow

1. **Collection.** `lfnav/slurm/collect_demos_array.sbatch` with `CFG`, `SEEDS` and `OUT`.
   Defaults: `CFG=$MP5/configs/lfnav/main.yaml`, `SEEDS=$MP5/configs/lfnav/train_seeds_demo.txt`, `OUT=$SCR/lfnav/data/demos_main_v3`.
   Array index maps to one task and difficulty through `lfnav/eval/baseline_index.py`. Tree and grass tasks skip medium and hard. Task 4-3 is skipped. See [`../docs/DEMOS.md`](../docs/DEMOS.md).

2. **Evaluation.** `lfnav/slurm/baseline_array.sbatch` with `POLICY=c1` or `POLICY=c2nav`, plus `CFG` and `ROOT_OUT`.
   Array index = 3 × task index + difficulty index, as computed by `lfnav/eval/baseline_index.py`.
   Defaults: `CFG=$MP5/configs/lfnav/main.yaml`, `SEEDS=$MP5/configs/lfnav/eval_seeds_baseline.txt`, `ROOT_OUT=$SCR/lfnav/runs/main`. Perception default is `gt`.

3. **Training.** `lfnav/slurm/train_c2nav.sbatch` with `CFG`, `FEAT` and `OUT`.
   Defaults: `CFG=$MP5/configs/lfnav/c2nav_easy.yaml`, `FEAT=$SCR/lfnav/features/nav_v3_easy`, `OUT=$SCR/lfnav/ckpt/c2nav_easy_r3`. This job reads cached features. It does not start MineDojo.

## A or legacy pipeline only

Feature extraction (`lfnav/slurm/extract_features.sbatch`) and joint training (`lfnav/slurm/train_joint.sbatch`) are for learned perception A or the superseded joint policy. They are not part of the main C2 line.

`configs/lfnav/matrix.yaml` lists the comparisons. The original MP5 agent stays on `eval/run_batch.py` and is not started here.
