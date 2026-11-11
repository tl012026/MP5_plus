# Slurm order

Account and partition are left blank in the scripts. Fill them in locally before submitting. Every script sets `--requeue`. Logs go to `/scratch/$USER/lfnav/logs/`. MineDojo jobs use `$SCR/env/minedojo.sif` and source `$SCR/mp5_env.sh` with absolute paths, because the container user is `user`.

On this cluster, GPU jobs must request `#SBATCH --gres=gpu:1`. `#SBATCH --gpus=1` is rejected.

MineDojo sbatch templates source `lfnav/env/gradle_home.sh`. That copies `$SCR/.gradle` onto `$SLURM_TMPDIR/gradle_$SLURM_JOB_ID` (or `/tmp/...`), skipping `daemon/` and `*.lock`, exports `GRADLE_USER_HOME` to the copy, and deletes the copy on EXIT. Jobs must not write the master. To build or refresh the master, run one job with `GRADLE_USE_MASTER=1` so it uses `$SCR/.gradle` directly. Several workers in one job share the copy and serialize `minedojo.make` (default). `GRADLE_WORKER_COPIES=1` gives each worker `gradle_<job>_w<k>` instead. `init.d/mixingradle-local.gradle` keeps its absolute `$SCR/env/m2` path.

Create `/scratch/$USER/lfnav/logs` on the login node before the first `sbatch`. Slurm opens `--output` before the job script can `mkdir`.

Do not submit these from the login node until the checkpoint below is done.

1. `sbatch lfnav/slurm/probe_obs.sbatch`
   Read `/scratch/$USER/lfnav/probe/obs.json`. Fill `configs/lfnav/obs_keys.yaml`. Set `actions.confirmed` only after the vectors match `action_space`.
2. `sbatch lfnav/slurm/calibrate_camera.sbatch` turns in place, fits a pinhole K, then overlays pig rays.
   Read `/scratch/$USER/lfnav/calib/intrinsics.json`. If `suggest.mode` is `fov`, copy `fov_deg` and `fov_axis` into `configs/lfnav/default.yaml`. If it is `intrinsics`, copy the `fx,fy,cx,cy,width,height` block instead.
   The old single-pixel script is `PIXEL=.. SIZE=.. ANGLE=.. AXIS=horizontal sbatch lfnav/slurm/calibrate_camera_manual.sbatch`.
   Then `sbatch lfnav/slurm/calibrate_env_state.sbatch`. Copy `time_day_min` and
   `brightness_min_light` into `default.yaml` and `biome_id_to_name` into `obs_keys.yaml`.
   Until those are filled, time and brightness conditions are unknown, not silent misses.
3. `sbatch lfnav/slurm/collect_demos.sbatch` with `configs/lfnav/train_seeds.txt`. Main-scene demos: `sbatch lfnav/slurm/collect_demos_array.sbatch` (`configs/lfnav/main.yaml` → `$SCR/lfnav/data/demos_main`; see [`../docs/DEMOS.md`](../docs/DEMOS.md)).
4. `sbatch lfnav/slurm/extract_features.sbatch` on a GPU partition (sources `~/lfnav_train_env.sh`). Frozen DINOv2 on filtered demo frames; writes `$SCR/lfnav/features/joint_v1`.
5. `sbatch lfnav/slurm/train_joint.sbatch` on a GPU partition. Joint A+C2; `OUT=$SCR/lfnav/ckpt/joint_v1`.
6. `sbatch lfnav/slurm/eval.sbatch` with `configs/lfnav/eval_seeds.txt`. For the learned stack: `POLICY=joint`.
   Frames are on by default (every 20 env steps). `FRAMES_EVERY=0` is not used; pass `--no-frames` or `--frames-every N` on the python command. After a run: `python3 lfnav/eval/make_contact_sheet.py --in $OUT`.
   Baseline C0/C1 × GT: `POLICY=c1 sbatch lfnav/slurm/baseline_array.sbatch` then `POLICY=c0 sbatch ...` (array 0–47 = 16 tasks × 3 difficulties, 5 seeds ending in 9). Default config is `configs/lfnav/main.yaml`, output `$SCR/lfnav/runs/main`. Summarize with `python3 -m lfnav.aggregate --baseline /scratch/$USER/lfnav/runs/main`.
   Command probe: `sbatch lfnav/slurm/probe_commands.sbatch`.
   `sbatch lfnav/slurm/eval_gpu.sbatch` records CPU latency of the fake encoder and writes the SVG. Point it at a real checkpoint on the GPU node when one exists.
7. `MINELLM_START='...' sbatch lfnav/slurm/minellm_baseline.sbatch` on a GPU partition.
   The command must serve `POST /image` on port 25541. The job kills that process on exit.

`configs/lfnav/matrix.yaml` lists the comparisons. The original MP5 agent stays on `eval/run_batch.py` and is not started here.
