# Slurm order

Account and partition are left blank in the scripts. Fill them in locally before submitting. Every script sets `--requeue`. Logs go to `/scratch/$USER/lfnav/logs/`. MineDojo jobs use `$SCR/env/minedojo.sif` and source `$SCR/mp5_env.sh` with absolute paths, because the container user is `user`.

Do not submit these from the login node until the checkpoint below is done.

1. `sbatch lfnav/slurm/probe_obs.sbatch`
   Read `/scratch/$USER/lfnav/probe/obs.json`. Fill `configs/lfnav/obs_keys.yaml`. Set `actions.confirmed` only after the vectors match `action_space`.
2. Measure one block and `PIXEL=.. SIZE=.. ANGLE=.. AXIS=horizontal sbatch lfnav/slurm/calibrate_camera.sbatch`.
   Copy `fov_deg` and `fov_axis` into `configs/lfnav/default.yaml`.
3. `FRAMES=1000 BACKEND=minedojo sbatch lfnav/slurm/collect_perception.sbatch`
   Draw a few frames with `lfnav/data/vis_labels.py`. If the tint misses the object, stop and refit the camera.
4. `sbatch lfnav/slurm/extract_features.sbatch` on a GPU partition, then `sbatch lfnav/slurm/train_perception.sbatch`.
   Source `~/lfnav_train_env.sh` so the job uses a PyTorch 2.x env, not the MineDojo image.
5. `sbatch lfnav/slurm/collect_demos.sbatch` with `configs/lfnav/train_seeds.txt`.
6. `sbatch lfnav/slurm/train_c2.sbatch`. Add `DAGGER=1` only for a second round.
7. `sbatch lfnav/slurm/eval.sbatch` with `configs/lfnav/eval_seeds.txt`.
   `sbatch lfnav/slurm/eval_gpu.sbatch` records CPU latency of the fake encoder and writes the SVG. Point it at a real checkpoint on the GPU node when one exists.
8. `MINELLM_START='...' sbatch lfnav/slurm/minellm_baseline.sbatch` on a GPU partition.
   The command must serve `POST /image` on port 25541. The job kills that process on exit.

`configs/lfnav/matrix.yaml` lists the comparisons. The original MP5 agent stays on `eval/run_batch.py` and is not started here.
