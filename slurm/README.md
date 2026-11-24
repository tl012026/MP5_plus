# Slurm templates

Do not submit from a login node until the image, overlay, and project account are ready.

## Variables

| Name | Meaning |
|---|---|
| `MP5_ROOT` | This repository |
| `SIF` | Apptainer/Singularity image (placeholder until `env/` is built) |
| `OVERLAY` | Writable overlay for Minecraft/MineDojo writes |
| `CONFIG` | `configs/code_default.yaml` or `configs/strict.yaml` |
| `TASKS` | Task JSON or a directory of JSON files |
| `SEEDS` | Seed file (`configs/seeds_example.txt`) |
| `OUT` | `/scratch/$USER/mp5_runs/<jobid>` |
| `KEYFILE` | `chmod 600` file that only `export`s `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`. Never commit it. |
| `WORKERS` | `run_batch.py --workers` |
| `MINELLM_ENV` / `MINELLM_LOG_DIR` / `MINELLM_READY_TIMEOUT` | MineLLM server settings for `minellm_server.sh` (port 25541 and weight paths are fixed by the server script). |

The account is not in the scripts; pass it with `-A` when submitting. `eval_cpu.sbatch` leaves the partition blank. `eval_gpu_minellm.sbatch` sets the A100 partition `c12m85-a100-1`. Both scripts use `--requeue`, `--skip-done`, `singularity exec` + overlay, and `xvfb-run -a`.

## Example

```bash
export SIF=/scratch/$USER/mp5.sif
export OVERLAY=/scratch/$USER/mp5.overlay
export KEYFILE=$HOME/.mp5_openai.env
sbatch slurm/eval_cpu.sbatch
env -u SBATCH_ACCOUNT -u SBATCH_PARTITION sbatch -A <account> slurm/eval_gpu_minellm.sbatch
```

Requeue-safe: finished `OUT/<task>/<seed>.json` files are skipped.

## MineLLM

`minellm_server.sh` is sourced by GPU jobs. It provides `minellm_start`, `minellm_wait_ready` (TCP check on 127.0.0.1:25541, 30 min limit) and `minellm_stop` (SIGTERM, then SIGKILL after 30 s). `eval_gpu_minellm.sbatch` uses it, runs the server on the host with `minellm_env`, and runs the agent in the MineDojo image. It takes `OPENAI_API_KEY` from the submitting environment, or from `KEYFILE` if set. `mp5_env.sh` exports `SBATCH_PARTITION`, which overrides `#SBATCH --partition`, so submit with `env -u SBATCH_ACCOUNT -u SBATCH_PARTITION`. Interface, timing and accuracy notes are in `docs/minellm_service.md`.

## Integration test (MineLLM + agent)

`integration_episode.sh` runs one episode end to end: it checks `OPENAI_API_KEY`, starts MineLLM and waits for the port, waits for host memory to settle, runs `run_agent.py` in the MineDojo image with `configs/openrouter.yaml` (a copy with `eval.max_steps` set to `MAX_STEPS`: 12000 in the A100 job, 2000 in the rehearsal), stops MineLLM, and writes `/scratch/wz2999/integration_test/summary_<jobid>.md`. Two wrappers call it:

- `integration_rehearsal_cpu.sbatch`: CPU partition, `tools/fake_minellm_server.py` instead of MineLLM. Run this first.
- `integration_wooden_pickaxe.sbatch`: A100, real MineLLM.

```bash
env -u SBATCH_ACCOUNT -u SBATCH_PARTITION sbatch -A <account> slurm/integration_rehearsal_cpu.sbatch
env -u SBATCH_ACCOUNT -u SBATCH_PARTITION sbatch -A <account> slurm/integration_wooden_pickaxe.sbatch
```

Submit from a host shell where `OPENAI_API_KEY` is exported. The default `--export=ALL` passes it to the job. The scripts never print it.
