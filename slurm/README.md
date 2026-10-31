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
| `MINELLM_SCRIPT` | `LAMM/src/mllm_api_mc_v1.5.py` (flags in the GPU script are placeholders; match the file on the compute node) |
| `MINELLM_PORT` / `MINELLM_CKPT` | Local MineLLM bind port and weights |

Both scripts use `--account=<SLURM_ACCOUNT>`, `--requeue`, `--skip-done`, `singularity exec` + overlay, and `xvfb-run -a`.

## Example

```bash
export SIF=/scratch/$USER/mp5.sif
export OVERLAY=/scratch/$USER/mp5.overlay
export KEYFILE=$HOME/.mp5_openai.env
sbatch slurm/eval_cpu.sbatch
sbatch slurm/eval_gpu_minellm.sbatch
```

Requeue-safe: finished `OUT/<task>/<seed>.json` files are skipped.
