#!/bin/bash
#SBATCH --job-name=mp5-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/mp5_runs/logs/%x-%j.out
#SBATCH --error=/scratch/%u/mp5_runs/logs/%x-%j.err

# CPU eval (no MineLLM). Image-only / gpt-vision or a remote mllm_url.
# Variables (edit or export before sbatch):
#   MP5_ROOT     repo root
#   SIF          singularity/apptainer image
#   OVERLAY      writable overlay
#   CONFIG       YAML under configs/
#   TASKS        task json or directory
#   SEEDS        seeds file
#   OUT          run directory
#   KEYFILE      mode 600 file that exports OPENAI_API_KEY (and optional OPENAI_BASE_URL)
#   WORKERS      process count
set -euo pipefail

MP5_ROOT=${MP5_ROOT:-$HOME/MP5_plus}
SIF=${SIF:-/path/to/mp5.sif}
OVERLAY=${OVERLAY:-/path/to/mp5.overlay}
CONFIG=${CONFIG:-$MP5_ROOT/configs/code_default.yaml}
TASKS=${TASKS:-$MP5_ROOT/MP5_agent/agent/tasks/process}
SEEDS=${SEEDS:-$MP5_ROOT/configs/seeds_example.txt}
OUT=${OUT:-/scratch/$USER/mp5_runs/${SLURM_JOB_ID}}
KEYFILE=${KEYFILE:-$HOME/.mp5_openai.env}
WORKERS=${WORKERS:-1}

mkdir -p /scratch/$USER/mp5_runs/logs "$OUT"
# shellcheck disable=SC1090
source "$KEYFILE"

singularity exec --overlay "$OVERLAY" "$SIF" \
  xvfb-run -a python3 "$MP5_ROOT/eval/run_batch.py" \
    --config "$CONFIG" \
    --tasks "$TASKS" \
    --seeds "$SEEDS" \
    --out "$OUT" \
    --workers "$WORKERS" \
    --skip-done
