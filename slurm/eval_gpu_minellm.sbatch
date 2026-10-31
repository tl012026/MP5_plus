#!/bin/bash
#SBATCH --job-name=mp5-gpu
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/mp5_runs/logs/%x-%j.out
#SBATCH --error=/scratch/%u/mp5_runs/logs/%x-%j.err

# GPU eval: start MineLLM in the background, then run_batch, then stop the server.
# Extra variables:
#   MINELLM_SCRIPT  LAMM/src/mllm_api_mc_v1.5.py (path inside the image or on the host)
#   MINELLM_PORT    default 25541
#   MINELLM_CKPT    MineLLM checkpoint (placeholder)
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
MINELLM_SCRIPT=${MINELLM_SCRIPT:-$MP5_ROOT/LAMM/src/mllm_api_mc_v1.5.py}
MINELLM_PORT=${MINELLM_PORT:-25541}
MINELLM_CKPT=${MINELLM_CKPT:-/path/to/minellm.ckpt}

mkdir -p /scratch/$USER/mp5_runs/logs "$OUT"
# shellcheck disable=SC1090
source "$KEYFILE"

MLLM_URL="http://127.0.0.1:${MINELLM_PORT}/image"
cleanup() {
  if [[ -n "${MLLM_PID:-}" ]]; then
    kill "$MLLM_PID" 2>/dev/null || true
    wait "$MLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

singularity exec --nv --overlay "$OVERLAY" "$SIF" \
  python3 "$MINELLM_SCRIPT" --port "$MINELLM_PORT" --ckpt "$MINELLM_CKPT" \
  >"$OUT/minellm.log" 2>&1 &
MLLM_PID=$!

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${MINELLM_PORT}/" >/dev/null 2>&1 \
     || curl -sf "$MLLM_URL" >/dev/null 2>&1; then
    break
  fi
  sleep 5
  if ! kill -0 "$MLLM_PID" 2>/dev/null; then
    echo "MineLLM exited early; see $OUT/minellm.log" >&2
    exit 1
  fi
done

singularity exec --nv --overlay "$OVERLAY" "$SIF" \
  xvfb-run -a python3 "$MP5_ROOT/eval/run_batch.py" \
    --config "$CONFIG" \
    --tasks "$TASKS" \
    --seeds "$SEEDS" \
    --out "$OUT" \
    --workers "$WORKERS" \
    --mllm-url "$MLLM_URL" \
    --skip-done
