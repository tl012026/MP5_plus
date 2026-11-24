#!/bin/bash
#SBATCH --job-name=mp5-gpu
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/mp5_runs/logs/%x-%j.out
#SBATCH --error=/scratch/%u/mp5_runs/logs/%x-%j.err

# GPU eval: start MineLLM on the host, wait for its port, run eval/run_batch.py in the
# MineDojo container against it, then stop the server. Everything stays in one job,
# because the cluster cancels jobs whose GPU/CPU use stays low for ~20 minutes.
#
# Submit (account on the command line; unset the SBATCH_* vars that mp5_env.sh exports,
# because they override the #SBATCH lines above):
#   env -u SBATCH_ACCOUNT -u SBATCH_PARTITION sbatch -A <account> slurm/eval_gpu_minellm.sbatch
# The output directory /scratch/$USER/mp5_runs/logs must exist before submitting.
#
# OPENAI_API_KEY (OpenRouter or OpenAI) must be in the submitting shell's environment;
# sbatch's default --export=ALL carries it into the job. Optionally KEYFILE names a
# mode-600 file that exports it. The key is never printed.
#
# Variables (export before sbatch to override):
#   MP5_ROOT   repo checkout with LAMM/ set up (see docs/minellm_service.md)
#   SIF        MineDojo image                      OVERLAY  optional writable overlay
#   ENV_FILE   host/container env file (GRADLE_USER_HOME, PYTHONUSERBASE, ...); its MP5 is ignored
#   CONFIG     YAML under configs/ (percipient.answer_model must be mllm)
#   TASKS      task JSON or directory              SEEDS    seed file
#   OUT        run directory                       WORKERS  keep at 1 (one MineLLM, no batching)
#   MINELLM_ENV / MINELLM_LOG_DIR / MINELLM_READY_TIMEOUT   see slurm/minellm_server.sh
set -uo pipefail

MP5_ROOT=${MP5_ROOT:-$HOME/MP5_plus}
SIF=${SIF:-/scratch/wz2999/archive/minedojo.sif}
OVERLAY=${OVERLAY:-}
ENV_FILE=${ENV_FILE:-/scratch/wz2999/mp5_env.sh}
CONFIG=${CONFIG:-$MP5_ROOT/configs/openrouter.yaml}
TASKS=${TASKS:-$MP5_ROOT/MP5_agent/agent/tasks/process}
SEEDS=${SEEDS:-$MP5_ROOT/configs/seeds_example.txt}
OUT=${OUT:-/scratch/$USER/mp5_runs/${SLURM_JOB_ID}}
WORKERS=${WORKERS:-1}
KEYFILE=${KEYFILE:-}
export MINELLM_LOG_DIR=${MINELLM_LOG_DIR:-$OUT}

mkdir -p "/scratch/$USER/mp5_runs/logs" "$OUT"

if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "$KEYFILE" ] && [ -r "$KEYFILE" ]; then
  # shellcheck disable=SC1090
  source "$KEYFILE"
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set in the job environment (value not shown)." >&2
  exit 2
fi

# shellcheck source=slurm/minellm_server.sh
source "$MP5_ROOT/slurm/minellm_server.sh"
trap minellm_stop EXIT
trap 'echo "received termination signal" >&2; exit 143' TERM INT

echo "node=${SLURMD_NODENAME:-$(cat /proc/sys/kernel/hostname)} job=${SLURM_JOB_ID:-} out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

minellm_start      || exit 1
minellm_wait_ready || exit 1
echo "MineLLM ready after ${MINELLM_READY_SECONDS}s, log $MINELLM_LOG"

overlay_args=()
[ -n "$OVERLAY" ] && overlay_args=(--overlay "$OVERLAY")

# No --cleanenv: OPENAI_API_KEY reaches the container through the inherited environment.
# ENV_FILE sets MP5 to its own checkout; MP5/PYTHONPATH are reset to MP5_ROOT after sourcing.
singularity exec --bind /scratch "${overlay_args[@]}" "$SIF" bash -c '
  source "$1" && export MP5="$2" PYTHONPATH="$2" ANONYMIZED_TELEMETRY=False
  cd "$2" && xvfb-run -a python3 eval/run_batch.py \
    --config "$3" --tasks "$4" --seeds "$5" --out "$6" \
    --workers "$7" --mllm-url "$8" --skip-done
' _ "$ENV_FILE" "$MP5_ROOT" "$CONFIG" "$TASKS" "$SEEDS" "$OUT" "$WORKERS" "$MINELLM_URL"
rc=$?
echo "run_batch exit code $rc"
exit $rc
