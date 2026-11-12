#!/bin/bash
#SBATCH --job-name=lfnav-mllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# Start the existing MineLLM HTTP server, wait for port 25541, run the baseline, then stop it.
# MINELLM_START must be a host command that launches LAMM/src/mllm_api_mc_v1.5.py in the background's environment.
# This file does not copy or modify that server.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
OUT=${OUT:-$SCR/lfnav/runs/minellm}
MINELLM_START=${MINELLM_START:?set MINELLM_START to the command that serves port 25541}
mkdir -p "$SCR/lfnav/logs" "$OUT"
cd "$MP5"
bash -c "$MINELLM_START" &
SERVER=$!
cleanup() { kill "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT
ready=0
for _ in $(seq 1 60); do
  python -c "import socket; socket.create_connection(('127.0.0.1', 25541), 2).close()" && ready=1 && break
  sleep 5
done
[ "$ready" = 1 ]
python -m lfnav.eval.run --config "$MP5/configs/lfnav/default.yaml" --tasks "${TASKS:-$MP5/lfnav/tasks/specs}" --seeds "${SEEDS:-$MP5/configs/lfnav/eval_seeds.txt}" --out "$OUT" --skip-done --perception minellm --policy c0 --backend "${BACKEND:-fake}"
