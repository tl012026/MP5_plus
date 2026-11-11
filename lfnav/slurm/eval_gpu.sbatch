#!/bin/bash
#SBATCH --job-name=lfnav-evalg
#SBATCH --account=
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# GPU perception with the training environment. MineDojo still needs the container;
# run the CPU eval script unless both stacks are on this node.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
OUT=${OUT:-$SCR/lfnav/runs/eval_gpu}
mkdir -p "$SCR/lfnav/logs" "$OUT"
if [ -f "$HOME/lfnav_train_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/lfnav_train_env.sh"
fi
cd "$MP5"
python lfnav/eval/measure_cost.py --out "$OUT/cost"
