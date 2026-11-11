#!/bin/bash
#SBATCH --job-name=lfnav-perc
#SBATCH --account=
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
mkdir -p "$SCR/lfnav/logs" "$SCR/lfnav/checkpoints"
if [ -f "$HOME/lfnav_train_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/lfnav_train_env.sh"
fi
cd "$MP5"
python lfnav/perception/train_perception.py --features "${FEATURES:-$SCR/lfnav/features}" --out "${OUT:-$SCR/lfnav/checkpoints/perception.pt}"
