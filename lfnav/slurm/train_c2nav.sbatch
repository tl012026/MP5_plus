#!/bin/bash
#SBATCH --job-name=lfnav-c2nav
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# C2Nav on cached DINOv2 features. Training env, not MineDojo.
set -euo pipefail
export PYTHONUNBUFFERED=1
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
FEAT=${FEAT:-$SCR/lfnav/features/joint_v1}
OUT=${OUT:-$SCR/lfnav/ckpt/c2nav}
mkdir -p "$SCR/lfnav/logs" "$OUT"
if [ -f "$HOME/lfnav_train_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/lfnav_train_env.sh"
fi
cd "$MP5"
python lfnav/policy/train_c2nav.py --features "$FEAT" --out "$OUT"
