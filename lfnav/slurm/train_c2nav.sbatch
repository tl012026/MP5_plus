#!/bin/bash
#SBATCH --job-name=lfnav-c2nav
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# CPU job on cached DINOv2 features. Does not submit itself.
set -euo pipefail
export PYTHONUNBUFFERED=1
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
CFG=${CFG:-$MP5/configs/lfnav/c2nav_easy.yaml}
FEAT=${FEAT:-$SCR/lfnav/features/nav_v3_easy}
OUT=${OUT:-$SCR/lfnav/ckpt/c2nav_easy_r3}
THREADS=${THREADS:-2}
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
mkdir -p "$SCR/lfnav/logs" "$OUT"
if [ -f "$HOME/lfnav_train_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/lfnav_train_env.sh"
fi
cd "$MP5"
python lfnav/policy/train_c2nav.py --config "$CFG" --features "$FEAT" --out "$OUT"
