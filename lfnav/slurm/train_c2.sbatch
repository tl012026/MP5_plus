#!/bin/bash
#SBATCH --job-name=lfnav-c2
#SBATCH --account=
#SBATCH --partition=
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# CPU job. PyTorch 2.x comes from ~/lfnav_train_env.sh, not the MineDojo image.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
DEMOS=${DEMOS:-$SCR/lfnav/data/demos_dynamic}
OUT=${OUT:-$SCR/lfnav/ckpt/c2_gt_v1}
mkdir -p "$SCR/lfnav/logs" "$OUT"
if [ -f "$HOME/lfnav_train_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/lfnav_train_env.sh"
fi
cd "$MP5"
python lfnav/policy/train_c2.py --demos "$DEMOS" --out "$OUT" ${DAGGER:+--dagger}
