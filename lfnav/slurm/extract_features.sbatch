#!/bin/bash
#SBATCH --job-name=lfnav-feat
#SBATCH --account=
#SBATCH --partition=
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# Frozen DINOv2 features. Existing shards are skipped. Uses the training conda env, not MineDojo.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
CONDA=${CONDA:-$SCR/conda_envs/lfnav}
SRC=${SRC:-$SCR/lfnav/data/perception}
OUT=${OUT:-$SCR/lfnav/features}
mkdir -p "$SCR/lfnav/logs" "$OUT"
if [ -f "$HOME/lfnav_train_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/lfnav_train_env.sh"
fi
cd "$MP5"
python lfnav/perception/extract_features.py --src "$SRC" --out "$OUT" --encoder "${ENCODER:-dinov2_vits14}"
