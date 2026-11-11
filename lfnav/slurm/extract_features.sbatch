#!/bin/bash
#SBATCH --job-name=lfnav-feat
#SBATCH --account=
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# Frozen DINOv2 features from filtered expert demos (one .npy per episode).
# Set SRC= to use the leftover perception-shard path instead.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
CONDA=${CONDA:-$SCR/conda_envs/lfnav}
VER=${VER:-joint_v1}
OUT=${OUT:-$SCR/lfnav/features/$VER}
mkdir -p "$SCR/lfnav/logs" "$OUT"
if [ -f "$HOME/lfnav_train_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/lfnav_train_env.sh"
fi
cd "$MP5"
if [ -n "${SRC:-}" ]; then
  python lfnav/perception/extract_features.py --src "$SRC" --out "${OUT_SHARDS:-$SCR/lfnav/features}" --encoder "${ENCODER:-dinov2_vits14}"
else
  python lfnav/perception/extract_features.py --out "$OUT" --encoder "${ENCODER:-dinov2_vits14}" --version "$VER" --config "${CONFIG:-configs/lfnav/main.yaml}"
fi
