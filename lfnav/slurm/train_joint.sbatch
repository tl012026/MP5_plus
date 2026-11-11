#!/bin/bash
#SBATCH --job-name=lfnav-joint
#SBATCH --account=
#SBATCH --partition=
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# Joint A+C2 on cached DINOv2 features. Training conda env, not MineDojo.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
VER=${VER:-joint_v1}
FEAT=${FEAT:-$SCR/lfnav/features/$VER}
OUT=${OUT:-$SCR/lfnav/ckpt/joint_v1}
mkdir -p "$SCR/lfnav/logs" "$OUT"
if [ -f "$HOME/lfnav_train_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/lfnav_train_env.sh"
fi
cd "$MP5"
python lfnav/policy/train_joint.py --features "$FEAT" --out "$OUT" --version "$VER"
