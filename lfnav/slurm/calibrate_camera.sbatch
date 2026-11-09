#!/bin/bash
#SBATCH --job-name=lfnav-fov
#SBATCH --account=
#SBATCH --partition=
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# Turn in place, fit K, then check pig-ray overlays.
# Create /scratch/$USER/lfnav/logs on the login node before sbatch: Slurm
# opens --output before this script runs.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
ENVSH=${ENVSH:-$SCR/mp5_env.sh}
SIF=${SIF:-$SCR/env/minedojo.sif}
ROT=${ROT:-$SCR/lfnav/calib/rot}
INTR=${INTR:-$SCR/lfnav/calib/intrinsics.json}
ALIGN=${ALIGN:-$SCR/lfnav/calib/align}
mkdir -p "$SCR/lfnav/logs" "$SCR/lfnav/calib" "$ROT" "$ALIGN"
# shellcheck disable=SC1091
source "$MP5/lfnav/env/gradle_home.sh"
lfnav_sif_exec "$SIF" bash -c "export GRADLE_USER_HOME='$GRADLE_USER_HOME' && source '$ENVSH' && cd '$MP5' && xvfb-run -a python lfnav/gt/capture_rotations.py --out '$ROT'"
lfnav_sif_exec "$SIF" bash -c "export GRADLE_USER_HOME='$GRADLE_USER_HOME' && source '$ENVSH' && cd '$MP5' && python lfnav/gt/fit_intrinsics.py --in '$ROT' --out '$INTR'"
lfnav_sif_exec "$SIF" bash -c "export GRADLE_USER_HOME='$GRADLE_USER_HOME' && source '$ENVSH' && cd '$MP5' && xvfb-run -a python lfnav/gt/verify_alignment.py --intrinsics '$INTR' --out '$ALIGN'"
