#!/bin/bash
#SBATCH --job-name=lfnav-fov
#SBATCH --account=
#SBATCH --partition=
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# Fit fov from a pixel you measured. Example:
#   PIXEL=80 SIZE=364 ANGLE=15 AXIS=horizontal sbatch lfnav/slurm/calibrate_camera.sbatch
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
ENVSH=${ENVSH:-$SCR/mp5_env.sh}
SIF=${SIF:-$SCR/env/minedojo.sif}
PIXEL=${PIXEL:?set PIXEL}
SIZE=${SIZE:?set SIZE}
ANGLE=${ANGLE:?set ANGLE}
AXIS=${AXIS:-horizontal}
mkdir -p "$SCR/lfnav/logs" "$SCR/lfnav/calib"
# shellcheck disable=SC1091
source "$MP5/lfnav/env/gradle_home.sh"
lfnav_sif_exec "$SIF" bash -c "export GRADLE_USER_HOME='$GRADLE_USER_HOME' && source '$ENVSH' && cd '$MP5' && python lfnav/gt/calibrate_camera.py --pixel $PIXEL --size $SIZE --angle-deg $ANGLE --axis $AXIS --out '$SCR/lfnav/calib/fov.json'"
