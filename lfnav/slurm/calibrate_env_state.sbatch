#!/bin/bash
#SBATCH --job-name=lfnav-env-state
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# Measure sun_brightness, light_level, is_raining, and biome_id.
# Copy the printed thresholds into default.yaml and obs_keys.yaml.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
ENVSH=${ENVSH:-$SCR/mp5_env.sh}
SIF=${SIF:-$SCR/env/minedojo.sif}
mkdir -p "$SCR/lfnav/logs" "$SCR/lfnav/calib"
# shellcheck disable=SC1091
source "$MP5/lfnav/env/gradle_home.sh"
lfnav_sif_exec "$SIF" bash -c "export GRADLE_USER_HOME='$GRADLE_USER_HOME' && source '$ENVSH' && cd '$MP5' && xvfb-run -a python lfnav/gt/calibrate_env_state.py --out '$SCR/lfnav/calib/env_state.json'"
