#!/bin/bash
#SBATCH --job-name=lfnav-collect
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# Perception frames. Requeue is safe: finished shards are skipped.
# BACKEND=fake is only for a smoke test. Real frames need BACKEND=minedojo after probe_obs.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
ENVSH=${ENVSH:-$SCR/mp5_env.sh}
SIF=${SIF:-$SCR/env/minedojo.sif}
FRAMES=${FRAMES:-1000}
BACKEND=${BACKEND:-minedojo}
OUT=${OUT:-$SCR/lfnav/data/perception_pilot}
mkdir -p "$SCR/lfnav/logs" "$OUT"
# shellcheck disable=SC1091
source "$MP5/lfnav/env/gradle_home.sh"
lfnav_sif_exec "$SIF" bash -c "export GRADLE_USER_HOME='$GRADLE_USER_HOME' && source '$ENVSH' && cd '$MP5' && xvfb-run -a python lfnav/data/collect_perception.py --config '$MP5/configs/lfnav/default.yaml' --out '$OUT' --frames $FRAMES --backend $BACKEND"
