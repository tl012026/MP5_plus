#!/bin/bash
#SBATCH --job-name=lfnav-demo
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# Expert rollouts. Finished seed files are skipped. Seeds ending in 9 are rejected.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
ENVSH=${ENVSH:-$SCR/mp5_env.sh}
SIF=${SIF:-$SCR/env/minedojo.sif}
OUT=${OUT:-$SCR/lfnav/data/demos}
TASKS=${TASKS:-$MP5/lfnav/tasks/specs}
SEEDS=${SEEDS:-$MP5/configs/lfnav/train_seeds.txt}
mkdir -p "$SCR/lfnav/logs" "$OUT"
# shellcheck disable=SC1091
source "$MP5/lfnav/env/gradle_home.sh"
lfnav_sif_exec "$SIF" bash -c "source '$ENVSH' && export GRADLE_USER_HOME='$GRADLE_USER_HOME' && cd '$MP5' && xvfb-run -a python lfnav/data/collect_demos.py --tasks '$TASKS' --seeds '$SEEDS' --out '$OUT' --backend minedojo"
