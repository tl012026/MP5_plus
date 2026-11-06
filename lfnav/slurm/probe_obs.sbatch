#!/bin/bash
#SBATCH --job-name=lfnav-probe
#SBATCH --account=
#SBATCH --partition=
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# Print one MineDojo observation. Paths are expanded on the host. The container
# must not use $USER (it is "user" inside the image).
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
ENVSH=${ENVSH:-$SCR/mp5_env.sh}
SIF=${SIF:-$SCR/env/minedojo.sif}
mkdir -p "$SCR/lfnav/logs" "$SCR/lfnav/probe"
singularity exec "$SIF" bash -c "source '$ENVSH' && cd '$MP5' && xvfb-run -a python lfnav/gt/probe_obs.py --out '$SCR/lfnav/probe/obs.json'"
