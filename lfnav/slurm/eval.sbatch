#!/bin/bash
#SBATCH --job-name=lfnav-eval
#SBATCH --account=
#SBATCH --partition=
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# CPU evaluation. --skip-done is always on. For a GPU perception backend, use eval_gpu.sbatch.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
ENVSH=${ENVSH:-$SCR/mp5_env.sh}
SIF=${SIF:-$SCR/env/minedojo.sif}
OUT=${OUT:-$SCR/lfnav/runs/eval}
TASKS=${TASKS:-$MP5/lfnav/tasks/specs}
SEEDS=${SEEDS:-$MP5/configs/lfnav/eval_seeds.txt}
POLICY=${POLICY:-c1}
PERCEPTION=${PERCEPTION:-gt}
mkdir -p "$SCR/lfnav/logs" "$OUT"
singularity exec "$SIF" bash -c "source '$ENVSH' && cd '$MP5' && xvfb-run -a python -m lfnav.eval.run --config '$MP5/configs/lfnav/default.yaml' --tasks '$TASKS' --seeds '$SEEDS' --out '$OUT' --skip-done --policy '$POLICY' --perception '$PERCEPTION' --backend minedojo"
