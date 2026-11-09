#!/bin/bash
#SBATCH --job-name=lfnav-base
#SBATCH --account=
#SBATCH --partition=
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-47
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%A_%a.err

# 16 tasks × 3 difficulties. Each array index runs 5 baseline seeds.
# POLICY=c0|c1 PERCEPTION=gt
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
ENVSH=${ENVSH:-$SCR/mp5_env.sh}
SIF=${SIF:-$SCR/env/minedojo.sif}
POLICY=${POLICY:-c1}
PERCEPTION=${PERCEPTION:-gt}
CFG=${CFG:-$MP5/configs/lfnav/baseline_eval.yaml}
SEEDS=${SEEDS:-$MP5/configs/lfnav/eval_seeds_baseline.txt}
ROOT_OUT=${ROOT_OUT:-$SCR/lfnav/runs/baseline}
mkdir -p "$SCR/lfnav/logs"
# shellcheck disable=SC1091
source "$MP5/lfnav/env/gradle_home.sh"
TASKS=(1-1 1-2 1-3 1-4 2-1 2-2 2-3 2-4 3-1 3-2 3-3 3-4 4-1 4-2 4-3 4-4)
DIFFS=(easy medium hard)
idx=${SLURM_ARRAY_TASK_ID}
TASK=${TASKS[$((idx / 3))]}
DIFF=${DIFFS[$((idx % 3))]}
OUT="$ROOT_OUT/$POLICY/$DIFF/$TASK"
mkdir -p "$OUT"
SPEC="$MP5/lfnav/tasks/specs/${TASK}.json"
lfnav_sif_exec "$SIF" bash -c "source '$ENVSH' && export GRADLE_USER_HOME='$GRADLE_USER_HOME' && cd '$MP5' && xvfb-run -a python -m lfnav.eval.run --config '$CFG' --tasks '$SPEC' --seeds '$SEEDS' --out '$OUT' --skip-done --policy '$POLICY' --perception '$PERCEPTION' --backend minedojo --difficulty '$DIFF'"
