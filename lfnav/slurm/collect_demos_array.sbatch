#!/bin/bash
#SBATCH --job-name=lfnav-demo
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-47
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%A_%a.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%A_%a.err

# Array 0–47 → (task, difficulty) via lfnav.eval.baseline_index.
# Tree/grass cells skip medium/hard. CFG/SEEDS/OUT override the dynamic-demo defaults.
# Host Python may only use the stdlib. Anything that imports numpy must run inside lfnav_sif_exec.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
ENVSH=${ENVSH:-$SCR/mp5_env.sh}
SIF=${SIF:-$SCR/env/minedojo.sif}
CFG=${CFG:-$MP5/configs/lfnav/main.yaml}
SEEDS=${SEEDS:-$MP5/configs/lfnav/train_seeds_demo.txt}
OUT=${OUT:-$SCR/lfnav/data/demos_main_v3}
mkdir -p "$SCR/lfnav/logs" "$OUT"
idx=${SLURM_ARRAY_TASK_ID}
PAIR=$(cd "$MP5" && python3 -c "from lfnav.eval.baseline_index import cell; t,d=cell($idx); print(t, d)")
TASK=${PAIR%% *}
DIFF=${PAIR##* }
SPEC="$MP5/lfnav/tasks/specs/${TASK}.json"
SKIP=$(python3 -c "import json; d=json.load(open('$SPEC')); print(1 if d.get('id')=='4-3' or (d.get('target') in ('tree','grass') and '$DIFF'!='easy') else 0)")
if [ "$SKIP" = "1" ]; then
  echo "skip block target $TASK $DIFF"
  exit 0
fi
# shellcheck disable=SC1091
source "$MP5/lfnav/env/gradle_home.sh"
lfnav_sif_exec "$SIF" bash -c "source '$ENVSH' && export GRADLE_USER_HOME='$GRADLE_USER_HOME' && cd '$MP5' && xvfb-run -a python -m lfnav.data.collect_demos --config '$CFG' --tasks '$SPEC' --seeds '$SEEDS' --out '$OUT' --backend minedojo --difficulty '$DIFF'"
