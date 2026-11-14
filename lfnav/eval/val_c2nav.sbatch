#!/bin/bash
#SBATCH --job-name=lfnav-c2val
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/lfnav/logs/%x-%j.out
#SBATCH --error=/scratch/%u/lfnav/logs/%x-%j.err

# Four validation episodes for one C2Nav checkpoint. CKPT and VAL_OUT are required.
set -euo pipefail
SCR=${SCR:-/scratch/$USER}
MP5=${MP5:-$HOME/MP5_plus}
ENVSH=${ENVSH:-$SCR/mp5_env.sh}
SIF=${SIF:-$SCR/env/minedojo.sif}
CFG=${CFG:-$MP5/configs/lfnav/c2nav_easy.yaml}
CKPT=${CKPT:?CKPT is required}
VAL_OUT=${VAL_OUT:?VAL_OUT is required}
mkdir -p "$SCR/lfnav/logs" "$VAL_OUT"
if [ -f "$HOME/lfnav_train_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/lfnav_train_env.sh"
fi
# shellcheck disable=SC1091
source "$MP5/lfnav/env/gradle_home.sh"
lfnav_sif_exec "$SIF" bash -c "source '$ENVSH' && export GRADLE_USER_HOME='$GRADLE_USER_HOME' && cd '$MP5' && xvfb-run -a python -m lfnav.eval.val_c2nav --config '$CFG' --ckpt '$CKPT' --out '$VAL_OUT' --backend minedojo"
python "$MP5/lfnav/eval/summarize_c2nav.py" --in "$VAL_OUT" --ckpt "$CKPT"
