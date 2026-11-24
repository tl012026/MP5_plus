#!/bin/bash
#SBATCH --job-name=mp5-int-pickaxe
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wz2999/logs/%x_%j.out
#SBATCH --error=/scratch/wz2999/logs/%x_%j.err

# Joint test: real MineLLM (A100) + one craft_wooden_pickaxe episode in MineDojo,
# OpenRouter chat/embeddings (configs/openrouter.yaml), 1 seed, max_steps 12000.
# The CPU rehearsal (job 38928) used all 2000 steps in under 5 minutes, still in find,
# so this job uses 12000 (the value in openrouter.yaml). Override with MAX_STEPS=... sbatch ...
# Summary: /scratch/wz2999/integration_test/summary_<jobid>.md
#
# Submit from a host shell where OPENAI_API_KEY (OpenRouter key) is exported;
# sbatch's default --export=ALL passes it to the job. Do not add --export=NONE.
# Unset the SBATCH_* vars from mp5_env.sh, which would override the lines above:
#   env -u SBATCH_ACCOUNT -u SBATCH_PARTITION sbatch -A <account> slurm/integration_wooden_pickaxe.sbatch
# With a Parser-generated task file instead of the handwritten one:
#   TASK_FILE=/scratch/.../parser_<ts>/process/generated_task.json MAX_STEPS=12000 \
#     env -u SBATCH_ACCOUNT -u SBATCH_PARTITION sbatch -A <account> slurm/integration_wooden_pickaxe.sbatch
# Run slurm/integration_rehearsal_cpu.sbatch first; it needs no GPU.

MP5_ROOT=${MP5_ROOT:-$HOME/MP5_plus}
export MINELLM_MODE=real
export MAX_STEPS=${MAX_STEPS:-12000}
# Task file: TASK_FILE if set (e.g. a Parser output from slurm/run_parser.sh), else the
# handwritten one. The path must contain a "process" directory so run_agent.py checks
# the final inventory for success.
export TASK_SOURCE=${TASK_FILE:+TASK_FILE}
export TASK=${TASK_FILE:-$MP5_ROOT/MP5_agent/agent/tasks/process/craft_wooden_pickaxe.json}
exec bash "$MP5_ROOT/slurm/integration_episode.sh"
