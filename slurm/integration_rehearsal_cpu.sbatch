#!/bin/bash
#SBATCH --job-name=mp5-int-rehearsal
#SBATCH --partition=n2c48m24
#SBATCH --mem=22G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wz2999/logs/%x_%j.out
#SBATCH --error=/scratch/wz2999/logs/%x_%j.err

# CPU rehearsal of slurm/integration_wooden_pickaxe.sbatch: same flow, but MineLLM is
# tools/fake_minellm_server.py (same /image interface, fixed answer, no GPU). It checks
# MineDojo startup, OpenRouter calls, the agent's requests to 127.0.0.1:25541 and the
# output files without waiting for an A100. Frames the agent sends are saved under
# /scratch/wz2999/integration_test/<jobid>/fake_minellm_frames/.
# Summary: /scratch/wz2999/integration_test/summary_<jobid>.md
#
# Resources: n2c48m24 nodes have 24 CPUs and 23500 MB RAM and are allocated exclusively,
# so --mem must stay below 23500M (24G = 24576M is rejected). 8 CPUs matches the A100 job.
# A local test on a node in this partition had 21.5 GB available and MineDojo plus the fake
# server started fine.
#
# Submit from a host shell where OPENAI_API_KEY (OpenRouter key) is exported:
#   env -u SBATCH_ACCOUNT -u SBATCH_PARTITION sbatch -A <account> slurm/integration_rehearsal_cpu.sbatch

MP5_ROOT=${MP5_ROOT:-$HOME/MP5_plus}
export MINELLM_MODE=fake
exec bash "$MP5_ROOT/slurm/integration_episode.sh"
