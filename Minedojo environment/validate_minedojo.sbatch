#!/bin/bash
#SBATCH --account=
#SBATCH --partition=
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --job-name=validate_minedojo
#SBATCH --requeue
#SBATCH --output=/scratch/%u/logs/%x_%j.out

# Set MP5_ENV_FILE to your scratch env script. Do not expand $USER inside
# the container (the image sets USER=user).
ENV_FILE=${MP5_ENV_FILE:?set MP5_ENV_FILE to your mp5_env.sh}
source "$ENV_FILE"

singularity exec "$SCR/env/minedojo.sif" bash -c "
  source $ENV_FILE
  echo GRADLE_USER_HOME=\$GRADLE_USER_HOME
  ls \$GRADLE_USER_HOME/init.d
  cd \$MP5 && xvfb-run -a python \"Minedojo environment/validate_minedojo.py\"
"
echo "exit code: $?"
