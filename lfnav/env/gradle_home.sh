# Per-job Gradle user home. Source this from MineDojo sbatch templates on the
# compute node (host, before singularity). The master $SCR/.gradle is read-only
# for evaluation jobs. To rebuild or refresh the master:
#   GRADLE_USE_MASTER=1 sbatch ...
# that job must write $SCR/.gradle directly and must not source the copy path.

_lfnav_gradle_cleanup() {
  if [ "${_LFNAV_GRADLE_OWNED:-0}" = "1" ] && [ -n "${GRADLE_USER_HOME:-}" ]; then
    rm -rf "$GRADLE_USER_HOME"
  fi
}

if [ "${GRADLE_USE_MASTER:-0}" = "1" ]; then
  export GRADLE_USER_HOME="${GRADLE_MASTER:-${SCR:-/scratch/$USER}/.gradle}"
  _LFNAV_GRADLE_OWNED=0
else
  _gradle_src="${GRADLE_MASTER:-${SCR:-/scratch/$USER}/.gradle}"
  _gradle_job="${SLURM_JOB_ID:-$$}"
  _gradle_tmp="${SLURM_TMPDIR:-}"
  if [ -z "$_gradle_tmp" ]; then
    _gradle_tmp="/tmp"
  fi
  _gradle_py="${MP5:-$HOME/MP5_plus}/lfnav/env/gradle_home.py"
  if [ "${GRADLE_WORKER_COPIES:-0}" = "1" ]; then
    _gradle_worker="${WORKER_ID:-${SLURM_PROCID:-0}}"
    export GRADLE_USER_HOME="${_gradle_tmp}/gradle_${_gradle_job}_w${_gradle_worker}"
  else
    export GRADLE_USER_HOME="${_gradle_tmp}/gradle_${_gradle_job}"
  fi
  python3 "$_gradle_py" copy --src "$_gradle_src" --dst "$GRADLE_USER_HOME"
  _LFNAV_GRADLE_OWNED=1
  trap _lfnav_gradle_cleanup EXIT
fi

lfnav_sif_exec() {
  # singularity exec with the job-local Gradle home bound in.
  local sif="${1:?sif}"
  shift
  local extra=()
  if [ -n "${GRADLE_USER_HOME:-}" ]; then
    extra+=(-B "$GRADLE_USER_HOME")
  fi
  singularity exec "${extra[@]}" "$sif" "$@"
}
