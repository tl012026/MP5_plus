#!/bin/bash
# Generate an MP5 process task file with B's Parser (tools/parse_task.py) inside the
# MineDojo image, the same way slurm/integration_episode.sh runs the agent.
# Makes 1-2 paid OpenRouter chat requests (generation + at most one repair). No game.
#
#   bash ~/MP5_plus/slurm/run_parser.sh                 # default request: one wooden pickaxe
#   bash ~/MP5_plus/slurm/run_parser.sh --request "..." --goals '{"...": 1}'   # passed through
#
# Run on a host shell (not the OOD container) where OPENAI_API_KEY is exported.
# The key is only read from the environment and never printed.
#
# Output: $OUT (default /scratch/wz2999/integration_test/parser_<date_time>/), log at $OUT.log.
# The task file for run_agent.py --task is $OUT/process/generated_task.json. It is only
# written when validation passes as a runnable single-goal task. Otherwise see
# $OUT/validation.json.
#
# Variables: MP5_ROOT, SIF, ENV_FILE (its MP5 is ignored), OUT_ROOT, OUT.
set -uo pipefail

MP5_ROOT=${MP5_ROOT:-$HOME/MP5_plus}
SIF=${SIF:-/scratch/wz2999/archive/minedojo.sif}
ENV_FILE=${ENV_FILE:-/scratch/wz2999/mp5_env.sh}
OUT_ROOT=${OUT_ROOT:-/scratch/wz2999/integration_test}
OUT=${OUT:-$OUT_ROOT/parser_$(date +%Y%m%d_%H%M%S)}
LOG="$OUT.log"

die() { echo "ERROR: $*" >&2; exit 2; }
redact() { sed -E 's/(sk-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]{8,}/\1<redacted>/g; s/(Bearer )[A-Za-z0-9._-]+/\1<redacted>/g'; }

[ -n "${OPENAI_API_KEY:-}" ] || die "OPENAI_API_KEY is not set in this shell (value not shown)."
for f in "$SIF" "$ENV_FILE" "$MP5_ROOT/tools/parse_task.py"; do
    [ -r "$f" ] || die "missing or unreadable: $f"
done
command -v singularity >/dev/null || die "singularity not found; run this on a host shell, not inside the OOD container."
# parse_task creates OUT itself and refuses an existing directory.
[ -e "$OUT" ] && die "$OUT already exists; the Parser does not overwrite output directories."
mkdir -p "$(dirname "$OUT")" || die "cannot create $(dirname "$OUT")"

echo "Parser output: $OUT"
echo "Log: $LOG"
echo "Repo: $MP5_ROOT at $(git -C "$MP5_ROOT" rev-parse --short HEAD 2>/dev/null) on $(git -C "$MP5_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)"

# No --cleanenv: OPENAI_API_KEY reaches the container through the inherited environment,
# so it never appears on a command line. ENV_FILE sets MP5 to its own checkout; reset it.
singularity exec --bind /scratch "$SIF" bash -c '
    source "$1" || exit 90
    export MP5="$2" PYTHONPATH="$2" ANONYMIZED_TELEMETRY=False
    cd "$2" || exit 91
    out=$3; shift 3
    exec python tools/parse_task.py --out "$out" "$@"
' _ "$ENV_FILE" "$MP5_ROOT" "$OUT" "$@" < /dev/null 2>&1 | redact | tee "$LOG"
rc=${PIPESTATUS[0]}

echo
echo "parse_task.py exit code: $rc"
TASK_JSON="$OUT/process/generated_task.json"
if [ "$rc" -eq 0 ] && [ -s "$TASK_JSON" ]; then
    echo "Runnable task file: $TASK_JSON"
    echo "Next (A100 job):"
    echo "  TASK_FILE=$TASK_JSON MAX_STEPS=12000 env -u SBATCH_ACCOUNT -u SBATCH_PARTITION sbatch -A <account> ~/MP5_plus/slurm/integration_wooden_pickaxe.sbatch"
else
    echo "No runnable generated_task.json. Validation result:"
    cat "$OUT/validation.json" 2>/dev/null || echo "  (no validation.json; see $LOG)"
    [ "$rc" -eq 0 ] && rc=1
fi
exit "$rc"
