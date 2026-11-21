#!/bin/bash
# One MP5 episode against MineLLM, run from inside an sbatch job:
#   check OPENAI_API_KEY -> start MineLLM (real or fake) and wait for its port
#   -> wait for host memory to settle -> run_agent.py in the MineDojo image
#   -> stop MineLLM -> write a markdown summary.
# Called by slurm/integration_wooden_pickaxe.sbatch (MINELLM_MODE=real) and
# slurm/integration_rehearsal_cpu.sbatch (MINELLM_MODE=fake). Not meant to be sourced.
#
# The API key comes only from the job environment (sbatch --export=ALL, the default).
# This script never prints it, never runs env/printenv, and does not use set -x.
#
# Variables (export before sbatch to override):
#   MINELLM_MODE     real | fake                     (set by the sbatch wrapper)
#   MP5_ROOT         repo checkout                   default $HOME/MP5_plus
#   SIF              MineDojo image                  default /scratch/wz2999/archive/minedojo.sif
#   ENV_FILE         MineDojo env file               default /scratch/wz2999/mp5_env.sh (its MP5 is ignored)
#   BASE_CONFIG      default $MP5_ROOT/configs/openrouter.yaml
#   TASK             default $MP5_ROOT/MP5_agent/agent/tasks/process/craft_wooden_pickaxe.json
#   SEED             default 10001
#   MAX_STEPS        default 2000 (written into a copy of BASE_CONFIG)
#   AGENT_TIMEOUT    seconds before the agent is stopped, default 6000
#   RESULTS_ROOT     default /scratch/wz2999/integration_test
#   MINELLM_ENV / MINELLM_LOG_DIR / MINELLM_READY_TIMEOUT   see slurm/minellm_server.sh
set -uo pipefail

MINELLM_MODE=${MINELLM_MODE:-real}
MP5_ROOT=${MP5_ROOT:-$HOME/MP5_plus}
SIF=${SIF:-/scratch/wz2999/archive/minedojo.sif}
ENV_FILE=${ENV_FILE:-/scratch/wz2999/mp5_env.sh}
BASE_CONFIG=${BASE_CONFIG:-$MP5_ROOT/configs/openrouter.yaml}
TASK=${TASK:-$MP5_ROOT/MP5_agent/agent/tasks/process/craft_wooden_pickaxe.json}
SEED=${SEED:-10001}
MAX_STEPS=${MAX_STEPS:-2000}
AGENT_TIMEOUT=${AGENT_TIMEOUT:-6000}
RESULTS_ROOT=${RESULTS_ROOT:-/scratch/wz2999/integration_test}

JOB=${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=$RESULTS_ROOT/$JOB
SUMMARY=$RESULTS_ROOT/summary_${JOB}.md
TASK_NAME=$(basename "$TASK" .json)
OUT_DIR=$RUN_DIR/episodes/$TASK_NAME/$SEED
EVENTS=$RUN_DIR/episodes/$TASK_NAME/${SEED}.events.jsonl
AGENT_STDOUT=$RUN_DIR/agent_stdout.log
CONFIG=$RUN_DIR/config.yaml
export MINELLM_LOG_DIR=${MINELLM_LOG_DIR:-$RUN_DIR}
FAKE_REQUESTS=""

STATUS="FAILED"
AGENT_RC=""
mkdir -p "$RUN_DIR" || { echo "cannot create $RUN_DIR" >&2; exit 1; }

say() { printf '%s\n' "$*" >> "$SUMMARY"; }
log() { echo "[$(date '+%F %T')] $*"; }
# Mask anything that looks like an API key before copying log text into the summary.
redact() { sed -E 's/(sk-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]{8,}/\1<redacted>/g; s/(Bearer )[A-Za-z0-9._-]+/\1<redacted>/g'; }
node_name() { echo "${SLURMD_NODENAME:-$(cat /proc/sys/kernel/hostname 2>/dev/null)}"; }
mem_avail_gb() { awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo; }

fail() {
    say ""; say "**ERROR: $***"
    log "ERROR: $*"
    exit 1
}

finish() {
    local rc=$?
    say ""; say "## Shutdown"; say ""
    if declare -F minellm_stop >/dev/null; then
        if minellm_stop 2>>"$RUN_DIR/minellm_stop.log"; then say "- MineLLM ($MINELLM_MODE) stopped at $(date '+%F %T')."
        else say "- **WARNING: MineLLM process may still be running (pid ${MINELLM_PID:-?}).**"; fi
    else
        say "- MineLLM was not started."
    fi
    say ""; say "## Files"; say ""
    say "- Run directory: \`$RUN_DIR\`"
    say "- Episode output: \`$OUT_DIR\` (logs/agent.log, images/, memory/)"
    say "- Events: \`$EVENTS\`"
    say "- Agent stdout/stderr: \`$AGENT_STDOUT\`"
    say "- MineLLM log: \`${MINELLM_LOG:-not started}\`"
    [ -n "$FAKE_REQUESTS" ] && say "- Fake MineLLM saved frames: \`$RUN_DIR/fake_minellm_frames/\`"
    say "- Config used: \`$CONFIG\`"
    say ""; say "Final status: **$STATUS** (script exit code $rc, agent exit code ${AGENT_RC:-not run})"
    log "summary: $SUMMARY"
}
trap finish EXIT
trap 'say ""; say "**ERROR: received termination signal (time limit or scancel) at $(date "+%F %T").**"; exit 143' TERM INT

# ------------------------------------------------------------------ header
cat > "$SUMMARY" <<EOF
# Integration episode: ${TASK_NAME}, job ${JOB}

- MineLLM mode: **${MINELLM_MODE}**
- Started: $(date '+%F %T %Z'), node $(node_name), partition ${SLURM_JOB_PARTITION:-n/a}
- Repo: \`$MP5_ROOT\` at $(git -C "$MP5_ROOT" rev-parse --short HEAD 2>/dev/null) on $(git -C "$MP5_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)$( [ -n "$(git -C "$MP5_ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ] && echo " (tracked files modified)")
- Task: \`$TASK\`, seed $SEED, max_steps $MAX_STEPS, agent timeout ${AGENT_TIMEOUT}s
EOF

# ------------------------------------------------------------------ checks
[ -n "${OPENAI_API_KEY:-}" ] || fail "OPENAI_API_KEY is not set in the job environment (value not shown). Submit from a shell where it is exported."
say "- OPENAI_API_KEY: present (value not shown)"

for f in "$SIF" "$ENV_FILE" "$BASE_CONFIG" "$TASK" "$MP5_ROOT/slurm/minellm_server.sh"; do
    [ -r "$f" ] || fail "missing or unreadable: $f"
done
command -v singularity >/dev/null || fail "singularity not found on $(node_name)"

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 https://openrouter.ai/api/v1/models || true)
say "- OpenRouter reachability (\`GET /api/v1/models\`, no key): HTTP $code"
[ "$code" = "200" ] || fail "cannot reach openrouter.ai from $(node_name) (HTTP $code)"

# Copy of the config with eval.max_steps replaced. Not committed.
sed -E "s/^([[:space:]]+max_steps:)[[:space:]]*[0-9]+/\1 $MAX_STEPS/" "$BASE_CONFIG" > "$CONFIG"
grep -Eq "^[[:space:]]+max_steps:[[:space:]]*$MAX_STEPS\$" "$CONFIG" || fail "could not set max_steps in $CONFIG"
say "- Config: \`$BASE_CONFIG\` copied to \`$CONFIG\` with eval.max_steps=$MAX_STEPS"

# ------------------------------------------------------------------ MineLLM
# shellcheck source=slurm/minellm_server.sh
source "$MP5_ROOT/slurm/minellm_server.sh"
say ""; say "## MineLLM ($MINELLM_MODE)"; say ""
MEM_BEFORE=$(mem_avail_gb)
if [ "$MINELLM_MODE" = "real" ]; then
    command -v nvidia-smi >/dev/null && say "- GPU: \`$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)\`"
    minellm_start      || fail "minellm_start failed (see MINELLM log / job stderr)"
    minellm_wait_ready || { say '```'; tail -n 40 "$MINELLM_LOG" | redact >> "$SUMMARY"; say '```'; fail "MineLLM did not become ready"; }
    say "- Ready after **${MINELLM_READY_SECONDS} s**; GPU memory: \`$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1)\`"
    grep -q "Loading delta checkpoint" "$MINELLM_LOG" && say "- Log confirms the MineLLM delta checkpoint was loaded."
    grep -q "while .* not found" "$MINELLM_LOG" && say "- **WARNING: delta checkpoint not found; server is plain Vicuna.**"
elif [ "$MINELLM_MODE" = "fake" ]; then
    MINELLM_LOG="$MINELLM_LOG_DIR/fake_minellm_${JOB}.log"
    _MINELLM_T0=$(date +%s)
    PYTHONUNBUFFERED=1 setsid nohup python3 "$MP5_ROOT/tools/fake_minellm_server.py" \
        --save-dir "$RUN_DIR/fake_minellm_frames" > "$MINELLM_LOG" 2> "$RUN_DIR/fake_minellm_stderr.log" < /dev/null &
    MINELLM_PID=$!
    FAKE_REQUESTS=$MINELLM_LOG
    minellm_wait_ready || { say '```'; tail -n 40 "$RUN_DIR/fake_minellm_stderr.log" >> "$SUMMARY"; say '```'; fail "fake MineLLM did not start"; }
    say "- Fake server (\`tools/fake_minellm_server.py\`) ready after ${MINELLM_READY_SECONDS} s; it answers every question with a fixed sentence."
else
    fail "MINELLM_MODE must be real or fake, got '$MINELLM_MODE'"
fi
say "- URL: $MINELLM_URL"

# Let host memory settle after the fp32 -> fp16 load before starting Minecraft:
# wait until MemAvailable changes by < 0.5 GB over 20 s (at most 5 minutes).
prev=$(mem_avail_gb); settled=0
for _ in $(seq 1 15); do
    sleep 20; cur=$(mem_avail_gb)
    if awk -v a="$prev" -v b="$cur" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<0.5)}'; then settled=1; break; fi
    prev=$cur
done
say "- Host MemAvailable: ${MEM_BEFORE} GB before MineLLM, $(mem_avail_gb) GB before MineDojo (settled: $([ $settled = 1 ] && echo yes || echo 'no, 5 min limit'))"

# ------------------------------------------------------------------ agent
say ""; say "## Agent episode"; say ""
say "Command (inside \`$SIF\`, \`xvfb-run -a\`, cwd \`MP5_agent/agent\`):"
say '```'
say "python run_agent.py --config $CONFIG --task $TASK --seed $SEED --out_dir $OUT_DIR \\"
say "  --mllm_url $MINELLM_URL --answer_model mllm --answer_method active"
say '```'
log "starting agent (timeout ${AGENT_TIMEOUT}s), stdout in $AGENT_STDOUT"
T_AGENT=$(date +%s)
# No --cleanenv: OPENAI_API_KEY reaches the container through the inherited environment,
# so it never appears on a command line. ENV_FILE sets MP5 to its own checkout; reset it.
timeout --signal=TERM --kill-after=60 "$AGENT_TIMEOUT" \
  singularity exec --bind /scratch "$SIF" bash -c '
    source "$1" || exit 90
    export MP5="$2" PYTHONPATH="$2" ANONYMIZED_TELEMETRY=False
    cd "$2/MP5_agent/agent" || exit 91
    exec xvfb-run -a python run_agent.py --config "$3" --task "$4" --seed "$5" --out_dir "$6" \
      --mllm_url "$7" --answer_model mllm --answer_method active
  ' _ "$ENV_FILE" "$MP5_ROOT" "$CONFIG" "$TASK" "$SEED" "$OUT_DIR" "$MINELLM_URL" \
  > "$AGENT_STDOUT" 2>&1 < /dev/null
AGENT_RC=$?
AGENT_S=$(( $(date +%s) - T_AGENT ))
say "- Agent exit code: $AGENT_RC after ${AGENT_S} s$([ "$AGENT_RC" = 124 ] && echo " (**killed by AGENT_TIMEOUT**)")"
[ "$MINELLM_MODE" = "real" ] && say "- GPU memory after episode: \`$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>&1)\`"

# ------------------------------------------------------------------ results
say ""; say "## Result (from the \`episode_end\` event)"; say ""
if [ -s "$EVENTS" ]; then
    python3 -I - "$EVENTS" >> "$SUMMARY" 2>&1 <<'PY'
import collections, json, sys
counts, end, perc = collections.Counter(), None, []
for line in open(sys.argv[1], encoding="utf-8"):
    try:
        rec = json.loads(line)
    except ValueError:
        continue
    counts[rec.get("event")] += 1
    if rec.get("event") == "episode_end":
        end = rec
    if "percept" in str(rec.get("event", "")):
        perc.append(rec)
if end:
    for k in ("outcome", "success", "steps", "elapsed_s", "failure_category", "llm_counts"):
        print(f"- {k}: `{end.get(k)}`")
else:
    print("- **No episode_end event** (the agent did not finish normally).")
print("- Event counts: " + ", ".join(f"`{k}`={v}" for k, v in sorted(counts.items(), key=lambda x: str(x[0]))))
if perc:
    print("\nFirst perception events (truncated):\n")
    for rec in perc[:5]:
        print("- `" + json.dumps(rec, ensure_ascii=False, default=str)[:400].replace("`", "'") + "`")
PY
else
    say "- **No events file** at \`$EVENTS\`."
fi

if [ -n "$FAKE_REQUESTS" ]; then
    n=$(grep -c '"question"' "$FAKE_REQUESTS" 2>/dev/null); n=${n:-0}
    say ""; say "### Requests the agent sent to the fake MineLLM: $n"; say ""
    if [ "$n" -gt 0 ]; then
        say '```'; grep '"question"' "$FAKE_REQUESTS" | head -n 5 | cut -c1-400 >> "$SUMMARY"; say '```'
    fi
fi

# Show log tails whenever the episode did not end with success=true (including outcome=error,
# which run_agent.py reports with exit code 0).
if [ "$AGENT_RC" != 0 ] || ! grep -q '"event": "episode_end".*"success": true' "$EVENTS" 2>/dev/null; then
    say ""; say "### Last 40 lines of agent stdout/stderr (keys masked)"; say '```'
    tail -n 40 "$AGENT_STDOUT" 2>/dev/null | redact >> "$SUMMARY"; say '```'
    if [ -f "$OUT_DIR/logs/agent.log" ]; then
        say ""; say "### Last 20 lines of agent.log (keys masked)"; say '```'
        tail -n 20 "$OUT_DIR/logs/agent.log" | redact >> "$SUMMARY"; say '```'
    fi
fi

if [ "$AGENT_RC" = 0 ] && grep -q '"episode_end"' "$EVENTS" 2>/dev/null; then STATUS="COMPLETED (see outcome above)"
else STATUS="FAILED (agent exit code $AGENT_RC)"; fi
exit 0
