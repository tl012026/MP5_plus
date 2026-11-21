# shellcheck shell=bash
# MineLLM server helpers. Source this file from an sbatch job; do not execute it.
#
#   source "$MP5_ROOT/slurm/minellm_server.sh"
#   minellm_start            || exit 1   # launch in background, log to scratch
#   minellm_wait_ready       || exit 1   # TCP check on 127.0.0.1:$MINELLM_PORT
#   ...                                  # use $MINELLM_URL
#   minellm_stop                         # SIGTERM, wait 30 s, then SIGKILL
#
# The server is LAMM/src/mllm_api_mc_v1.5.py from IranQin/LAMM. It takes no
# command-line flags: port 25541, weights and paths are hard-coded relative to
# LAMM/src. See docs/minellm_service.md.
#
# Defaults point at wz2999's setup. Other users export these before sourcing:
#   MP5_ROOT          repo checkout that contains LAMM/ (git-ignored clone of
#                     IranQin/LAMM with patches/lamm_openlamm_optional_lightllm.patch
#                     applied and the weight symlinks from models.lock.json)
#   MINELLM_ENV       python env with torch 1.13.1+cu117, transformers 4.31.0
#                     (requirements/minellm_env.freeze.txt), run on the host, not in a container
#   MINELLM_LOG_DIR   where the server log goes (must be writable)
#   MINELLM_READY_TIMEOUT  seconds to wait for the port (default 1800)
#
# Sets: MINELLM_PID, MINELLM_LOG, MINELLM_URL, MINELLM_READY_SECONDS.
# Functions return non-zero on failure and never call exit, so the caller decides.
# Nothing here reads or prints API keys.

MP5_ROOT=${MP5_ROOT:-$HOME/MP5_plus}
MINELLM_ENV=${MINELLM_ENV:-/scratch/wz2999/env/minellm_env}
MINELLM_LOG_DIR=${MINELLM_LOG_DIR:-/scratch/wz2999/logs}
MINELLM_READY_TIMEOUT=${MINELLM_READY_TIMEOUT:-1800}
MINELLM_PORT=25541   # hard-coded in mllm_api_mc_v1.5.py; changing it here does not move the server
MINELLM_SRC=${MINELLM_SRC:-$MP5_ROOT/LAMM/src}
MINELLM_URL="http://127.0.0.1:${MINELLM_PORT}/image"
MINELLM_PID=${MINELLM_PID:-}
MINELLM_LOG=${MINELLM_LOG:-}
MINELLM_READY_SECONDS=${MINELLM_READY_SECONDS:-}
_MINELLM_T0=""

_minellm_msg() { echo "[minellm $(date '+%F %T')] $*" >&2; }

# Exit status 0 iff something accepts TCP connections on 127.0.0.1:$MINELLM_PORT.
_minellm_port_open() {
    "$MINELLM_ENV/bin/python" -I -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',$MINELLM_PORT)) == 0 else 1)" 2>/dev/null
}

# Start the server in its own process group. Returns 1 if a prerequisite is missing.
minellm_start() {
    local f
    for f in "$MINELLM_SRC/mllm_api_mc_v1.5.py" "$MINELLM_ENV/bin/python" \
             "$MINELLM_SRC/../ckpt/Mine_52k_Vicuna_13b_v1.5_epoch_3_mlp2x_gelu/pytorch_model.pt" \
             "$MINELLM_SRC/../model_zoo/vicuna_ckpt/13b_v1.5/config.json" \
             "$MINELLM_SRC/../model_zoo/mineclip_ckpt/mineclip_image_encoder_vit-B_196tokens.pth"; do
        [ -r "$f" ] || { _minellm_msg "missing or unreadable: $f"; return 1; }
    done
    if [ -n "$MINELLM_PID" ] && kill -0 "$MINELLM_PID" 2>/dev/null; then
        _minellm_msg "already running (pid $MINELLM_PID)"; return 1
    fi
    if _minellm_port_open; then
        _minellm_msg "port $MINELLM_PORT is already in use on this node"; return 1
    fi
    mkdir -p "$MINELLM_LOG_DIR" || return 1
    MINELLM_LOG="$MINELLM_LOG_DIR/minellm_server_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"
    _MINELLM_T0=$(date +%s)
    # cd in a subshell so the caller's cwd is unchanged. setsid makes the server
    # a process-group leader (pgid == pid) so minellm_stop can signal the group.
    MINELLM_PID=$(
        cd "$MINELLM_SRC" || exit 1
        PATH="$MINELLM_ENV/bin:$PATH" PYTHONUNBUFFERED=1 \
            setsid nohup python mllm_api_mc_v1.5.py > "$MINELLM_LOG" 2>&1 < /dev/null &
        echo $!
    ) || { _minellm_msg "cannot cd to $MINELLM_SRC"; return 1; }
    _minellm_msg "started pid $MINELLM_PID, log $MINELLM_LOG"
}

# Wait until 127.0.0.1:$MINELLM_PORT accepts a TCP connection.
# Do not use `curl -sf` on / or /image: they return 404/405, so -f never succeeds.
minellm_wait_ready() {
    [ -n "$MINELLM_PID" ] || { _minellm_msg "not started"; return 1; }
    local t0=${_MINELLM_T0:-$(date +%s)} now
    while :; do
        if _minellm_port_open; then
            MINELLM_READY_SECONDS=$(( $(date +%s) - t0 ))
            _minellm_msg "ready after ${MINELLM_READY_SECONDS}s at $MINELLM_URL"
            if grep -q "while .* not found" "$MINELLM_LOG" 2>/dev/null; then
                _minellm_msg "WARNING: delta checkpoint not found; server is plain Vicuna"
            fi
            return 0
        fi
        if ! kill -0 "$MINELLM_PID" 2>/dev/null; then
            _minellm_msg "server exited before the port opened; last log lines:"
            tail -n 40 "$MINELLM_LOG" >&2
            return 1
        fi
        now=$(date +%s)
        if [ $(( now - t0 )) -ge "$MINELLM_READY_TIMEOUT" ]; then
            _minellm_msg "port $MINELLM_PORT not reachable after $(( now - t0 ))s; last log lines:"
            tail -n 40 "$MINELLM_LOG" >&2
            return 1
        fi
        sleep 5
    done
}

# Stop the server: SIGTERM to its process group, wait up to 30 s, then SIGKILL.
# Safe to call more than once (e.g. from an EXIT trap).
minellm_stop() {
    [ -n "$MINELLM_PID" ] || return 0
    local i
    if kill -0 "$MINELLM_PID" 2>/dev/null; then
        _minellm_msg "stopping pid $MINELLM_PID"
        kill -TERM -- "-$MINELLM_PID" 2>/dev/null || kill -TERM "$MINELLM_PID" 2>/dev/null
        for i in $(seq 1 30); do
            kill -0 "$MINELLM_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$MINELLM_PID" 2>/dev/null; then
            _minellm_msg "still alive after 30s; sending SIGKILL"
            kill -KILL -- "-$MINELLM_PID" 2>/dev/null || kill -KILL "$MINELLM_PID" 2>/dev/null
            sleep 1
        fi
    fi
    if kill -0 "$MINELLM_PID" 2>/dev/null; then
        _minellm_msg "WARNING: pid $MINELLM_PID still alive"; return 1
    fi
    _minellm_msg "stopped"
    MINELLM_PID=""
}
