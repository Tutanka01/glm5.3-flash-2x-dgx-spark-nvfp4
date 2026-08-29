#!/usr/bin/env bash
# soak-day.sh — daily probe for the multi-day OpenCode soak (promotion item).
#
# Run on the HEAD node at any cadence (daily recommended). Prints a
# paste-ready markdown block for the soak journal in docs/SOAK.md:
# container uptime/restarts (head + worker), /health, error greps on the
# server logs, and a small concurrent tool-call probe (issue #10 watch).
#
#   vllm-exl3/scripts/soak-day.sh              # full block with tool probe
#   vllm-exl3/scripts/soak-day.sh --no-probe   # bookkeeping only
#
# Exit 0 = healthy day; 1 = unhealthy signals (see the block); 2 = usage error.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

log() { printf '\033[1;36m[soak-day]\033[0m %s\n' "$*" >&2; }

PROBE=1
case "${1:-}" in
    "") ;;
    --no-probe) PROBE=0 ;;
    *) log "unknown arg: $1"; exit 2 ;;
esac

# .env for PORT / WORKER_SSH (same precedence rules as start.sh: caller wins).
ENV_PORT="$(grep -E '^PORT=' "$LANE_DIR/.env" 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
ENV_WORKER_SSH="$(grep -E '^WORKER_SSH=' "$LANE_DIR/.env" 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
PORT="${PORT:-${ENV_PORT:-8888}}"
WORKER_SSH="${WORKER_SSH:-${ENV_WORKER_SSH:-}}"

mkdir -p "$LANE_DIR/logs" "$LANE_DIR/results"

oneline() { tr -d '\r\n'; }

head_uptime() {
    local out
    out="$(docker inspect glm53-exl3-head --format '{{.State.Status}} restarts={{.RestartCount}} started={{.State.StartedAt}}' 2>/dev/null | oneline)"
    [ -n "$out" ] || out="container not found"
    printf '%s' "$out"
}
worker_uptime() {
    local out=""
    if [ -n "$WORKER_SSH" ]; then
        out="$(ssh -o ConnectTimeout=5 "$WORKER_SSH" \
            "docker inspect glm53-exl3-worker --format '{{.State.Status}} restarts={{.RestartCount}} started={{.State.StartedAt}}'" 2>/dev/null | oneline)"
        [ -n "$out" ] || out="worker unreachable via WORKER_SSH"
    else
        out="skipped (no WORKER_SSH in env)"
    fi
    printf '%s' "$out"
}

HEALTH="$(curl -s -o /dev/null -w '%{http_code}' -m 10 "http://127.0.0.1:${PORT}/health" | oneline || true)"
[ -n "$HEALTH" ] || HEALTH=000
ERR_PATTERNS='Traceback|CUDA (out of memory|error)|NCCL|EngineDead|Engine core|retract'

count_errors() {
    local file="$1"
    if [ -f "$file" ]; then
        grep -c -E "$ERR_PATTERNS" "$file" 2>/dev/null || echo 0
    else
        echo "no-log"
    fi
}
last_errors() {
    local file="$1"
    [ -f "$file" ] && grep -E "$ERR_PATTERNS" "$file" 2>/dev/null | tail -n 3 || true
}

HEAD_ERR="$(count_errors "$LANE_DIR/logs/head.log")"
WORKER_ERR="$(count_errors "$LANE_DIR/logs/worker.log")"

PROBE_LINE="skipped"
PROBE_STATUS=0
if [ "$PROBE" -eq 1 ] && [ "$HEALTH" = "200" ]; then
    if python3 "$LANE_DIR/tests/soak_tool_calls.py" \
        --base-url "http://127.0.0.1:${PORT}/v1" \
        --agents 2 --turns 2 --filler-words 4000 \
        --out "$LANE_DIR/results/soak-day-probe.json" >"$LANE_DIR/logs/soak-day-probe.out" 2>&1; then
        PROBE_LINE="$(grep '^\[soak-tool-calls\] turns' "$LANE_DIR/logs/soak-day-probe.out" | tail -n1 || echo 'ran')"
    else
        PROBE_LINE="FAILED (see logs/soak-day-probe.out)"
        PROBE_STATUS=1
    fi
fi

UNHEALTHY=0
[ "$HEALTH" = "200" ] || UNHEALTHY=1
case "$HEAD_ERR" in ''|0|no-log) ;; *) UNHEALTHY=1 ;; esac
case "$WORKER_ERR" in ''|0|no-log) ;; *) UNHEALTHY=1 ;; esac
[ "$PROBE_STATUS" -eq 0 ] || UNHEALTHY=1

cat <<BLOCK
## Soak day — $(date -u +%Y-%m-%d\ %H:%M\ UTC)

| signal | value |
|---|---|
| head container | $(head_uptime) |
| worker container | $(worker_uptime) |
| /health | HTTP $HEALTH |
| error-line count head.log | $HEAD_ERR |
| error-line count worker.log | $WORKER_ERR |
| tool-call probe | $PROBE_LINE |
| verdict | $([ "$UNHEALTHY" -eq 0 ] && echo healthy || echo UNHEALTHY) |

$(last_errors "$LANE_DIR/logs/head.log")
$(last_errors "$LANE_DIR/logs/worker.log")
BLOCK

log "paste the block above into vllm-exl3/docs/SOAK.md (journal section)"
exit "$UNHEALTHY"
