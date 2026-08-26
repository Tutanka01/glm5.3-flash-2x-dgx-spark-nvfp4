#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

ROLE="${1:-head}"
PROFILE="${2:-}"
case "$ROLE" in head|worker) ;; *) glm53_die "start-guard-node role must be head or worker" ;; esac
glm53_load_config "$PROFILE"

if [ "${OOM_GUARD:-1}" != "1" ]; then
  glm53_info "OOM guard disabled on $ROLE"
  exit 0
fi

PID_FILE="$GLM53_ROOT/.glm53-guard-$ROLE.pid"
LOG_FILE="$GLM53_ROOT/.glm53-guard-$ROLE.log"
glm53_stop_guard_pid_file "$PID_FILE"

nohup env \
  GLM53_PROJECT_NAME="$GLM53_PROJECT_NAME" \
  GLM53_SERVICE="$GLM53_SERVICE" \
  GLM53_GUARD_HEALTH_URL="http://$MASTER_ADDR:${VLLM_PORT:-8888}/health" \
  OOM_GUARD_MIN_AVAILABLE_MB="${OOM_GUARD_MIN_AVAILABLE_MB:-6144}" \
  OOM_GUARD_TIMEOUT="${OOM_GUARD_TIMEOUT:-3600}" \
  "$SCRIPT_DIR/oom-guard-node.sh" >"$LOG_FILE" 2>&1 </dev/null &
GUARD_PID=$!
printf '%s\n' "$GUARD_PID" > "$PID_FILE"
glm53_info "OOM guard started on $ROLE (pid=$GUARD_PID, log=$LOG_FILE)"
