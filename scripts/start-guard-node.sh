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

GUARD_STATE_DIR="${GLM53_GUARD_STATE_DIR:-$GLM53_ROOT}"
PID_FILE="$GUARD_STATE_DIR/.glm53-guard-$ROLE.pid"
LOG_FILE="$GUARD_STATE_DIR/.glm53-guard-$ROLE.log"
if [ "$ROLE" = "head" ]; then
  # The head guard runs on the API host itself. Using MASTER_ADDR here can fail
  # on hosts without local hairpin routing/firewall allowance and leave the
  # startup-only guard armed after readiness.
  GUARD_HEALTH_HOST=127.0.0.1
else
  GUARD_HEALTH_HOST="$MASTER_ADDR"
fi
glm53_stop_guard_pid_file "$PID_FILE"

nohup env \
  GLM53_PROJECT_NAME="$GLM53_PROJECT_NAME" \
  GLM53_SERVICE="$GLM53_SERVICE" \
  GLM53_GUARD_HEALTH_URL="http://$GUARD_HEALTH_HOST:${API_PORT}/v1/models" \
  GLM53_GUARD_API_KEY="${API_KEY:-}" \
  OOM_GUARD_MIN_AVAILABLE_MB="${OOM_GUARD_MIN_AVAILABLE_MB:-6144}" \
  OOM_GUARD_TIMEOUT="${OOM_GUARD_TIMEOUT:-3600}" \
  "$SCRIPT_DIR/oom-guard-node.sh" >"$LOG_FILE" 2>&1 </dev/null &
GUARD_PID=$!
printf '%s\n' "$GUARD_PID" > "$PID_FILE"
glm53_info "OOM guard started on $ROLE (pid=$GUARD_PID, log=$LOG_FILE)"
