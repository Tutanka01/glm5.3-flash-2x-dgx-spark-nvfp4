#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

ROLE="${1:-head}"
GUARD_STATE_DIR="${GLM53_GUARD_STATE_DIR:-$GLM53_ROOT}"
PID_FILE="$GUARD_STATE_DIR/.glm53-guard-$ROLE.pid"
glm53_stop_guard_pid_file "$PID_FILE"
