#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

ROLE="${1:-head}"
PID_FILE="$GLM53_ROOT/.glm53-guard-$ROLE.pid"
glm53_stop_guard_pid_file "$PID_FILE"
