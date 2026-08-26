#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE="${1:-}"
"$ROOT_DIR/scripts/validate-env.sh" "$PROFILE" || exit $?
glm53_load_config "$PROFILE"

glm53_info "Checking passwordless SSH to $WORKER_HOST"
if ! glm53_ssh true; then
  glm53_die "Cannot reach worker with non-interactive SSH"
fi
glm53_sync_worker || glm53_die "Failed to synchronize doctor files to the worker"

RESULT=0
"$ROOT_DIR/scripts/doctor-node.sh" head "$GLM53_PROFILE_RESOLVED" || RESULT=1
glm53_worker_script doctor-node.sh worker "$GLM53_PROFILE_RESOLVED" || RESULT=1

if command -v ip >/dev/null 2>&1; then
  glm53_info "Head route to worker fabric IP"
  ip route get "$WORKER_VLLM_HOST_IP" || RESULT=1
fi
glm53_info "Worker route to head fabric IP"
glm53_ssh "ip route get $(glm53_shell_join "$VLLM_HOST_IP")" || RESULT=1

if [ "$RESULT" -ne 0 ]; then
  glm53_die "Doctor found blocking problems"
fi
glm53_info "Two-node doctor completed successfully"
