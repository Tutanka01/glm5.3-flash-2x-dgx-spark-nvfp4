#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE="${1:-}"
STARTED=0
READY=0

cleanup_failed_start() {
  local status=$?
  trap - EXIT
  if [ "$status" -ne 0 ] && [ "$STARTED" = "1" ] && [ "$READY" = "0" ]; then
    set +e
    glm53_warn "Bring-up failed before readiness; collecting logs and stopping both ranks"
    NODE_RANK=0 HEADLESS= glm53_compose logs --no-color --tail=160 "$GLM53_SERVICE" >&2
    glm53_worker_compose logs --no-color --tail=160 "$GLM53_SERVICE" >&2
    NODE_RANK=0 HEADLESS= glm53_compose stop "$GLM53_SERVICE" >/dev/null 2>&1
    glm53_worker_compose stop "$GLM53_SERVICE" >/dev/null 2>&1
    "$ROOT_DIR/scripts/stop-guard-node.sh" head >/dev/null 2>&1
    glm53_worker_script stop-guard-node.sh worker >/dev/null 2>&1
  fi
  exit "$status"
}
trap cleanup_failed_start EXIT

"$ROOT_DIR/scripts/validate-env.sh" "$PROFILE"
glm53_load_config "$PROFILE"

glm53_info "Connecting to worker and synchronizing the selected profile"
glm53_ssh true
glm53_sync_worker

if [ "${START_RUN_DOCTOR:-1}" = "1" ]; then
  "$ROOT_DIR/scripts/doctor-node.sh" head "$GLM53_PROFILE_RESOLVED"
  glm53_worker_script doctor-node.sh worker "$GLM53_PROFILE_RESOLVED"
fi

glm53_info "Verifying the pinned checkpoint offline on both nodes"
"$ROOT_DIR/scripts/checkpoint-node.sh" head local "$GLM53_PROFILE_RESOLVED"
glm53_worker_script checkpoint-node.sh worker local "$GLM53_PROFILE_RESOLVED"

if glm53_container_running_local; then
  glm53_die "Head service is already running; stop it before a profile change"
fi
if [ -n "$(glm53_worker_compose ps --status running -q "$GLM53_SERVICE")" ]; then
  glm53_die "Worker service is already running; stop it before a profile change"
fi

glm53_info "Starting worker first (rank 1, headless)"
STARTED=1
glm53_worker_compose up -d --no-build --force-recreate "$GLM53_SERVICE"
glm53_worker_script start-guard-node.sh worker "$GLM53_PROFILE_RESOLVED"
sleep 5
if [ -z "$(glm53_worker_compose ps --status running -q "$GLM53_SERVICE")" ]; then
  glm53_die "Worker container exited during initial startup"
fi

glm53_info "Starting head (rank 0, OpenAI-compatible API)"
NODE_RANK=0 HEADLESS= glm53_compose up -d --no-build --force-recreate "$GLM53_SERVICE"
"$ROOT_DIR/scripts/start-guard-node.sh" head "$GLM53_PROFILE_RESOLVED"

WAIT_TIMEOUT="${START_WAIT_TIMEOUT:-3600}"
WAIT_INTERVAL="${START_WAIT_INTERVAL:-15}"
START_TIME="$(date +%s)"
NEXT_PROGRESS=0
glm53_info "Waiting up to ${WAIT_TIMEOUT}s for http://127.0.0.1:${VLLM_PORT:-8888}/v1/models"

while :; do
  NOW="$(date +%s)"
  ELAPSED=$((NOW - START_TIME))
  if [ "$ELAPSED" -ge "$WAIT_TIMEOUT" ]; then
    glm53_die "Timed out waiting for vLLM readiness after ${ELAPSED}s"
  fi

  if ! glm53_container_running_local; then
    glm53_die "Head container exited before readiness"
  fi
  if [ -z "$(glm53_worker_compose ps --status running -q "$GLM53_SERVICE")" ]; then
    glm53_die "Worker container exited before readiness"
  fi

  if GLM53_CURL_MAX_TIME=5 glm53_api_curl "http://127.0.0.1:${VLLM_PORT:-8888}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi

  if [ "$ELAPSED" -ge "$NEXT_PROGRESS" ]; then
    glm53_info "Still loading (${ELAPSED}s elapsed); this 320B MoE can take many minutes"
    NEXT_PROGRESS=$((ELAPSED + 60))
  fi
  sleep "$WAIT_INTERVAL"
done

"$ROOT_DIR/scripts/stop-guard-node.sh" head || true
glm53_worker_script stop-guard-node.sh worker || true
glm53_info "API is ready with profile=$GLM53_PROFILE_RESOLVED"

if [ "${START_SMOKE:-1}" = "1" ]; then
  "$ROOT_DIR/smoke-glm53.sh" --profile "$GLM53_PROFILE_RESOLVED"
fi

trap - EXIT
printf '\nGLM-5.3-Flash is serving at http://%s:%s/v1\n' "$VLLM_HOST_IP" "${VLLM_PORT:-8888}"
