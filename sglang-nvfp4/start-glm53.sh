#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE=""
STARTED=0
READY=0

usage() {
  cat <<'EOF'
Usage: ./start-glm53.sh [profile]

All profiles start directly. Legacy --allow-candidate and
--allow-unsafe-profile options are accepted but no longer required.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --allow-candidate)
      : # Backward-compatible no-op: admission is no longer blocked.
      ;;
    --allow-unsafe-profile)
      : # Backward-compatible no-op: admission is no longer blocked.
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -* )
      glm53_die "Unknown start option: $1"
      ;;
    *)
      [ -z "$PROFILE" ] || glm53_die "Only one profile may be selected"
      PROFILE="$1"
      ;;
  esac
  shift
done

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

case "$PROFILE_TIER" in
  validated) ;;
  experimental)
    glm53_warn "Starting experimental profile '$GLM53_PROFILE_RESOLVED'; preserve both-rank logs if it fails"
    ;;
  capacity-candidate)
    glm53_warn "Starting unproven capacity candidate '$GLM53_PROFILE_RESOLVED'"
    ;;
  quarantined)
    glm53_warn "Starting quarantined profile '$GLM53_PROFILE_RESOLVED'; this lane has prior or expected crash risk"
    ;;
  *)
    glm53_die "Unknown PROFILE_TIER=$PROFILE_TIER"
    ;;
esac

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
glm53_info "Waiting up to ${WAIT_TIMEOUT}s for model and tokenizer readiness"

api_reports_ready() {
  local response tokenizer_response
  response="$(GLM53_CURL_MAX_TIME=30 glm53_api_curl \
    "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null)" || return 1
  python3 -c '
import json, sys
payload = json.load(sys.stdin)
expected = sys.argv[1]
ids = [item.get("id") for item in payload.get("data", [])]
raise SystemExit(0 if expected in ids else 1)
' "$SERVED_MODEL_NAME" <<< "$response" >/dev/null 2>&1 || return 1

  tokenizer_response="$(GLM53_CURL_MAX_TIME=30 glm53_api_curl \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"readiness probe\"}" \
    "http://127.0.0.1:${API_PORT}/v1/tokenize" 2>/dev/null)" || return 1
  python3 -c '
import json, sys
payload = json.load(sys.stdin)
valid = isinstance(payload.get("count"), int) or isinstance(payload.get("num_tokens"), int)
valid = valid or isinstance(payload.get("tokens"), list) or isinstance(payload.get("input_ids"), list)
raise SystemExit(0 if valid else 1)
' <<< "$tokenizer_response" >/dev/null 2>&1
}

while :; do
  NOW="$(date +%s)"
  ELAPSED=$((NOW - START_TIME))
  if [ "$ELAPSED" -ge "$WAIT_TIMEOUT" ]; then
    glm53_die "Timed out waiting for SGLang readiness after ${ELAPSED}s"
  fi

  if ! glm53_container_running_local; then
    glm53_die "Head container exited before readiness"
  fi
  if [ -z "$(glm53_worker_compose ps --status running -q "$GLM53_SERVICE")" ]; then
    glm53_die "Worker container exited before readiness"
  fi

  if api_reports_ready; then
    break
  fi

  if [ "$ELAPSED" -ge "$NEXT_PROGRESS" ]; then
    glm53_info "Still loading (${ELAPSED}s elapsed); this 320B MoE can take many minutes"
    NEXT_PROGRESS=$((ELAPSED + 60))
  fi
  sleep "$WAIT_INTERVAL"
done

glm53_info "Disarming startup-only memory guards before accepting traffic"
"$ROOT_DIR/scripts/stop-guard-node.sh" head || \
  glm53_die "Could not stop the head startup guard"
glm53_worker_script stop-guard-node.sh worker || \
  glm53_die "Could not stop the worker startup guard"
glm53_info "API is ready with profile=$GLM53_PROFILE_RESOLVED"

if [ "${START_SMOKE:-1}" = "1" ]; then
  "$ROOT_DIR/smoke-glm53.sh" --profile "$GLM53_PROFILE_RESOLVED"
fi

READY=1
trap - EXIT
printf '\nGLM-5.3-Flash is serving at http://%s:%s/v1\n' "$API_ADVERTISE_HOST" "$API_PORT"
