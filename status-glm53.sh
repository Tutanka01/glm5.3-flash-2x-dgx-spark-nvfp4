#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

REQUESTED_PROFILE="${1:-}"
glm53_load_config "$REQUESTED_PROFILE"
CONFIGURED_PROFILE="$GLM53_PROFILE_RESOLVED"

HEAD_CONTAINER_ID="$(glm53_container_id_local)"
HEAD_CONTAINER_ENV=""
RUNNING_PROFILE=""
RUNNING_MAX_MODEL_LEN=""
RUNNING_MAX_NUM_SEQS=""
RUNNING_DISABLE_CUDA_GRAPH=""
RUNNING_MTP_NUM_TOKENS=""
if [ -n "$HEAD_CONTAINER_ID" ]; then
  HEAD_CONTAINER_ENV="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' \
    "$HEAD_CONTAINER_ID" 2>/dev/null || true)"
  RUNNING_PROFILE="$(printf '%s\n' "$HEAD_CONTAINER_ENV" | sed -n 's/^GLM53_PROFILE_NAME=//p' | head -n 1)"
  RUNNING_MAX_MODEL_LEN="$(printf '%s\n' "$HEAD_CONTAINER_ENV" | sed -n 's/^MAX_MODEL_LEN=//p' | head -n 1)"
  RUNNING_MAX_NUM_SEQS="$(printf '%s\n' "$HEAD_CONTAINER_ENV" | sed -n 's/^MAX_NUM_SEQS=//p' | head -n 1)"
  RUNNING_DISABLE_CUDA_GRAPH="$(printf '%s\n' "$HEAD_CONTAINER_ENV" | sed -n 's/^DISABLE_CUDA_GRAPH=//p' | head -n 1)"
  RUNNING_MTP_NUM_TOKENS="$(printf '%s\n' "$HEAD_CONTAINER_ENV" | sed -n 's/^MTP_NUM_TOKENS=//p' | head -n 1)"
fi

if [ -n "$RUNNING_PROFILE" ] && [ "$RUNNING_PROFILE" != "unknown" ]; then
  if [ -z "$REQUESTED_PROFILE" ] && [ "$RUNNING_PROFILE" != "$CONFIGURED_PROFILE" ]; then
    glm53_load_config "$RUNNING_PROFILE"
  elif [ -n "$REQUESTED_PROFILE" ] && [ "$RUNNING_PROFILE" != "$CONFIGURED_PROFILE" ]; then
    glm53_warn "Requested profile=$CONFIGURED_PROFILE but running profile=$RUNNING_PROFILE"
  fi
  printf 'Profile: %s (running)\n' "$RUNNING_PROFILE"
else
  printf 'Profile configuration: %s\n' "$CONFIGURED_PROFILE"
  if [ -n "$HEAD_CONTAINER_ID" ]; then
    printf 'Running profile: unknown (container predates profile metadata)\n'
  fi
fi
if [ -n "$RUNNING_MAX_MODEL_LEN" ]; then
  printf 'Runtime: context=%s requests=%s graphs-disabled=%s MTP=%s\n' \
    "$RUNNING_MAX_MODEL_LEN" "${RUNNING_MAX_NUM_SEQS:-unknown}" \
    "${RUNNING_DISABLE_CUDA_GRAPH:-unknown}" "${RUNNING_MTP_NUM_TOKENS:-unknown}"
fi
printf 'Endpoint: http://%s:%s/v1\n' "$API_ADVERTISE_HOST" "$API_PORT"

printf '\n===== HEAD / rank 0 =====\n'
NODE_RANK=0 HEADLESS= glm53_compose ps || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || true
fi

printf '\n===== WORKER / rank 1 =====\n'
if glm53_ssh true; then
  glm53_worker_compose ps || true
  glm53_ssh "nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null || true
else
  printf 'worker unreachable: %s\n' "$WORKER_HOST" >&2
fi

printf '\n===== API =====\n'
STATUS_TMP="$(mktemp /tmp/glm53-status.XXXXXX)"
trap 'rm -f "$STATUS_TMP"' EXIT
EXPECTED_CONTEXT="${RUNNING_MAX_MODEL_LEN:-$MAX_MODEL_LEN}"
if GLM53_CURL_MAX_TIME=30 glm53_api_curl "http://127.0.0.1:${API_PORT}/v1/models" > "$STATUS_TMP" 2>/dev/null; then
  python3 - "$STATUS_TMP" "$EXPECTED_CONTEXT" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
expected=int(sys.argv[2])
for model in payload.get("data", []):
    actual=model.get("max_model_len")
    print(f"ready model={model.get('id')} max_model_len={actual or 'unknown'}")
    if isinstance(actual, int) and actual != expected:
        print(
            f"WARNING: API context={actual} differs from status configuration={expected}",
            file=sys.stderr,
        )
PY
else
  printf 'API not ready\n'
fi

TOKENIZER_TMP="$(mktemp /tmp/glm53-tokenizer-status.XXXXXX)"
trap 'rm -f "$STATUS_TMP" "$TOKENIZER_TMP"' EXIT
if GLM53_CURL_MAX_TIME=30 glm53_api_curl \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"status probe\"}" \
  "http://127.0.0.1:${API_PORT}/v1/tokenize" > "$TOKENIZER_TMP" 2>/dev/null; then
  printf 'tokenizer ready\n'
else
  printf 'tokenizer not ready (the HTTP front end may be loading or shutting down)\n'
fi

STATUS_RESULT=0
guard_pid_file="$ROOT_DIR/.glm53-guard-head.pid"
HEAD_GUARD_ACTIVE=0
if [ -s "$guard_pid_file" ]; then
  guard_pid="$(sed -n '1p' "$guard_pid_file" 2>/dev/null || true)"
  case "$guard_pid" in
    ''|*[!0-9]*) ;;
    *)
      if kill -0 "$guard_pid" 2>/dev/null; then
        HEAD_GUARD_ACTIVE=1
        printf '\nWARNING: head startup memory guard is ACTIVE (pid=%s).\n' "$guard_pid" >&2
        printf 'Do not benchmark: finish start-glm53.sh or run ./scripts/stop-guard-node.sh head.\n' >&2
        STATUS_RESULT=1
      fi
      ;;
  esac
fi
if [ "$HEAD_GUARD_ACTIVE" = "0" ]; then
  printf '\nHead startup memory guard: inactive (expected after readiness)\n'
fi

guard_log="$ROOT_DIR/.glm53-guard-head.log"
if [ -f "$guard_log" ]; then
  printf '\n===== HEAD startup guard log (historical, last 5 lines) =====\n'
  tail -n 5 "$guard_log"
fi
if glm53_ssh true; then
  printf '\n===== WORKER startup guard log (historical, last 5 lines) =====\n'
  glm53_ssh "tail -n 5 $(glm53_shell_join "$WORKER_DIR/.glm53-guard-worker.log") 2>/dev/null || true"
fi
exit "$STATUS_RESULT"
