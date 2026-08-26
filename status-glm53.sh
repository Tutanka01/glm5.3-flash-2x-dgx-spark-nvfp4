#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE="${1:-}"
glm53_load_config "$PROFILE"

printf 'Profile: %s\n' "$GLM53_PROFILE_RESOLVED"
printf 'Endpoint: http://%s:%s/v1\n' "$VLLM_HOST_IP" "${VLLM_PORT:-8888}"

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
if GLM53_CURL_MAX_TIME=10 glm53_api_curl "http://127.0.0.1:${VLLM_PORT:-8888}/v1/models" > "$STATUS_TMP" 2>/dev/null; then
  python3 - "$STATUS_TMP" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
for model in payload.get("data", []):
    print(f"ready model={model.get('id')} max_model_len={model.get('max_model_len', 'unknown')}")
PY
else
  printf 'API not ready\n'
fi

guard_log="$ROOT_DIR/.glm53-guard-head.log"
if [ -f "$guard_log" ]; then
  printf '\n===== HEAD guard (last 5 lines) =====\n'
  tail -n 5 "$guard_log"
fi
if glm53_ssh true; then
  printf '\n===== WORKER guard (last 5 lines) =====\n'
  glm53_ssh "tail -n 5 $(glm53_shell_join "$WORKER_DIR/.glm53-guard-worker.log") 2>/dev/null || true"
fi
