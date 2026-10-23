#!/usr/bin/env bash
set -uo pipefail

PROJECT_NAME="${GLM53_PROJECT_NAME:-glm53}"
SERVICE_NAME="${GLM53_SERVICE:-sglang-glm53}"
HEALTH_URL="${GLM53_GUARD_HEALTH_URL:?GLM53_GUARD_HEALTH_URL is required}"
GUARD_API_KEY="${GLM53_GUARD_API_KEY:-}"
GUARD_MODEL="${GLM53_GUARD_MODEL:-}"
MIN_AVAILABLE_MB="${OOM_GUARD_MIN_AVAILABLE_MB:-6144}"
TIMEOUT_SECONDS="${OOM_GUARD_TIMEOUT:-3600}"
INTERVAL_SECONDS="${OOM_GUARD_INTERVAL:-5}"
GUARD_CURL_MAX_TIME="${GLM53_GUARD_CURL_MAX_TIME:-10}"

curl_args=(-fsS --noproxy '*' --connect-timeout 2 --max-time "$GUARD_CURL_MAX_TIME")
if [ -n "$GUARD_API_KEY" ]; then
  curl_args+=(-H "Authorization: Bearer $GUARD_API_KEY")
fi

started_at="$(date +%s)"
printf '%s guard started threshold=%sMiB health=%s\n' "$(date -u +%FT%TZ)" "$MIN_AVAILABLE_MB" "$HEALTH_URL"

while :; do
  health_args=("${curl_args[@]}")
  if [ -n "$GUARD_MODEL" ]; then
    health_args+=(
      -H 'Content-Type: application/json'
      -d "{\"model\":\"$GUARD_MODEL\",\"prompt\":\"guard probe\"}"
    )
  fi
  if curl "${health_args[@]}" "$HEALTH_URL" >/dev/null 2>&1; then
    printf '%s API healthy; guard complete\n' "$(date -u +%FT%TZ)"
    exit 0
  fi

  container_id="$(docker ps -q \
    --filter "label=com.docker.compose.project=$PROJECT_NAME" \
    --filter "label=com.docker.compose.service=$SERVICE_NAME" 2>/dev/null | head -n 1)"
  if [ -n "$container_id" ] && [ -r /proc/meminfo ]; then
    available_mb="$(awk '/^MemAvailable:/ {printf "%d", $2 / 1024}' /proc/meminfo)"
    printf '%s available=%sMiB container=%s\n' "$(date -u +%FT%TZ)" "${available_mb:-0}" "$container_id"
    if [ "${available_mb:-0}" -lt "$MIN_AVAILABLE_MB" ]; then
      printf '%s GUARD TRIP: stopping exact container %s\n' "$(date -u +%FT%TZ)" "$container_id" >&2
      docker stop --time 10 "$container_id"
      exit 2
    fi
  fi

  now="$(date +%s)"
  if [ $((now - started_at)) -ge "$TIMEOUT_SECONDS" ]; then
    printf '%s guard timeout; start script owns final cleanup\n' "$(date -u +%FT%TZ)" >&2
    exit 3
  fi
  sleep "$INTERVAL_SECONDS"
done
