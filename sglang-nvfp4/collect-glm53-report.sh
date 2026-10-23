#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE=""
LABEL="failure"
TAIL_LINES=2000

usage() {
  printf 'Usage: %s [--profile NAME] [--label LABEL] [--tail N]\n' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) shift; PROFILE="${1:?--profile requires a value}" ;;
    --label) shift; LABEL="${1:?--label requires a value}" ;;
    --tail) shift; TAIL_LINES="${1:?--tail requires a value}" ;;
    -h|--help) usage; exit 0 ;;
    *) glm53_die "Unknown option: $1" ;;
  esac
  shift
done

case "$LABEL" in
  *[!A-Za-z0-9._-]*|"") glm53_die "--label contains unsupported characters" ;;
esac
case "$TAIL_LINES" in
  ''|*[!0-9]*) glm53_die "--tail must be an integer" ;;
esac

glm53_load_config "$PROFILE"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_ROOT="${GLM53_REPORT_ROOT:-$ROOT_DIR/results/diagnostics}"
REPORT_DIR="$REPORT_ROOT/${TIMESTAMP}-${GLM53_PROFILE_RESOLVED}-${LABEL}"
mkdir -p "$REPORT_DIR"

capture() {
  local target="$1"
  shift
  "$@" >"$REPORT_DIR/$target" 2>&1 || true
}

{
  printf 'collected_utc=%s\n' "$TIMESTAMP"
  printf 'profile=%s\n' "$GLM53_PROFILE_RESOLVED"
  printf 'profile_tier=%s\n' "$PROFILE_TIER"
  printf 'speculative_algorithm=%s\n' "$SPECULATIVE_ALGORITHM"
  printf 'mamba_ssm_dtype=%s\n' "$MAMBA_SSM_DTYPE"
  printf 'max_mamba_cache_size=%s\n' "$MAX_MAMBA_CACHE_SIZE"
  printf 'dflash_draft_window_size=%s\n' "$DFLASH_DRAFT_WINDOW_SIZE"
  printf 'runtime_image=%s\n' "$RUNTIME_IMAGE_EFFECTIVE"
  printf 'model_revision=%s\n' "$MODEL_REVISION"
} >"$REPORT_DIR/manifest.txt"

capture status.txt "$ROOT_DIR/status-glm53.sh" "$GLM53_PROFILE_RESOLVED"
capture logs-both.txt "$ROOT_DIR/logs-glm53.sh" \
  --profile "$GLM53_PROFILE_RESOLVED" --node both --tail "$TAIL_LINES"
capture docker-head-ps.txt glm53_compose ps -a
capture docker-worker-ps.txt glm53_worker_compose ps -a

HEAD_CONTAINER_ID="$(NODE_RANK=0 HEADLESS= glm53_compose ps -a -q "$GLM53_SERVICE" 2>/dev/null | head -n 1 || true)"
if [ -n "$HEAD_CONTAINER_ID" ]; then
  capture docker-head-state.json docker inspect \
    -f '{{json .State}}' "$HEAD_CONTAINER_ID"
  capture docker-head-image.txt docker inspect \
    -f 'image={{.Config.Image}} image_id={{.Image}} restart_count={{.RestartCount}}' \
    "$HEAD_CONTAINER_ID"
fi

WORKER_CONTAINER_ID="$(glm53_worker_compose ps -a -q "$GLM53_SERVICE" 2>/dev/null | head -n 1 || true)"
if [ -n "$WORKER_CONTAINER_ID" ]; then
  capture docker-worker-state.json glm53_ssh docker inspect \
    -f '{{json .State}}' "$WORKER_CONTAINER_ID"
  capture docker-worker-image.txt glm53_ssh docker inspect \
    -f 'image={{.Config.Image}} image_id={{.Image}} restart_count={{.RestartCount}}' \
    "$WORKER_CONTAINER_ID"
fi

capture host-head.txt bash -c '
  date -Is
  uname -a
  command -v free >/dev/null 2>&1 && free -h || true
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  command -v docker >/dev/null 2>&1 && docker stats --no-stream || true
  if command -v journalctl >/dev/null 2>&1; then
    journalctl -k --no-pager -n 500 2>/dev/null | tail -n 500
  elif command -v dmesg >/dev/null 2>&1; then
    dmesg 2>/dev/null | tail -n 500
  fi
'

REMOTE_DIAG='date -Is; uname -a; command -v free >/dev/null 2>&1 && free -h || true; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true; command -v docker >/dev/null 2>&1 && docker stats --no-stream || true; if command -v journalctl >/dev/null 2>&1; then journalctl -k --no-pager -n 500 2>/dev/null | tail -n 500; elif command -v dmesg >/dev/null 2>&1; then dmesg 2>/dev/null | tail -n 500; fi'
capture host-worker.txt glm53_ssh bash -lc "$(printf '%q' "$REMOTE_DIAG")"

ARCHIVE="$REPORT_DIR.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$REPORT_DIR")" "$(basename "$REPORT_DIR")"
glm53_info "Diagnostic archive ready: $ARCHIVE"
printf '%s\n' "$ARCHIVE"
