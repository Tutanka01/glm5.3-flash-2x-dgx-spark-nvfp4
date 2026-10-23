#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE="${1:-}"
"$ROOT_DIR/scripts/validate-env.sh" "$PROFILE"
glm53_load_config "$PROFILE"

if [ "${PREPARE_CHECK_UPSTREAM:-1}" = "1" ]; then
  "$ROOT_DIR/check-upstream-glm53.sh" "$GLM53_PROFILE_RESOLVED"
fi

glm53_info "Checking worker SSH and copying the recipe"
glm53_ssh true
glm53_sync_worker

if [ "${PREPARE_RUN_DOCTOR:-1}" = "1" ]; then
  "$ROOT_DIR/scripts/doctor-node.sh" head "$GLM53_PROFILE_RESOLVED"
  glm53_worker_script doctor-node.sh worker "$GLM53_PROFILE_RESOLVED"
fi

glm53_info "Each node will download and retain the full ~181.3 GiB snapshot"
if [ "${PREPARE_PARALLEL:-0}" = "1" ]; then
  "$ROOT_DIR/scripts/checkpoint-node.sh" head download "$GLM53_PROFILE_RESOLVED" &
  HEAD_PREPARE_PID=$!
  WORKER_RESULT=0
  glm53_worker_script checkpoint-node.sh worker download "$GLM53_PROFILE_RESOLVED" || WORKER_RESULT=$?
  HEAD_RESULT=0
  wait "$HEAD_PREPARE_PID" || HEAD_RESULT=$?
  [ "$HEAD_RESULT" -eq 0 ] || glm53_die "Head checkpoint preparation failed"
  [ "$WORKER_RESULT" -eq 0 ] || glm53_die "Worker checkpoint preparation failed"
else
  "$ROOT_DIR/scripts/checkpoint-node.sh" head download "$GLM53_PROFILE_RESOLVED"
  glm53_worker_script checkpoint-node.sh worker download "$GLM53_PROFILE_RESOLVED"
fi

glm53_info "Preparation complete: pinned image and checkpoint validated on both nodes"
