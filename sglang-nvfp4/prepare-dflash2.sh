#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE="${1:-128k-dflash2}"
"$ROOT_DIR/scripts/validate-env.sh" "$PROFILE"
glm53_load_config "$PROFILE"
[ "$SPECULATIVE_ALGORITHM" = "DFLASH" ] || glm53_die "$PROFILE is not a DFlash profile"

glm53_info "Synchronizing the DFlash2 build recipe to the worker"
glm53_ssh true
glm53_sync_worker

glm53_info "Ensuring the target checkpoint and base image exist on both nodes"
"$ROOT_DIR/scripts/checkpoint-node.sh" head download "$PROFILE" &
HEAD_TARGET_PID=$!
WORKER_TARGET=0
glm53_worker_script checkpoint-node.sh worker download "$PROFILE" || WORKER_TARGET=$?
wait "$HEAD_TARGET_PID"
[ "$WORKER_TARGET" -eq 0 ] || glm53_die "Worker target preparation failed"

glm53_info "Building the pinned SGLang DFlash2 image on both nodes"
"$ROOT_DIR/scripts/build-dflash2-node.sh" head "$PROFILE" &
HEAD_BUILD_PID=$!
WORKER_BUILD=0
glm53_worker_script build-dflash2-node.sh worker "$PROFILE" || WORKER_BUILD=$?
wait "$HEAD_BUILD_PID"
[ "$WORKER_BUILD" -eq 0 ] || glm53_die "Worker DFlash2 build failed"

glm53_info "Downloading and validating the pinned 1B DFlash2 draft on both nodes"
"$ROOT_DIR/scripts/prepare-dflash2-node.sh" head download "$PROFILE" &
HEAD_DRAFT_PID=$!
WORKER_DRAFT=0
glm53_worker_script prepare-dflash2-node.sh worker download "$PROFILE" || WORKER_DRAFT=$?
wait "$HEAD_DRAFT_PID"
[ "$WORKER_DRAFT" -eq 0 ] || glm53_die "Worker DFlash2 draft preparation failed"

glm53_info "DFlash2 preparation complete on both nodes"
printf 'Start with: ./start-glm53.sh %s\n' "$PROFILE"
