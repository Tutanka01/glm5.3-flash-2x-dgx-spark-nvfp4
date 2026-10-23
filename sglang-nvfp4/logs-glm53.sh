#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE=""
NODE="both"
TAIL_LINES=300
FOLLOW=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) shift; PROFILE="${1:?--profile requires a value}" ;;
    --node) shift; NODE="${1:?--node requires head, worker, or both}" ;;
    --tail) shift; TAIL_LINES="${1:?--tail requires a value}" ;;
    -f|--follow) FOLLOW=1 ;;
    -h|--help)
      printf 'Usage: %s [--profile NAME] [--node head|worker|both] [--tail N] [-f]\n' "$0"
      exit 0
      ;;
    *) glm53_die "Unknown option: $1" ;;
  esac
  shift
done
case "$NODE" in head|worker|both) ;; *) glm53_die "--node must be head, worker, or both" ;; esac
case "$TAIL_LINES" in ''|*[!0-9]*) glm53_die "--tail must be an integer" ;; esac
[ "$FOLLOW" = "0" ] || [ "$NODE" != "both" ] || glm53_die "Follow one node at a time"

glm53_load_config "$PROFILE"
LOG_ARGS=(logs --no-color --tail="$TAIL_LINES")
[ "$FOLLOW" = "0" ] || LOG_ARGS+=(-f)
LOG_ARGS+=("$GLM53_SERVICE")

if [ "$NODE" = "head" ] || [ "$NODE" = "both" ]; then
  printf '\n===== HEAD / rank 0 =====\n'
  NODE_RANK=0 HEADLESS= glm53_compose "${LOG_ARGS[@]}" || LOG_RESULT=$?
fi
if [ "$NODE" = "worker" ] || [ "$NODE" = "both" ]; then
  printf '\n===== WORKER / rank 1 =====\n'
  glm53_worker_compose "${LOG_ARGS[@]}" || LOG_RESULT=$?
fi
exit "${LOG_RESULT:-0}"
