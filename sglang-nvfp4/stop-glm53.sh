#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE=""
DOWN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --down) DOWN=1 ;;
    --profile) shift; PROFILE="${1:?--profile requires a value}" ;;
    -h|--help) printf 'Usage: %s [--profile NAME] [--down]\n' "$0"; exit 0 ;;
    -*) glm53_die "Unknown option: $1" ;;
    *) PROFILE="$1" ;;
  esac
  shift
done
glm53_load_config "$PROFILE"

RESULT=0
glm53_info "Stopping head first"
if [ "$DOWN" = "1" ]; then
  NODE_RANK=0 HEADLESS= glm53_compose down --remove-orphans || RESULT=1
else
  NODE_RANK=0 HEADLESS= glm53_compose stop "$GLM53_SERVICE" || RESULT=1
fi
"$ROOT_DIR/scripts/stop-guard-node.sh" head || true

glm53_info "Stopping worker"
if glm53_ssh true; then
  if [ "$DOWN" = "1" ]; then
    glm53_worker_compose down --remove-orphans || RESULT=1
  else
    glm53_worker_compose stop "$GLM53_SERVICE" || RESULT=1
  fi
  glm53_worker_script stop-guard-node.sh worker || true
else
  glm53_warn "Worker is unreachable; its rank may still be running"
  RESULT=1
fi

[ "$RESULT" -eq 0 ] && glm53_info "Both ranks stopped"
exit "$RESULT"
