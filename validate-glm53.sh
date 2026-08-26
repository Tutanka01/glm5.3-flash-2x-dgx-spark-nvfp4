#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE=""
CONFIG_ONLY=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config-only) CONFIG_ONLY=1 ;;
    --profile) shift; PROFILE="${1:?--profile requires a value}" ;;
    -h|--help)
      printf 'Usage: %s [--profile NAME] [--config-only]\n' "$0"
      exit 0
      ;;
    -*) glm53_die "Unknown option: $1" ;;
    *) [ -z "$PROFILE" ] || glm53_die "Only one profile may be specified"; PROFILE="$1" ;;
  esac
  shift
done

"$ROOT_DIR/scripts/validate-env.sh" "$PROFILE"
if [ "$CONFIG_ONLY" = "1" ]; then
  exit 0
fi

glm53_load_config "$PROFILE"
glm53_info "Checking SSH and syncing validation files"
glm53_ssh true
glm53_sync_worker

"$ROOT_DIR/scripts/checkpoint-node.sh" head local "$GLM53_PROFILE_RESOLVED"
glm53_worker_script checkpoint-node.sh worker local "$GLM53_PROFILE_RESOLVED"
glm53_info "Both nodes have a complete, audited checkpoint"

