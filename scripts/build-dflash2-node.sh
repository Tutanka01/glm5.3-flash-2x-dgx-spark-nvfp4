#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

ROLE="${1:-head}"
PROFILE="${2:-128k-dflash2}"
case "$ROLE" in head|worker) ;; *) glm53_die "role must be head or worker" ;; esac
glm53_load_config "$PROFILE"
[ "$SPECULATIVE_ALGORITHM" = "DFLASH" ] || glm53_die "$PROFILE is not a DFlash profile"
docker image inspect "$GLM53_RUNTIME_IMAGE" >/dev/null 2>&1 || \
  glm53_die "Base SGLang image is missing on $ROLE"

glm53_info "Building $RUNTIME_IMAGE_EFFECTIVE on $ROLE from pinned SGLang/SM121 revisions"
docker build \
  --pull=false \
  --file "$GLM53_ROOT/dflash2/sglang/Dockerfile" \
  --tag "$RUNTIME_IMAGE_EFFECTIVE" \
  "$GLM53_ROOT/dflash2/sglang"

"$GLM53_ROOT/scripts/check-dflash2-runtime.sh" --image "$RUNTIME_IMAGE_EFFECTIVE"
