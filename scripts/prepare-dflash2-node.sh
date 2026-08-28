#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

ROLE="${1:-head}"
MODE="${2:-download}"
PROFILE="${3:-128k-dflash2}"
case "$ROLE" in head|worker) ;; *) glm53_die "role must be head or worker" ;; esac
case "$MODE" in download|local) ;; *) glm53_die "mode must be download or local" ;; esac
glm53_load_config "$PROFILE"
[ "$SPECULATIVE_ALGORITHM" = "DFLASH" ] || glm53_die "$PROFILE is not a DFlash profile"

NODE_CACHE="$HF_CACHE"
[ "$ROLE" = "head" ] || NODE_CACHE="$WORKER_HF_CACHE"
mkdir -p "$NODE_CACHE" "$NODE_CACHE/tmp"

DOCKER_ARGS=(
  run --rm --pull never --network host
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp/glm53-dflash-user
  -e HF_HOME=/cache/huggingface
  -e TMPDIR=/cache/huggingface/tmp
  -e "HF_HUB_OFFLINE=$([ "$MODE" = local ] && printf 1 || printf 0)"
  -e "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}"
  -e "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-1800}"
  -e "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}"
  -v "$NODE_CACHE:/cache/huggingface"
  -v "$GLM53_ROOT/scripts:/recipe/scripts:ro"
  --entrypoint python3
)
if [ -n "${HF_TOKEN:-}" ]; then DOCKER_ARGS+=(-e HF_TOKEN); fi

PYTHON_ARGS=(
  /recipe/scripts/prepare_dflash2.py
  --model-id incoai/GLM-5.3-Flash-DFlash2
  --revision 7d74cdd881ed7e32c31175984a67823127b66cfe
  --max-workers "${HF_DOWNLOAD_WORKERS:-4}"
)
[ "$MODE" = download ] || PYTHON_ARGS+=(--local-only)

docker image inspect "$GLM53_RUNTIME_IMAGE" >/dev/null 2>&1 || \
  glm53_die "Base SGLang image is missing; run ./prepare-glm53.sh first"
docker "${DOCKER_ARGS[@]}" "$GLM53_RUNTIME_IMAGE" "${PYTHON_ARGS[@]}"
