#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

ROLE="${1:-head}"
MODE="${2:-local}"
PROFILE="${3:-}"

case "$ROLE" in
  head|worker) ;;
  *) glm53_die "checkpoint-node role must be head or worker" ;;
esac
case "$MODE" in
  download|local) ;;
  *) glm53_die "checkpoint-node mode must be download or local" ;;
esac

glm53_load_config "$PROFILE"
glm53_require_command docker
glm53_require_command id
glm53_require_command mkdir

NODE_CACHE="$HF_CACHE"
if [ "$ROLE" = "worker" ]; then
  NODE_CACHE="$WORKER_HF_CACHE"
fi
[ -n "$NODE_CACHE" ] || glm53_die "Hugging Face cache path is empty for role=$ROLE"

mkdir -p \
  "$NODE_CACHE" \
  "$NODE_CACHE/tmp" \
  "$NODE_CACHE/vllm-cache" \
  "$NODE_CACHE/triton-cache" \
  "$NODE_CACHE/torch-extensions" \
  "$NODE_CACHE/cuda-cache"

if [ "$MODE" = "download" ]; then
  glm53_info "Pulling pinned runtime image on $ROLE: $GLM53_VLLM_IMAGE"
  docker pull "$GLM53_VLLM_IMAGE"
else
  docker image inspect "$GLM53_VLLM_IMAGE" >/dev/null 2>&1 || \
    glm53_die "Pinned image is not local on $ROLE. Run ./prepare-glm53.sh first."
fi

DOCKER_ARGS=(
  run --rm --pull never --network host
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp/glm53-user
  -e HF_HOME=/cache/huggingface
  -e XDG_CACHE_HOME=/cache/huggingface/xdg
  -e TMPDIR=/cache/huggingface/tmp
  -e "HF_HUB_OFFLINE=$([ "$MODE" = "local" ] && printf 1 || printf 0)"
  -e "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}"
  -e "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-1800}"
  -e "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}"
  -v "$NODE_CACHE:/cache/huggingface"
  -v "$GLM53_ROOT/scripts:/recipe/scripts:ro"
  -v "$GLM53_ROOT/metadata:/recipe/metadata:ro"
  --entrypoint python3
)
if [ -n "${HF_TOKEN:-}" ]; then
  DOCKER_ARGS+=(-e HF_TOKEN)
fi

PYTHON_ARGS=(
  /recipe/scripts/prepare_checkpoint.py
  --model-id "$MODEL_ID"
  --revision "$MODEL_REVISION"
  --manifest /recipe/metadata/checkpoint-manifest.json
  --max-workers "${HF_DOWNLOAD_WORKERS:-4}"
)
if [ "$MODE" = "local" ]; then
  PYTHON_ARGS+=(--local-only)
fi

glm53_info "$([ "$MODE" = "download" ] && printf 'Preparing' || printf 'Validating') checkpoint on $ROLE"
docker "${DOCKER_ARGS[@]}" "$GLM53_VLLM_IMAGE" "${PYTHON_ARGS[@]}"

