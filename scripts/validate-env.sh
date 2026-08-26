#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

PROFILE="${1:-}"
glm53_load_config "$PROFILE"

glm53_require_command docker
glm53_require_command python3
glm53_require_command ssh
glm53_require_command scp
glm53_require_command curl
docker compose version >/dev/null 2>&1 || glm53_die "Docker Compose v2 is required"

REQUIRED_VARS=(
  WORKER_HOST WORKER_DIR MASTER_ADDR MASTER_PORT
  VLLM_HOST_IP WORKER_VLLM_HOST_IP
  NCCL_IB_HCA NCCL_SOCKET_IFNAME TP_SOCKET_IFNAME GLOO_SOCKET_IFNAME
  NCCL_IB_GID_INDEX HF_CACHE MODEL_ID MODEL_REVISION GLM53_VLLM_IMAGE
)
for required_name in "${REQUIRED_VARS[@]}"; do
  required_value="${!required_name:-}"
  if glm53_is_placeholder "$required_value"; then
    glm53_die "$required_name is missing or still contains an example placeholder"
  fi
done
glm53_validate_remote_paths

if [[ ! "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  glm53_die "MODEL_REVISION must be a lowercase 40-character commit SHA"
fi

MANIFEST_VALUES="$(python3 -c '
import json, sys
m=json.load(open(sys.argv[1], encoding="utf-8"))
print(m["model"]["id"])
print(m["model"]["revision"])
print(m["runtime"]["image_tag"] + "@" + m["runtime"]["image_digest"])
' "$GLM53_ROOT/metadata/checkpoint-manifest.json")"
EXPECTED_MODEL_ID="$(printf '%s\n' "$MANIFEST_VALUES" | sed -n '1p')"
EXPECTED_REVISION="$(printf '%s\n' "$MANIFEST_VALUES" | sed -n '2p')"
EXPECTED_IMAGE="$(printf '%s\n' "$MANIFEST_VALUES" | sed -n '3p')"
[ "$MODEL_ID" = "$EXPECTED_MODEL_ID" ] || glm53_die "MODEL_ID differs from the audited manifest"
[ "$MODEL_REVISION" = "$EXPECTED_REVISION" ] || glm53_die "MODEL_REVISION differs from the audited manifest"
[ "$GLM53_VLLM_IMAGE" = "$EXPECTED_IMAGE" ] || glm53_die "GLM53_VLLM_IMAGE differs from the audited manifest"

python3 - "$MASTER_ADDR" "$VLLM_HOST_IP" "$WORKER_VLLM_HOST_IP" <<'PY'
import ipaddress
import sys
for value in sys.argv[1:]:
    try:
        ipaddress.ip_address(value)
    except ValueError as exc:
        raise SystemExit(f"invalid fabric IP {value!r}: {exc}")
if sys.argv[2] == sys.argv[3]:
    raise SystemExit("VLLM_HOST_IP and WORKER_VLLM_HOST_IP must differ")
PY

case "$MASTER_PORT" in ''|*[!0-9]*) glm53_die "MASTER_PORT must be an integer" ;; esac
case "${VLLM_PORT:-8888}" in ''|*[!0-9]*) glm53_die "VLLM_PORT must be an integer" ;; esac
[ "$MASTER_PORT" -ge 1 ] && [ "$MASTER_PORT" -le 65535 ] || glm53_die "MASTER_PORT is out of range"
[ "${VLLM_PORT:-8888}" -ge 1 ] && [ "${VLLM_PORT:-8888}" -le 65535 ] || glm53_die "VLLM_PORT is out of range"
case "$NCCL_IB_GID_INDEX" in ''|*[!0-9]*) glm53_die "NCCL_IB_GID_INDEX must be a decimal integer" ;; esac
if [ -n "${WORKER_NCCL_IB_GID_INDEX:-}" ]; then
  case "$WORKER_NCCL_IB_GID_INDEX" in ''|*[!0-9]*) glm53_die "WORKER_NCCL_IB_GID_INDEX must be a decimal integer" ;; esac
fi

for integer_name in MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MTP_NUM_TOKENS HF_DOWNLOAD_WORKERS; do
  integer_value="${!integer_name:-}"
  case "$integer_value" in ''|*[!0-9]*) glm53_die "$integer_name must be a non-negative integer" ;; esac
done
[ "$MAX_MODEL_LEN" -ge 4096 ] || glm53_die "MAX_MODEL_LEN must be at least 4096"
[ "$MAX_MODEL_LEN" -le 1048576 ] || glm53_die "MAX_MODEL_LEN exceeds the checkpoint limit"
[ "$MAX_NUM_SEQS" -ge 1 ] || glm53_die "MAX_NUM_SEQS must be at least 1"
[ "$MAX_NUM_BATCHED_TOKENS" -ge 512 ] || glm53_die "MAX_NUM_BATCHED_TOKENS must be at least 512"
[ "$MTP_NUM_TOKENS" -le 16 ] || glm53_die "MTP_NUM_TOKENS above 16 is rejected by this recipe"
[ "$HF_DOWNLOAD_WORKERS" -ge 1 ] && [ "$HF_DOWNLOAD_WORKERS" -le 16 ] || \
  glm53_die "HF_DOWNLOAD_WORKERS must be between 1 and 16"

for bool_name in ENFORCE_EAGER ENABLE_CHUNKED_PREFILL ENABLE_PREFIX_CACHING; do
  bool_value="${!bool_name:-}"
  case "$bool_value" in 0|1) ;; *) glm53_die "$bool_name must be 0 or 1" ;; esac
done
[ "$ENABLE_CHUNKED_PREFILL" = "1" ] || {
  [ "$MAX_NUM_BATCHED_TOKENS" -ge "$MAX_MODEL_LEN" ] || \
    glm53_die "chunked prefill is required when MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN"
}
if [ "$MOE_BACKEND" = "marlin" ] && [ "$ENFORCE_EAGER" != "1" ]; then
  glm53_die "Marlin on sm_121 requires ENFORCE_EAGER=1 in this recipe"
fi
case "$MOE_BACKEND" in marlin|auto) ;; *) glm53_die "MOE_BACKEND must be marlin or auto" ;; esac
for switch_value in "${OOM_GUARD:-1}" "${START_SMOKE:-1}" "${REQUIRE_SWAP_OFF:-0}" "${ALLOW_UNSUPPORTED_PLATFORM:-0}"; do
  case "$switch_value" in 0|1) ;; *) glm53_die "Boolean recipe switches must be 0 or 1" ;; esac
done

python3 - "$GPU_MEMORY_UTILIZATION" <<'PY'
import sys
try:
    value=float(sys.argv[1])
except ValueError:
    raise SystemExit("GPU_MEMORY_UTILIZATION must be a number")
if not 0.5 <= value <= 0.95:
    raise SystemExit("GPU_MEMORY_UTILIZATION must be between 0.5 and 0.95")
PY

case "${CONTAINER_MEMORY_LIMIT:-112g}" in
  [1-9][0-9]g|1[01][0-9]g|12[0-7]g) ;;
  *) glm53_die "CONTAINER_MEMORY_LIMIT must be an integer between 10g and 127g" ;;
esac

if [ "$MASTER_ADDR" != "$VLLM_HOST_IP" ]; then
  glm53_warn "MASTER_ADDR differs from VLLM_HOST_IP; this is valid only when both route over the intended fabric"
fi
if [ "${HF_HUB_OFFLINE:-1}" != "1" ] || [ "${TRANSFORMERS_OFFLINE:-1}" != "1" ]; then
  glm53_warn "Offline mode is disabled; start may attempt a 181 GiB network download"
fi
if [ "${OOM_GUARD:-1}" = "1" ] && [ "${VLLM_HOST:-0.0.0.0}" != "0.0.0.0" ]; then
  glm53_die "OOM_GUARD=1 requires VLLM_HOST=0.0.0.0 so the worker can observe head readiness"
fi

NODE_RANK=0 HEADLESS= glm53_compose config --quiet

glm53_info "Configuration is valid"
printf '  profile: %s\n' "$GLM53_PROFILE_RESOLVED"
printf '  model: %s@%s\n' "$MODEL_ID" "$MODEL_REVISION"
printf '  image: %s\n' "$GLM53_VLLM_IMAGE"
printf '  head/worker fabric: %s / %s\n' "$VLLM_HOST_IP" "$WORKER_VLLM_HOST_IP"
printf '  context/sequences/batch: %s / %s / %s\n' "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS"
printf '  backend/eager/MTP: %s / %s / %s\n' "$MOE_BACKEND" "$ENFORCE_EAGER" "$MTP_NUM_TOKENS"
printf '  memory: gpu-util=%s container-limit=%s\n' "$GPU_MEMORY_UTILIZATION" "${CONTAINER_MEMORY_LIMIT:-112g}"
