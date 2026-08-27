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
  HEAD_FABRIC_IP WORKER_FABRIC_IP
  NCCL_IB_HCA NCCL_SOCKET_IFNAME TP_SOCKET_IFNAME GLOO_SOCKET_IFNAME
  NCCL_IB_ADDR_RANGE HF_CACHE MODEL_ID MODEL_REVISION GLM53_RUNTIME_IMAGE
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
[ "$MODEL_REVISION" = "$EXPECTED_REVISION" ] || \
  glm53_die "MODEL_REVISION is stale; copy the current pin from .env.glm53.example"
[ "$GLM53_RUNTIME_IMAGE" = "$EXPECTED_IMAGE" ] || \
  glm53_die "GLM53_RUNTIME_IMAGE differs from the audited SGLang SM121 image"

python3 - \
  "$MASTER_ADDR" "$HEAD_FABRIC_IP" "$WORKER_FABRIC_IP" \
  "$NCCL_IB_ADDR_RANGE" "${WORKER_NCCL_IB_ADDR_RANGE:-$NCCL_IB_ADDR_RANGE}" <<'PY'
import ipaddress
import sys

addresses = [ipaddress.ip_address(value) for value in sys.argv[1:4]]
if addresses[1] == addresses[2]:
    raise SystemExit("HEAD_FABRIC_IP and WORKER_FABRIC_IP must differ")
for label, value in zip(("head", "worker"), sys.argv[4:6]):
    network = ipaddress.ip_network(value, strict=False)
    if network.version != 4:
        raise SystemExit(f"{label} NCCL_IB_ADDR_RANGE must be IPv4")
    address = addresses[1] if label == "head" else addresses[2]
    if address not in network:
        raise SystemExit(f"{label} fabric IP {address} is outside {network}")
PY

case "$MASTER_PORT" in ''|*[!0-9]*) glm53_die "MASTER_PORT must be an integer" ;; esac
case "$API_PORT" in ''|*[!0-9]*) glm53_die "API_PORT must be an integer" ;; esac
[ "$MASTER_PORT" -ge 1 ] && [ "$MASTER_PORT" -le 65535 ] || glm53_die "MASTER_PORT is out of range"
[ "$API_PORT" -ge 1 ] && [ "$API_PORT" -le 65535 ] || glm53_die "API_PORT is out of range"

for integer_name in \
  MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS CUDA_GRAPH_MAX_BS \
  MTP_NUM_TOKENS HF_DOWNLOAD_WORKERS; do
  integer_value="${!integer_name:-}"
  case "$integer_value" in ''|*[!0-9]*) glm53_die "$integer_name must be a non-negative integer" ;; esac
done
[ "$MAX_MODEL_LEN" -ge 4096 ] || glm53_die "MAX_MODEL_LEN must be at least 4096"
[ "$MAX_MODEL_LEN" -le 1048576 ] || glm53_die "MAX_MODEL_LEN exceeds the checkpoint limit"
[ "$MAX_NUM_SEQS" -ge 1 ] || glm53_die "MAX_NUM_SEQS must be at least 1"
[ "$MAX_NUM_BATCHED_TOKENS" -ge 512 ] || glm53_die "MAX_NUM_BATCHED_TOKENS must be at least 512"
[ "$CUDA_GRAPH_MAX_BS" -ge 1 ] || glm53_die "CUDA_GRAPH_MAX_BS must be at least 1"
[ "$CUDA_GRAPH_MAX_BS" -ge "$MAX_NUM_SEQS" ] || \
  glm53_die "CUDA_GRAPH_MAX_BS must cover MAX_NUM_SEQS"
[ "$MTP_NUM_TOKENS" -le 8 ] || glm53_die "MTP_NUM_TOKENS above 8 is rejected by this recipe"
[ "$HF_DOWNLOAD_WORKERS" -ge 1 ] && [ "$HF_DOWNLOAD_WORKERS" -le 16 ] || \
  glm53_die "HF_DOWNLOAD_WORKERS must be between 1 and 16"

case "$DISABLE_CUDA_GRAPH" in 0|1) ;; *) glm53_die "DISABLE_CUDA_GRAPH must be 0 or 1" ;; esac
case "$MOE_BACKEND" in
  flashinfer_cutlass|marlin) ;;
  *) glm53_die "MOE_BACKEND must be flashinfer_cutlass or marlin" ;;
esac
case "$DSA_PREFILL_BACKEND:$DSA_DECODE_BACKEND" in
  flashinfer_sparse_mla:flashinfer_sparse_mla) ;;
  *) glm53_die "This audited image requires flashinfer_sparse_mla for both DSA backends" ;;
esac
case "$KV_CACHE_DTYPE" in
  fp8_e4m3|bfloat16) ;;
  *) glm53_die "KV_CACHE_DTYPE must be fp8_e4m3 or bfloat16" ;;
esac

for switch_value in \
  "${OOM_GUARD:-1}" "${START_SMOKE:-1}" "${REQUIRE_SWAP_OFF:-0}" \
  "${ALLOW_UNSUPPORTED_PLATFORM:-0}" "${STRICT_FABRIC_ROUTE:-0}"; do
  case "$switch_value" in 0|1) ;; *) glm53_die "Boolean recipe switches must be 0 or 1" ;; esac
done

python3 - "$MEM_FRACTION_STATIC" <<'PY'
import sys
try:
    value=float(sys.argv[1])
except ValueError:
    raise SystemExit("MEM_FRACTION_STATIC must be a number")
if not 0.70 <= value <= 0.90:
    raise SystemExit("MEM_FRACTION_STATIC must be between 0.70 and 0.90")
PY

case "${CONTAINER_MEMORY_LIMIT:-120g}" in
  [1-9][0-9]g|1[01][0-9]g|12[0-7]g) ;;
  *) glm53_die "CONTAINER_MEMORY_LIMIT must be an integer between 10g and 127g" ;;
esac

if [ "$MASTER_ADDR" != "$HEAD_FABRIC_IP" ]; then
  glm53_warn "MASTER_ADDR differs from HEAD_FABRIC_IP; both must route over the intended fabric"
fi
if [ "${HF_HUB_OFFLINE:-1}" != "1" ] || [ "${TRANSFORMERS_OFFLINE:-1}" != "1" ]; then
  glm53_warn "Offline mode is disabled; boot could attempt an unexpected network lookup"
fi
if [ "${OOM_GUARD:-1}" = "1" ] && [ "$API_HOST" != "0.0.0.0" ]; then
  glm53_die "OOM_GUARD=1 requires API_HOST=0.0.0.0 so the worker can observe head readiness"
fi
if [ -n "${NCCL_IB_GID_INDEX:-}" ] || [ -n "${WORKER_NCCL_IB_GID_INDEX:-}" ]; then
  glm53_die "Remove NCCL_IB_GID_INDEX: NCCL >= 2.21 dynamically selects the RoCE v2 GID"
fi

NODE_RANK=0 HEADLESS= glm53_compose config --quiet

glm53_info "Configuration is valid"
printf '  profile: %s\n' "$GLM53_PROFILE_RESOLVED"
printf '  engine: SGLang (patched SM121 runtime)\n'
printf '  model: %s@%s\n' "$MODEL_ID" "$MODEL_REVISION"
printf '  image: %s\n' "$GLM53_RUNTIME_IMAGE"
printf '  head/worker fabric: %s / %s\n' "$HEAD_FABRIC_IP" "$WORKER_FABRIC_IP"
printf '  context/requests/prefill: %s / %s / %s\n' \
  "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS"
printf '  MoE/DSA/KV: %s / %s / %s\n' "$MOE_BACKEND" "$DSA_DECODE_BACKEND" "$KV_CACHE_DTYPE"
printf '  graphs-disabled/MTP: %s / %s\n' "$DISABLE_CUDA_GRAPH" "$MTP_NUM_TOKENS"
printf '  memory: static=%s container-limit=%s\n' \
  "$MEM_FRACTION_STATIC" "${CONTAINER_MEMORY_LIMIT:-120g}"
