#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

ROLE="${1:-head}"
PROFILE="${2:-}"
case "$ROLE" in head|worker) ;; *) glm53_die "doctor-node role must be head or worker" ;; esac
glm53_load_config "$PROFILE"

ERRORS=0
WARNINGS=0

pass() { printf '  [OK] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*" >&2; WARNINGS=$((WARNINGS + 1)); }
fail() { printf '  [FAIL] %s\n' "$*" >&2; ERRORS=$((ERRORS + 1)); }

if [ "$ROLE" = "head" ]; then
  NODE_IP="$VLLM_HOST_IP"
  NODE_CACHE="$HF_CACHE"
  NODE_HCA="$NCCL_IB_HCA"
  NODE_NCCL_IF="$NCCL_SOCKET_IFNAME"
  NODE_TP_IF="$TP_SOCKET_IFNAME"
  NODE_GLOO_IF="$GLOO_SOCKET_IFNAME"
  NODE_GID="$NCCL_IB_GID_INDEX"
else
  NODE_IP="$WORKER_VLLM_HOST_IP"
  NODE_CACHE="$WORKER_HF_CACHE"
  NODE_HCA="${WORKER_NCCL_IB_HCA:-$NCCL_IB_HCA}"
  NODE_NCCL_IF="${WORKER_NCCL_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
  NODE_TP_IF="${WORKER_TP_SOCKET_IFNAME:-$TP_SOCKET_IFNAME}"
  NODE_GLOO_IF="${WORKER_GLOO_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}"
  NODE_GID="${WORKER_NCCL_IB_GID_INDEX:-$NCCL_IB_GID_INDEX}"
fi

printf 'Node doctor: %s\n' "$ROLE"

NODE_ARCH="$(uname -m 2>/dev/null || true)"
case "$NODE_ARCH" in
  aarch64|arm64) pass "architecture=$NODE_ARCH" ;;
  *)
    if [ "${ALLOW_UNSUPPORTED_PLATFORM:-0}" = "1" ]; then
      warn "unsupported architecture=$NODE_ARCH (override enabled)"
    else
      fail "expected aarch64/arm64, got $NODE_ARCH"
    fi
    ;;
esac

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  pass "Docker daemon reachable"
  if docker compose version >/dev/null 2>&1; then
    pass "Docker Compose v2 available"
  else
    fail "Docker Compose v2 unavailable"
  fi
else
  fail "Docker daemon is not reachable by the current user"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
  if [ -n "$GPU_LINE" ]; then
    pass "GPU=$GPU_LINE"
    case "$GPU_LINE" in *GB10*|*Spark*) ;; *) warn "GPU name does not explicitly identify GB10" ;; esac
  else
    fail "nvidia-smi returned no GPU"
  fi
else
  fail "nvidia-smi not found"
fi

if [ -r /proc/meminfo ]; then
  AVAILABLE_MB="$(awk '/^MemAvailable:/ {printf "%d", $2 / 1024}' /proc/meminfo)"
  if [ "${AVAILABLE_MB:-0}" -ge "${MIN_START_AVAILABLE_MB:-100000}" ]; then
    pass "available unified memory=${AVAILABLE_MB} MiB"
  else
    RUNNING_RECIPE="$(docker ps -q \
      --filter "label=com.docker.compose.project=$GLM53_PROJECT_NAME" \
      --filter "label=com.docker.compose.service=$GLM53_SERVICE" 2>/dev/null | head -n 1)"
    if [ -n "$RUNNING_RECIPE" ]; then
      warn "available memory=${AVAILABLE_MB} MiB while this recipe is already running"
    else
      fail "available memory=${AVAILABLE_MB} MiB; need at least ${MIN_START_AVAILABLE_MB:-100000} MiB before load"
    fi
  fi
else
  fail "/proc/meminfo unavailable (this recipe targets Linux)"
fi

if [ -r /proc/swaps ] && [ "$(wc -l < /proc/swaps)" -gt 1 ]; then
  if [ "${REQUIRE_SWAP_OFF:-0}" = "1" ]; then
    fail "swap is active; disable it before loading this unified-memory model"
  else
    warn "swap is active; memswap_limit protects the container, but swapoff is safer during bring-up"
  fi
else
  pass "swap is disabled"
fi

if mkdir -p "$NODE_CACHE" 2>/dev/null && [ -w "$NODE_CACHE" ]; then
  pass "cache is writable: $NODE_CACHE"
  SNAPSHOT_DIR="$NODE_CACHE/hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"
  if [ -d "$SNAPSHOT_DIR" ]; then
    pass "pinned snapshot directory exists"
  else
    FREE_GIB="$(df -Pk "$NODE_CACHE" | awk 'NR==2 {printf "%d", $4 / 1024 / 1024}')"
    if [ "${FREE_GIB:-0}" -ge "${MIN_CACHE_FREE_GIB:-205}" ]; then
      pass "cache free space=${FREE_GIB} GiB"
    else
      fail "cache free space=${FREE_GIB} GiB; need at least ${MIN_CACHE_FREE_GIB:-205} GiB for first download"
    fi
  fi
else
  fail "cache is not writable: $NODE_CACHE"
fi

if command -v ip >/dev/null 2>&1; then
  for node_if in "$NODE_NCCL_IF" "$NODE_TP_IF" "$NODE_GLOO_IF"; do
    if ip link show dev "$node_if" >/dev/null 2>&1; then
      pass "interface exists: $node_if"
    else
      fail "interface missing: $node_if"
    fi
  done
  if ip -4 -o addr show dev "$NODE_GLOO_IF" 2>/dev/null \
    | awk '{split($4, parts, "/"); print parts[1]}' \
    | grep -Fqx "$NODE_IP"; then
    pass "fabric IP $NODE_IP is assigned to $NODE_GLOO_IF"
  else
    fail "fabric IP $NODE_IP is not assigned to $NODE_GLOO_IF"
  fi
else
  fail "ip command not found"
fi

case "$NODE_GID" in ''|*[!0-9]*) fail "NCCL_IB_GID_INDEX is not a decimal integer: $NODE_GID" ;; esac

OLD_IFS="$IFS"
IFS=','
read -r -a HCA_TOKENS <<< "${NODE_HCA#=}"
IFS="$OLD_IFS"
for hca_token in "${HCA_TOKENS[@]}"; do
  case "$hca_token" in
    ^*) warn "excluded HCA selector not inspected: $hca_token"; continue ;;
  esac
  hca_name="${hca_token%%:*}"
  hca_port="1"
  if [ "$hca_name" != "$hca_token" ]; then
    hca_remainder="${hca_token#*:}"
    case "${hca_remainder%%:*}" in ''|*[!0-9]*) ;; *) hca_port="${hca_remainder%%:*}" ;; esac
  fi
  if [ -d "/sys/class/infiniband/$hca_name" ]; then
    pass "RDMA HCA exists: $hca_name"
  else
    fail "RDMA HCA missing: $hca_name"
    continue
  fi
  if [ "$NODE_GID" -ge 0 ] 2>/dev/null; then
    gid_path="/sys/class/infiniband/$hca_name/ports/$hca_port/gids/$NODE_GID"
    gid_type_path="/sys/class/infiniband/$hca_name/ports/$hca_port/gid_attrs/types/$NODE_GID"
    if [ -r "$gid_path" ]; then
      gid_value="$(tr -d '[:space:]' < "$gid_path")"
      case "$gid_value" in ''|::|0000:0000:0000:0000:0000:0000:0000:0000) fail "empty GID at $hca_name:$hca_port index $NODE_GID" ;; *) pass "GID $NODE_GID on $hca_name:$hca_port = $gid_value" ;; esac
      if [ -r "$gid_type_path" ]; then
        gid_type="$(tr -d '\r\n' < "$gid_type_path")"
        case "$gid_type" in *v2*) pass "GID type=$gid_type" ;; *) warn "GID type is '$gid_type', expected RoCE v2" ;; esac
      fi
    else
      fail "GID index $NODE_GID does not exist on $hca_name:$hca_port"
    fi
  fi
done

if [ -e /dev/infiniband ]; then
  pass "/dev/infiniband is present"
else
  fail "/dev/infiniband is missing"
fi

if docker image inspect "$GLM53_VLLM_IMAGE" >/dev/null 2>&1; then
  pass "pinned vLLM image is local"
else
  warn "pinned vLLM image is not local yet (prepare will pull it)"
fi

if [ "$ERRORS" -gt 0 ]; then
  printf 'Node doctor failed: %d error(s), %d warning(s)\n' "$ERRORS" "$WARNINGS" >&2
  exit 1
fi
printf 'Node doctor passed: %d warning(s)\n' "$WARNINGS"
