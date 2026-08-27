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
  NODE_IP="$HEAD_FABRIC_IP"
  PEER_IP="$WORKER_FABRIC_IP"
  NODE_CACHE="$HF_CACHE"
  NODE_HCA="$NCCL_IB_HCA"
  NODE_NCCL_IF="$NCCL_SOCKET_IFNAME"
  NODE_TP_IF="$TP_SOCKET_IFNAME"
  NODE_GLOO_IF="$GLOO_SOCKET_IFNAME"
else
  NODE_IP="$WORKER_FABRIC_IP"
  PEER_IP="$HEAD_FABRIC_IP"
  NODE_CACHE="$WORKER_HF_CACHE"
  NODE_HCA="${WORKER_NCCL_IB_HCA:-$NCCL_IB_HCA}"
  NODE_NCCL_IF="${WORKER_NCCL_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
  NODE_TP_IF="${WORKER_TP_SOCKET_IFNAME:-$TP_SOCKET_IFNAME}"
  NODE_GLOO_IF="${WORKER_GLOO_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}"
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

  ROUTE_LINE="$(ip -4 route get "$PEER_IP" 2>/dev/null | head -n 1)"
  ROUTE_DEV="$(printf '%s\n' "$ROUTE_LINE" | awk '{for (i=1;i<=NF;i++) if ($i=="dev") print $(i+1)}')"
  ROUTE_SRC="$(printf '%s\n' "$ROUTE_LINE" | awk '{for (i=1;i<=NF;i++) if ($i=="src") print $(i+1)}')"
  if [ "$ROUTE_DEV" = "$NODE_GLOO_IF" ] && [ "$ROUTE_SRC" = "$NODE_IP" ]; then
    pass "route to $PEER_IP uses $ROUTE_DEV with source $ROUTE_SRC"
  else
    fail "route to $PEER_IP is '$ROUTE_LINE'; expected dev $NODE_GLOO_IF src $NODE_IP"
  fi
else
  fail "ip command not found"
fi

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

  gid_root="/sys/class/infiniband/$hca_name/ports/$hca_port"
  roce_v2_found=0
  for gid_path in "$gid_root"/gids/*; do
    [ -e "$gid_path" ] || continue
    gid_index="${gid_path##*/}"
    gid_value="$(cat "$gid_path" 2>/dev/null || true)"
    case "$gid_value" in ''|::|0000:0000:0000:0000:0000:0000:0000:0000) continue ;; esac
    gid_type="$(cat "$gid_root/gid_attrs/types/$gid_index" 2>/dev/null || true)"
    gid_ndev="$(cat "$gid_root/gid_attrs/ndevs/$gid_index" 2>/dev/null || true)"
    printf '  [INFO] GID %s = %s type=%s ndev=%s\n' \
      "$gid_index" "$gid_value" "${gid_type:-unknown}" "${gid_ndev:-unknown}"
    case "$gid_type" in
      *v2*)
        if [ -z "$gid_ndev" ] || [ "$gid_ndev" = "$NODE_NCCL_IF" ]; then
          roce_v2_found=1
        fi
        ;;
    esac
  done
  if [ "$roce_v2_found" = "1" ]; then
    pass "populated RoCE v2 GID available; NCCL will select it dynamically"
  else
    fail "no populated RoCE v2 GID on $hca_name:$hca_port for $NODE_NCCL_IF"
  fi
done

if [ -e /dev/infiniband ]; then
  pass "/dev/infiniband is present"
else
  fail "/dev/infiniband is missing"
fi

if docker image inspect "$GLM53_RUNTIME_IMAGE" >/dev/null 2>&1; then
  pass "pinned SGLang image is local"
else
  warn "pinned SGLang image is not local yet (prepare will pull it)"
fi

if [ "$ERRORS" -gt 0 ]; then
  printf 'Node doctor failed: %d error(s), %d warning(s)\n' "$ERRORS" "$WARNINGS" >&2
  exit 1
fi
printf 'Node doctor passed: %d warning(s)\n' "$WARNINGS"
