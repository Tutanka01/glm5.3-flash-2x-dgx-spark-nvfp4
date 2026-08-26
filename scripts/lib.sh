#!/usr/bin/env bash

if [ "${GLM53_LIB_LOADED:-0}" = "1" ]; then
  return 0 2>/dev/null || exit 0
fi
GLM53_LIB_LOADED=1

GLM53_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLM53_ENV_FILE="${GLM53_ENV_FILE:-$GLM53_ROOT/.env.glm53}"
GLM53_COMPOSE_FILE="$GLM53_ROOT/docker-compose.glm53.yml"
GLM53_PROJECT_NAME="${GLM53_PROJECT_NAME:-glm53}"
GLM53_SERVICE="vllm-glm53"

glm53_info() {
  printf '[glm53] %s\n' "$*"
}

glm53_warn() {
  printf '[glm53] WARNING: %s\n' "$*" >&2
}

glm53_die() {
  printf '[glm53] ERROR: %s\n' "$*" >&2
  exit 1
}

glm53_require_command() {
  command -v "$1" >/dev/null 2>&1 || glm53_die "Required command not found: $1"
}

glm53_is_placeholder() {
  case "${1:-}" in
    ""|*REQUIRED*|*replace-me*|*worker-host*|*management-ip*|*head-roce-ip*|*worker-roce-ip*|*/USER/*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

glm53_load_config() {
  local requested_profile="${1:-}"

  [ -f "$GLM53_ENV_FILE" ] || \
    glm53_die "Missing $GLM53_ENV_FILE. Run: cp .env.glm53.example .env.glm53"
  [ -f "$GLM53_COMPOSE_FILE" ] || glm53_die "Missing $GLM53_COMPOSE_FILE"

  set -a
  # shellcheck disable=SC1090
  source "$GLM53_ENV_FILE"
  set +a

  GLM53_PROFILE_RESOLVED="${requested_profile:-${GLM53_PROFILE:-32k}}"
  case "$GLM53_PROFILE_RESOLVED" in
    *[!A-Za-z0-9._-]*|"")
      glm53_die "Invalid profile name: $GLM53_PROFILE_RESOLVED"
      ;;
  esac

  GLM53_PROFILE_FILE="$GLM53_ROOT/profiles/$GLM53_PROFILE_RESOLVED.env"
  [ -f "$GLM53_PROFILE_FILE" ] || \
    glm53_die "Unknown profile '$GLM53_PROFILE_RESOLVED' (missing $GLM53_PROFILE_FILE)"

  set -a
  # shellcheck disable=SC1090
  source "$GLM53_PROFILE_FILE"
  set +a

  WORKER_HF_CACHE="${WORKER_HF_CACHE:-${HF_CACHE:-}}"
  SSH_PORT="${SSH_PORT:-22}"
  export GLM53_ROOT GLM53_ENV_FILE GLM53_COMPOSE_FILE GLM53_PROJECT_NAME
  export GLM53_SERVICE GLM53_PROFILE_RESOLVED GLM53_PROFILE_FILE
  export WORKER_HF_CACHE SSH_PORT

  GLM53_SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=10 -p "$SSH_PORT")
  GLM53_SCP_ARGS=(-q -o BatchMode=yes -o ConnectTimeout=10 -P "$SSH_PORT")
  if [ -n "${SSH_IDENTITY_FILE:-}" ]; then
    GLM53_SSH_ARGS+=(-i "$SSH_IDENTITY_FILE")
    GLM53_SCP_ARGS+=(-i "$SSH_IDENTITY_FILE")
  fi
}

glm53_compose() {
  COMPOSE_DISABLE_ENV_FILE=1 \
  NODE_RANK="${NODE_RANK:-0}" \
  HEADLESS="${HEADLESS:-}" \
  docker compose \
    --project-name "$GLM53_PROJECT_NAME" \
    --env-file "$GLM53_ENV_FILE" \
    --env-file "$GLM53_PROFILE_FILE" \
    -f "$GLM53_COMPOSE_FILE" \
    "$@"
}

glm53_validate_remote_paths() {
  case "${WORKER_HOST:-}" in
    ""|*[!A-Za-z0-9_.:@%+-]*)
      glm53_die "WORKER_HOST contains unsupported characters: ${WORKER_HOST:-<empty>}"
      ;;
  esac
  case "${WORKER_DIR:-}" in
    /*) ;;
    *) glm53_die "WORKER_DIR must be an absolute path (no ~): ${WORKER_DIR:-<empty>}" ;;
  esac
  case "$WORKER_DIR" in
    *[!A-Za-z0-9_./+:-]*|*/../*|*/..|*/./*|*/.)
      glm53_die "WORKER_DIR contains unsupported or ambiguous path components: $WORKER_DIR"
      ;;
  esac
}

glm53_ssh() {
  ssh "${GLM53_SSH_ARGS[@]}" "$WORKER_HOST" "$@"
}

glm53_scp() {
  scp "${GLM53_SCP_ARGS[@]}" "$@"
}

glm53_shell_join() {
  local joined=""
  local item quoted
  for item in "$@"; do
    printf -v quoted '%q' "$item"
    joined+="${joined:+ }$quoted"
  done
  printf '%s' "$joined"
}

glm53_worker_compose() {
  local worker_hca="${WORKER_NCCL_IB_HCA:-${NCCL_IB_HCA:-}}"
  local worker_nccl_if="${WORKER_NCCL_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-}}"
  local worker_tp_if="${WORKER_TP_SOCKET_IFNAME:-${TP_SOCKET_IFNAME:-}}"
  local worker_gloo_if="${WORKER_GLOO_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME:-}}"
  local worker_gid="${WORKER_NCCL_IB_GID_INDEX:-${NCCL_IB_GID_INDEX:-}}"
  local remote_command wrapped
  local -a command_parts

  command_parts=(
    env
    COMPOSE_DISABLE_ENV_FILE=1
    NODE_RANK=1
    HEADLESS=1
    "HF_CACHE=$WORKER_HF_CACHE"
    "VLLM_HOST_IP=$WORKER_VLLM_HOST_IP"
    "NCCL_IB_HCA=$worker_hca"
    "NCCL_SOCKET_IFNAME=$worker_nccl_if"
    "TP_SOCKET_IFNAME=$worker_tp_if"
    "GLOO_SOCKET_IFNAME=$worker_gloo_if"
    "NCCL_IB_GID_INDEX=$worker_gid"
    docker compose
    --project-name "$GLM53_PROJECT_NAME"
    --env-file .env.glm53
    --env-file "profiles/$GLM53_PROFILE_RESOLVED.env"
    -f docker-compose.glm53.yml
  )
  command_parts+=("$@")

  remote_command="cd $(glm53_shell_join "$WORKER_DIR") && $(glm53_shell_join "${command_parts[@]}")"
  printf -v wrapped 'bash -lc %q' "$remote_command"
  glm53_ssh "$wrapped"
}

glm53_worker_script() {
  local script_name="$1"
  shift
  local remote_command wrapped
  remote_command="cd $(glm53_shell_join "$WORKER_DIR") && $(glm53_shell_join "./scripts/$script_name" "$@")"
  printf -v wrapped 'bash -lc %q' "$remote_command"
  glm53_ssh "$wrapped"
}

glm53_sync_worker() {
  local remote_mkdir remote_chmod
  glm53_validate_remote_paths

  remote_mkdir="mkdir -p $(glm53_shell_join "$WORKER_DIR/scripts") $(glm53_shell_join "$WORKER_DIR/profiles") $(glm53_shell_join "$WORKER_DIR/metadata")"
  glm53_ssh "$remote_mkdir"

  glm53_scp "$GLM53_COMPOSE_FILE" "$GLM53_ENV_FILE" \
    "$WORKER_HOST:$WORKER_DIR/"
  glm53_scp "$GLM53_ROOT/scripts/"*.sh "$GLM53_ROOT/scripts/"*.py \
    "$WORKER_HOST:$WORKER_DIR/scripts/"
  glm53_scp "$GLM53_ROOT/profiles/"*.env \
    "$WORKER_HOST:$WORKER_DIR/profiles/"
  glm53_scp "$GLM53_ROOT/metadata/"*.json \
    "$WORKER_HOST:$WORKER_DIR/metadata/"

  remote_chmod="chmod 600 $(glm53_shell_join "$WORKER_DIR/.env.glm53") && chmod +x $(glm53_shell_join "$WORKER_DIR/scripts")/*.sh"
  glm53_ssh "$remote_chmod"
}

glm53_api_curl() {
  local -a curl_args
  curl_args=(-fsS --connect-timeout 5 --max-time "${GLM53_CURL_MAX_TIME:-30}")
  if [ -n "${VLLM_API_KEY:-}" ]; then
    curl_args+=(-H "Authorization: Bearer $VLLM_API_KEY")
  fi
  curl "${curl_args[@]}" "$@"
}

glm53_container_id_local() {
  glm53_compose ps -q "$GLM53_SERVICE" 2>/dev/null | head -n 1
}

glm53_container_running_local() {
  local container_id
  container_id="$(glm53_container_id_local)"
  [ -n "$container_id" ] || return 1
  [ "$(docker inspect -f '{{.State.Running}}' "$container_id" 2>/dev/null)" = "true" ]
}

glm53_stop_guard_pid_file() {
  local pid_file="$1"
  local guard_pid guard_command
  [ -f "$pid_file" ] || return 0
  guard_pid="$(sed -n '1p' "$pid_file" 2>/dev/null || true)"
  case "$guard_pid" in
    ''|*[!0-9]*) ;;
    *)
      if kill -0 "$guard_pid" 2>/dev/null; then
        guard_command="$(ps -p "$guard_pid" -o command= 2>/dev/null || true)"
        case "$guard_command" in
          *oom-guard-node.sh*) kill "$guard_pid" 2>/dev/null || true ;;
          *) glm53_warn "Refusing to kill stale PID $guard_pid; it is not an oom-guard-node process" ;;
        esac
      fi
      ;;
  esac
  rm -f "$pid_file"
}

glm53_stop_guard_local() {
  local pid_file="$GLM53_ROOT/.glm53-guard-head.pid"
  glm53_stop_guard_pid_file "$pid_file"
}
