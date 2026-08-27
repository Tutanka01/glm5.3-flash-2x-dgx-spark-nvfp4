#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMMAND="$(ruby -ryaml -e '
value=YAML.load_file(ARGV[0]).fetch("services").fetch("sglang-glm53").fetch("command").first
print value.gsub("$$", "$")
' "$ROOT_DIR/docker-compose.glm53.yml")"

RUN_OVERRIDES=""

run_profile() {
  local profile="$1"
  # Optional per-call knob overrides, e.g. RUN_OVERRIDES="EP_SIZE=1".
  local overrides="$RUN_OVERRIDES"
  RUN_OVERRIDES=""
  local output_file
  output_file="$(mktemp /tmp/glm53-command.XXXXXX)"
  unset EP_SIZE ENABLE_TORCH_COMPILE ENABLE_MIXED_CHUNK SCHEDULE_CONSERVATIVENESS
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/profiles/$profile.env"
  set +a
  if [ -n "$overrides" ]; then
    # shellcheck disable=SC2086
    export $overrides
  fi
  # Mirror the compose environment defaults for optional knobs.
  EP_SIZE="${EP_SIZE:-2}"; export EP_SIZE
  ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-0}"; export ENABLE_TORCH_COMPILE
  TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-4}"; export TORCH_COMPILE_MAX_BS
  ENABLE_MIXED_CHUNK="${ENABLE_MIXED_CHUNK:-0}"; export ENABLE_MIXED_CHUNK
  SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS:-1.0}"; export SCHEDULE_CONSERVATIVENESS
  PATH="$ROOT_DIR/tests/mock-bin:$PATH" \
  MODEL_SNAPSHOT_CONTAINER=/cache/huggingface/hub/models--test/snapshots/f4aa \
  SERVED_MODEL_NAME=glm-5.3-flash-nvfp4 \
  API_HOST=0.0.0.0 API_PORT=8888 API_KEY= \
  NODE_RANK=1 HEADLESS=1 MASTER_ADDR=10.10.10.1 MASTER_PORT=25000 \
  bash -c "$COMMAND" > "$output_file"

  grep -Fx -- '-m' "$output_file" >/dev/null
  grep -Fx -- 'sglang.launch_server' "$output_file" >/dev/null
  grep -Fx -- '--model-path' "$output_file" >/dev/null
  grep -Fx -- '/cache/huggingface/hub/models--test/snapshots/f4aa' "$output_file" >/dev/null
  grep -Fx -- '--tp-size' "$output_file" >/dev/null
  grep -Fx -- '--ep-size' "$output_file" >/dev/null
  grep -Fx -- '--nnodes' "$output_file" >/dev/null
  grep -Fx -- '--node-rank' "$output_file" >/dev/null
  grep -Fx -- '--dist-init-addr' "$output_file" >/dev/null
  grep -Fx -- '10.10.10.1:25000' "$output_file" >/dev/null
  grep -Fx -- "$MAX_MODEL_LEN" "$output_file" >/dev/null
  grep -Fx -- "$MAX_NUM_SEQS" "$output_file" >/dev/null
  grep -Fx -- "$MOE_BACKEND" "$output_file" >/dev/null
  grep -Fx -- "$DSA_DECODE_BACKEND" "$output_file" >/dev/null
  grep -Fx -- '--disable-shared-experts-fusion' "$output_file" >/dev/null
  ! grep -Fx -- '--chat-template' "$output_file" >/dev/null

  if [ "$DISABLE_CUDA_GRAPH" = "1" ]; then
    grep -Fx -- '--disable-cuda-graph' "$output_file" >/dev/null
  else
    ! grep -Fx -- '--disable-cuda-graph' "$output_file" >/dev/null
  fi
  if [ "$MTP_NUM_TOKENS" -gt 0 ]; then
    grep -Fx -- '--speculative-algorithm' "$output_file" >/dev/null
    grep -Fx -- 'NEXTN' "$output_file" >/dev/null
    grep -Fx -- '--speculative-num-steps' "$output_file" >/dev/null
    grep -Fx -- "$MTP_NUM_TOKENS" "$output_file" >/dev/null
    grep -Fx -- '--speculative-num-draft-tokens' "$output_file" >/dev/null
  else
    ! grep -Fx -- '--speculative-algorithm' "$output_file" >/dev/null
  fi

  grep -Fx -- '--schedule-conservativeness' "$output_file" >/dev/null
  grep -A1 -Fx -- '--schedule-conservativeness' "$output_file" | grep -Fx -- "$SCHEDULE_CONSERVATIVENESS" >/dev/null
  grep -A1 -Fx -- '--ep-size' "$output_file" | grep -Fx -- "$EP_SIZE" >/dev/null
  if [ "$ENABLE_TORCH_COMPILE" = "1" ]; then
    grep -Fx -- '--enable-torch-compile' "$output_file" >/dev/null
    grep -A1 -Fx -- '--torch-compile-max-bs' "$output_file" | grep -Fx -- "$TORCH_COMPILE_MAX_BS" >/dev/null
  else
    ! grep -Fx -- '--enable-torch-compile' "$output_file" >/dev/null
  fi
  if [ "$ENABLE_MIXED_CHUNK" = "1" ]; then
    grep -Fx -- '--enable-mixed-chunk' "$output_file" >/dev/null
  else
    ! grep -Fx -- '--enable-mixed-chunk' "$output_file" >/dev/null
  fi
  rm -f "$output_file"
}

for profile in 32k 32k-batch4 32k-batch8 64k 128k 128k-batch4 128k-batch4-mtp 128k-batch4-mtp3 128k-batch2-mtp 128k-batch4-8k 128k-batch8 128k-ep1 256k 256k-graphs 32k-mtp 32k-eager; do
  run_profile "$profile"
done

# The optional optimization knobs must reach the rendered command.
RUN_OVERRIDES="EP_SIZE=1 ENABLE_TORCH_COMPILE=1 SCHEDULE_CONSERVATIVENESS=1.3" run_profile 128k-batch4
RUN_OVERRIDES="" run_profile 128k-batch4
printf 'SGLang compose command profiles: OK\n'
