#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMMAND="$(ruby -ryaml -e '
value=YAML.load_file(ARGV[0]).fetch("services").fetch("sglang-glm53").fetch("command").first
print value.gsub("$$", "$")
' "$ROOT_DIR/docker-compose.glm53.yml")"

run_profile() {
  local profile="$1"
  local output_file
  output_file="$(mktemp /tmp/glm53-command.XXXXXX)"
  set -a
  # shellcheck disable=SC1090
  source "$ROOT_DIR/profiles/$profile.env"
  set +a
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
  rm -f "$output_file"
}

for profile in 32k 32k-batch4 32k-batch8 64k 128k 128k-batch4 128k-batch8 256k 32k-mtp 32k-eager; do
  run_profile "$profile"
done
printf 'SGLang compose command profiles: OK\n'
