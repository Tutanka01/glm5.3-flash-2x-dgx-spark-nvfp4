#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMMAND="$(ruby -ryaml -e '
value=YAML.load_file(ARGV[0]).fetch("services").fetch("vllm-glm53").fetch("command").first
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
  MODEL_ID=LibertAIDAI/GLM-5.3-Flash-NVFP4 \
  MODEL_REVISION=11d73216cd636238e82e1d77fe1042ffab36e7fa \
  SERVED_MODEL_NAME=glm-5.3-flash-nvfp4 \
  VLLM_HOST=0.0.0.0 VLLM_PORT=8888 NODE_RANK=1 HEADLESS=1 \
  MASTER_ADDR=10.10.10.1 MASTER_PORT=25000 \
  bash -c "$COMMAND" > "$output_file"

  grep -Fx -- 'serve' "$output_file" >/dev/null
  grep -Fx -- 'LibertAIDAI/GLM-5.3-Flash-NVFP4' "$output_file" >/dev/null
  grep -Fx -- '--tensor-parallel-size' "$output_file" >/dev/null
  grep -Fx -- '--headless' "$output_file" >/dev/null
  grep -Fx -- "$MAX_MODEL_LEN" "$output_file" >/dev/null

  if [ "$MOE_BACKEND" = "auto" ]; then
    ! grep -Fx -- '--moe-backend' "$output_file" >/dev/null
  else
    grep -Fx -- '--moe-backend' "$output_file" >/dev/null
    grep -Fx -- "$MOE_BACKEND" "$output_file" >/dev/null
  fi
  if [ "$MTP_NUM_TOKENS" -gt 0 ]; then
    grep -Fx -- '--speculative-config' "$output_file" >/dev/null
    grep -F -- "\"num_speculative_tokens\":$MTP_NUM_TOKENS" "$output_file" >/dev/null
  else
    ! grep -Fx -- '--speculative-config' "$output_file" >/dev/null
  fi
  rm -f "$output_file"
}

for profile in 32k 64k 128k 256k 32k-mtp 32k-native; do
  run_profile "$profile"
done
printf 'compose command profiles: OK\n'

