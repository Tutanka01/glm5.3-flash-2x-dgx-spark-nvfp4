#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEST_STATE="$(mktemp -d /tmp/glm53-start-test.XXXXXX)"
cleanup() { rm -rf "$TEST_STATE"; }
trap cleanup EXIT

export GLM53_MOCK_STATE="$TEST_STATE"
export GLM53_ENV_FILE="$ROOT_DIR/tests/fixtures/valid.env"
export PATH="$ROOT_DIR/tests/mock-start-bin:$PATH"
export START_RUN_DOCTOR=0
export START_SMOKE=0
export OOM_GUARD=0

"$ROOT_DIR/start-glm53.sh" 256k > "$TEST_STATE/validated-256k-start" 2>&1
! grep -F "Starting unproven capacity candidate '256k'" \
  "$TEST_STATE/validated-256k-start" >/dev/null
grep -F 'GLM-5.3-Flash is serving' "$TEST_STATE/validated-256k-start" >/dev/null
"$ROOT_DIR/stop-glm53.sh" --profile 256k >/dev/null

"$ROOT_DIR/start-glm53.sh" 512k-mtp-eager > "$TEST_STATE/quarantine-start" 2>&1
grep -F "Starting quarantined profile '512k-mtp-eager'" \
  "$TEST_STATE/quarantine-start" >/dev/null
grep -F 'GLM-5.3-Flash is serving' "$TEST_STATE/quarantine-start" >/dev/null
"$ROOT_DIR/stop-glm53.sh" --profile 512k-mtp-eager >/dev/null

"$ROOT_DIR/start-glm53.sh" 32k > "$TEST_STATE/start-output"
grep -F 'GLM-5.3-Flash is serving' "$TEST_STATE/start-output" >/dev/null
[ -f "$TEST_STATE/head" ]
[ -f "$TEST_STATE/worker" ]

"$ROOT_DIR/status-glm53.sh" > "$TEST_STATE/status-output"
grep -F 'Profile: 32k (running)' "$TEST_STATE/status-output" >/dev/null
grep -F 'Runtime: context=32768 requests=1 prefill-chunk=4096 static=0.90 graphs-disabled=0 MTP=0' \
  "$TEST_STATE/status-output" >/dev/null
grep -F 'Admission tier: validated' "$TEST_STATE/status-output" >/dev/null
grep -F 'Speculation: NONE' "$TEST_STATE/status-output" >/dev/null
grep -F 'Head startup memory guard: inactive' "$TEST_STATE/status-output" >/dev/null

export GLM53_MOCK_PROFILE=256k-mtp
export GLM53_MOCK_MAX_MODEL_LEN=262144
export GLM53_MOCK_MAX_NUM_BATCHED_TOKENS=4096
export GLM53_MOCK_MEM_FRACTION_STATIC=0.90
export GLM53_MOCK_DISABLE_CUDA_GRAPH=0
export GLM53_MOCK_MTP_NUM_TOKENS=5
"$ROOT_DIR/status-glm53.sh" 256k-mtp > "$TEST_STATE/unsafe-status" 2>&1
grep -F 'Cold >128k preflight: EXPERIMENTAL' "$TEST_STATE/unsafe-status" >/dev/null
unset GLM53_MOCK_PROFILE GLM53_MOCK_MAX_MODEL_LEN
unset GLM53_MOCK_MAX_NUM_BATCHED_TOKENS GLM53_MOCK_MEM_FRACTION_STATIC
unset GLM53_MOCK_DISABLE_CUDA_GRAPH GLM53_MOCK_MTP_NUM_TOKENS

"$ROOT_DIR/stop-glm53.sh" --profile 32k > "$TEST_STATE/stop-output"
[ ! -f "$TEST_STATE/head" ]
[ ! -f "$TEST_STATE/worker" ]
grep -F 'Both ranks stopped' "$TEST_STATE/stop-output" >/dev/null

export OOM_GUARD=1
export GLM53_GUARD_STATE_DIR="$TEST_STATE/guards"
mkdir -p "$GLM53_GUARD_STATE_DIR"
"$ROOT_DIR/scripts/start-guard-node.sh" head 32k >/dev/null
for _ in 1 2 3 4 5 6 7 8 9 10; do
  grep -F 'API healthy; guard complete' \
    "$GLM53_GUARD_STATE_DIR/.glm53-guard-head.log" >/dev/null 2>&1 && break
  /bin/sleep 0.05
done
grep -F 'health=http://127.0.0.1:8888/v1/tokenize' \
  "$GLM53_GUARD_STATE_DIR/.glm53-guard-head.log" >/dev/null
grep -F 'API healthy; guard complete' \
  "$GLM53_GUARD_STATE_DIR/.glm53-guard-head.log" >/dev/null
"$ROOT_DIR/scripts/stop-guard-node.sh" head

"$ROOT_DIR/scripts/start-guard-node.sh" worker 32k >/dev/null
for _ in 1 2 3 4 5 6 7 8 9 10; do
  grep -F 'API healthy; guard complete' \
    "$GLM53_GUARD_STATE_DIR/.glm53-guard-worker.log" >/dev/null 2>&1 && break
  /bin/sleep 0.05
done
grep -F 'health=http://10.10.10.1:8888/v1/tokenize' \
  "$GLM53_GUARD_STATE_DIR/.glm53-guard-worker.log" >/dev/null
grep -F 'API healthy; guard complete' \
  "$GLM53_GUARD_STATE_DIR/.glm53-guard-worker.log" >/dev/null
"$ROOT_DIR/scripts/stop-guard-node.sh" worker

printf 'mocked worker-first start/stop: OK\n'
