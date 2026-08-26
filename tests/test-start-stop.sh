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

"$ROOT_DIR/start-glm53.sh" 32k > "$TEST_STATE/start-output"
grep -F 'GLM-5.3-Flash is serving' "$TEST_STATE/start-output" >/dev/null
[ -f "$TEST_STATE/head" ]
[ -f "$TEST_STATE/worker" ]

"$ROOT_DIR/stop-glm53.sh" --profile 32k > "$TEST_STATE/stop-output"
[ ! -f "$TEST_STATE/head" ]
[ ! -f "$TEST_STATE/worker" ]
grep -F 'Both ranks stopped' "$TEST_STATE/stop-output" >/dev/null
printf 'mocked worker-first start/stop: OK\n'
