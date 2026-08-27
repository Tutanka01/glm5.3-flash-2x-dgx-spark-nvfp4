#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEST_TMP="$(mktemp -d /tmp/glm53-api-test.XXXXXX)"
SERVER_PID=""
cleanup() {
  case "$SERVER_PID" in
    ''|*[!0-9]*) ;;
    *)
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
      ;;
  esac
  rm -rf "$TEST_TMP"
}
trap cleanup EXIT

python3 "$ROOT_DIR/tests/mock_openai_server.py" --port-file "$TEST_TMP/port" &
SERVER_PID=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [ -s "$TEST_TMP/port" ] && break
  sleep 0.1
done
[ -s "$TEST_TMP/port" ] || { printf 'mock server failed to start\n' >&2; exit 1; }
PORT="$(cat "$TEST_TMP/port")"

GLM53_ENV_FILE="$ROOT_DIR/tests/fixtures/valid.env" \
  "$ROOT_DIR/smoke-glm53.sh" \
  --profile 32k --tools --base-url "http://127.0.0.1:$PORT/v1" >/dev/null

(
  cd "$TEST_TMP"
  "$ROOT_DIR/bench-glm53.py" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model glm-5.3-flash-nvfp4 \
    --prompts "$ROOT_DIR/examples/benchmark-prompts.jsonl" \
    --runs 1 \
    --output "$TEST_TMP/result.json" >/dev/null
)
python3 - "$TEST_TMP/result.json" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
assert len(payload["results"]) == 3
assert all(item["ok"] for item in payload["results"])
assert all(item["completion_tokens"] == 2 for item in payload["results"])
PY

(
  cd "$TEST_TMP"
  "$ROOT_DIR/bench-long-context.py" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model glm-5.3-flash-nvfp4 \
    --target-tokens 1024 \
    --cold \
    --label mock \
    --output "$TEST_TMP/long-result.json" >/dev/null
)
python3 - "$TEST_TMP/long-result.json" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["ok"] is True
assert payload["retrieval_ok"] is True
assert payload["api_healthy_after"] is True
assert payload["raw_message_tokens"] >= 960
PY
printf 'smoke + short/long benchmark clients: OK\n'
