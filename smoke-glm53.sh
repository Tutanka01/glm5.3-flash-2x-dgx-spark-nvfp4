#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE=""
TOOLS=0
BASE_URL=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) shift; PROFILE="${1:?--profile requires a value}" ;;
    --tools|--full) TOOLS=1 ;;
    --base-url) shift; BASE_URL="${1:?--base-url requires a value}" ;;
    -h|--help)
      printf 'Usage: %s [--profile NAME] [--tools] [--base-url URL]\n' "$0"
      exit 0
      ;;
    *) glm53_die "Unknown option: $1" ;;
  esac
  shift
done

glm53_load_config "$PROFILE"
glm53_require_command curl
glm53_require_command python3
BASE_URL="${BASE_URL:-http://127.0.0.1:${API_PORT}/v1}"
BASE_URL="${BASE_URL%/}"

SMOKE_TMP="$(mktemp -d /tmp/glm53-smoke.XXXXXX)"
trap 'rm -rf "$SMOKE_TMP"' EXIT

glm53_info "Checking model discovery at $BASE_URL/models"
GLM53_CURL_MAX_TIME=30 glm53_api_curl "$BASE_URL/models" > "$SMOKE_TMP/models.json"
python3 - "$SMOKE_TMP/models.json" "$SERVED_MODEL_NAME" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
ids=[item.get("id") for item in payload.get("data", [])]
if sys.argv[2] not in ids:
    raise SystemExit(f"served model {sys.argv[2]!r} not found; API returned {ids!r}")
print(f"  model id: {sys.argv[2]}")
PY

python3 - "$SMOKE_TMP/chat-request.json" "$SERVED_MODEL_NAME" <<'PY'
import json, sys
payload={
    "model": sys.argv[2],
    "messages": [
        {"role": "system", "content": "Follow the user's output-format instruction exactly."},
        {"role": "user", "content": "Reply with exactly GLM53_OK and nothing else."},
    ],
    "temperature": 0,
    "max_tokens": 64,
}
json.dump(payload, open(sys.argv[1], "w", encoding="utf-8"))
PY

glm53_info "Running deterministic chat smoke test"
GLM53_CURL_MAX_TIME=300 glm53_api_curl \
  -H 'Content-Type: application/json' \
  --data-binary "@$SMOKE_TMP/chat-request.json" \
  "$BASE_URL/chat/completions" > "$SMOKE_TMP/chat-response.json"

python3 - "$SMOKE_TMP/chat-response.json" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
try:
    choice=payload["choices"][0]
    message=choice["message"]
except (KeyError, IndexError, TypeError) as exc:
    raise SystemExit(f"malformed chat response: {exc}; payload={payload!r}")
content=message.get("content") or ""
if "GLM53_OK" not in content:
    reasoning=message.get("reasoning_content") or ""
    raise SystemExit(
        "coherence marker missing from final content; "
        f"content={content!r}, reasoning_tail={reasoning[-300:]!r}"
    )
usage=payload.get("usage", {})
print(f"  final content: {content.strip()!r}")
print(f"  finish reason: {choice.get('finish_reason')}")
print(f"  usage: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
PY

if [ "$TOOLS" = "1" ]; then
  python3 - "$SMOKE_TMP/tool-request.json" "$SERVED_MODEL_NAME" <<'PY'
import json, sys
payload={
    "model": sys.argv[2],
    "messages": [{"role": "user", "content": "Use the tool to get the temperature in Paris."}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_temperature",
            "description": "Get the current temperature for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 256,
}
json.dump(payload, open(sys.argv[1], "w", encoding="utf-8"))
PY
  glm53_info "Running tool-call parser smoke test"
  GLM53_CURL_MAX_TIME=300 glm53_api_curl \
    -H 'Content-Type: application/json' \
    --data-binary "@$SMOKE_TMP/tool-request.json" \
    "$BASE_URL/chat/completions" > "$SMOKE_TMP/tool-response.json"
  python3 - "$SMOKE_TMP/tool-response.json" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], encoding="utf-8"))
message=payload.get("choices", [{}])[0].get("message", {})
calls=message.get("tool_calls") or []
matching=[call for call in calls if call.get("function", {}).get("name") == "get_temperature"]
if not matching:
    raise SystemExit(f"expected get_temperature tool call, got: {calls!r}")
arguments=matching[0].get("function", {}).get("arguments", "")
decoded=json.loads(arguments) if isinstance(arguments, str) else arguments
if not isinstance(decoded, dict):
    raise SystemExit(f"tool arguments are not a JSON object: {decoded!r}")
if "paris" not in str(decoded.get("city", "")).lower():
    raise SystemExit(f"tool call has unexpected arguments: {decoded!r}")
print(f"  tool call: get_temperature({decoded!r})")
PY
fi

if [ "$TOOLS" = "1" ]; then
  glm53_info "Full chat + tool-call smoke test passed"
else
  glm53_info "Basic chat smoke test passed"
fi
