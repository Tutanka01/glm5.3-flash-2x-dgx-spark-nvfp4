#!/usr/bin/env bash
# Local (no-GPU) test suite for the vllm-exl3 lane.
#
# Validates shell syntax, the fragile text contracts of start.sh, the overlay
# patch unit tests, and the bench clients against a mock OpenAI server.
# Nothing here downloads the checkpoint or requires docker/GPU — hardware
# validation is the on-cluster protocol in ../vllm-exl3/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."

pass=0 fail=0
step() { printf '[vllm-exl3-tests] %s\n' "$*"; }
ok()   { pass=$((pass + 1)); printf '[vllm-exl3-tests]   OK  %s\n' "$1"; }
bad()  { fail=$((fail + 1)); printf '[vllm-exl3-tests]  FAIL %s\n' "$1"; }

# 1. shell syntax
if bash -n start.sh && bash -n stop.sh && bash -n download.sh \
   && bash -n scripts/boot-shape-warmup.sh; then ok "bash -n (start/stop/download/warmup)"; else bad "bash -n"; fi

# 2. python compile of everything importable
if python3 -m py_compile tests/bench_decode.py tests/bench_prefix_cache.py \
    tests/bench_long_context.py tests/soak_tool_calls.py tests/compare_c4.py \
    tests/grade_ab_quality.py tests/test_*.py overlay/patch_*.py 2>/dev/null; then
    ok "py_compile (benches, soak, comparators, tests, overlay patches)"
else
    bad "py_compile"
fi

# 3. fragile start.sh contracts (keep these green when editing start.sh)
c1=$(grep -c "\$(python3 -S -c 'import json,os" start.sh || true)
c2=$(grep -c "python3 /opt/glm53/patch_xgrammar_termination.py" start.sh || true)
c3=$(grep -c 'XGRAMMAR_PATCH_HOST="${XGRAMMAR_PATCH_HOST:-' start.sh || true)
if [ "$c1" = "2" ] && [ "$c2" = "2" ] && [ "$c3" = "1" ]; then
    ok "start.sh text contracts (spec heredocs x2, xgrammar wiring)"
else
    bad "start.sh text contracts changed: heredoc=$c1 (want 2), xgrammar-run=$c2 (want 2), xgrammar-env=$c3 (want 1)"
fi

# 4. per-rank GID wiring (PR #26) must reach both docker runs exactly once
h=$(grep -c 'NCCL_IB_GID_INDEX=$HEAD_GID' start.sh || true)
w=$(grep -c "NCCL_IB_GID_INDEX='\$WORKER_GID'" start.sh || true)
if [ "$h" = "1" ] && [ "$w" = "1" ]; then ok "per-rank GID wiring (head+worker)"; else bad "per-rank GID wiring h=$h w=$w"; fi

# 5. vendor pure-python tests
for t in test_start_overrides test_warm_restart_stdout test_xgrammar_termination test_suppress_stops; do
    if python3 "tests/${t}.py" >/dev/null 2>&1; then ok "$t"; else bad "$t"; fi
done

# 4b. comparator/grader self-tests (math + deterministic graders, no server)
if python3 tests/compare_c4.py --self-test >/dev/null 2>&1; then ok "compare_c4 self-test"; else bad "compare_c4 self-test"; fi
if python3 tests/grade_ab_quality.py --self-test >/dev/null 2>&1; then ok "grade_ab_quality self-test"; else bad "grade_ab_quality self-test"; fi
# needs jinja2; self-skips loudly if unavailable
if python3 tests/test_chat_template.py >/dev/null 2>&1; then
    ok "test_chat_template"
else
    step "test_chat_template skipped (jinja2 missing?)"
fi

# 6. bench clients against the mock OpenAI server
MOCK_PORT_FILE="$(mktemp)"
python3 ../tests/mock_openai_server.py --port 0 --port-file "$MOCK_PORT_FILE" &
MOCK_PID=$!
trap '{ kill "$MOCK_PID" 2>/dev/null || true; wait "$MOCK_PID" 2>/dev/null || true; } 2>/dev/null; { kill "${BLANK_PID:-}" 2>/dev/null || true; wait "${BLANK_PID:-}" 2>/dev/null || true; } 2>/dev/null; rm -f "$MOCK_PORT_FILE" "$BLANK_PORT_FILE"' EXIT
for _ in $(seq 1 20); do
    [ -s "$MOCK_PORT_FILE" ] && break
    sleep 0.2
done
if [ -s "$MOCK_PORT_FILE" ]; then
    MOCK_PORT="$(cat "$MOCK_PORT_FILE")"
    TMP_OUT="$(mktemp -d)"
    if python3 tests/bench_prefix_cache.py --base-url "http://127.0.0.1:${MOCK_PORT}/v1" \
        --model glm-5.3-flash-nvfp4 --runs 1 --prompt-tokens 256 \
        --output "$TMP_OUT/prefix.json" >/dev/null 2>&1; then
        ok "bench_prefix_cache against mock"
    else
        bad "bench_prefix_cache against mock"
    fi
    # clean failure when nothing listens (no traceback)
    if python3 tests/bench_prefix_cache.py --base-url http://127.0.0.1:1/v1 \
        --runs 1 --prompt-tokens 64 --output "$TMP_OUT/none.json" >/dev/null 2>&1; then
        bad "bench_prefix_cache should exit 1 when the server is unreachable"
    else
        ok "bench_prefix_cache clean failure path"
    fi

    # 6b. tool-call soak against the mock (issue #10 mitigation rehearsal)
    if python3 tests/soak_tool_calls.py --base-url "http://127.0.0.1:${MOCK_PORT}/v1" \
        --model glm-5.3-flash-nvfp4 --agents 3 --turns 4 --filler-words 50 \
        --out "$TMP_OUT/soak-clean.json" >/dev/null 2>&1 \
        && python3 - "$TMP_OUT/soak-clean.json" <<-'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["turns_ok"] == 12 and d["blank_arg_events"] == 0
PY
    then
        ok "soak_tool_calls validation (clean mock, 3×4 concurrent turns)"
    else
        bad "soak_tool_calls validation"
    fi

    # 6c. blank-args injection: client retry must recover every turn. One
    # agent keeps the counter parity deterministic under interleaving.
    BLANK_PORT_FILE="$(mktemp)"
    python3 ../tests/mock_openai_server.py --port 0 --port-file "$BLANK_PORT_FILE" \
        --blank-tool-args-every 2 &
    BLANK_PID=$!
    for _ in $(seq 1 20); do [ -s "$BLANK_PORT_FILE" ] && break; sleep 0.2; done
    if [ -s "$BLANK_PORT_FILE" ]; then
        BLANK_PORT="$(cat "$BLANK_PORT_FILE")"
        if python3 tests/soak_tool_calls.py --base-url "http://127.0.0.1:${BLANK_PORT}/v1" \
            --model glm-5.3-flash-nvfp4 --agents 1 --turns 4 --filler-words 50 \
            --retries 2 --out "$TMP_OUT/soak-blank.json" >/dev/null 2>&1 \
            && python3 - "$TMP_OUT/soak-blank.json" <<-'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["blank_arg_events"] == 3 and d["recovered_after_retry"] == 3
assert d["turns_ok"] == 4 and d["invalid_args_unrecovered"] == 0
PY
        then
            ok "soak_tool_calls retry path (blank args injected, all recovered)"
        else
            bad "soak_tool_calls retry path"
        fi
        { kill "$BLANK_PID" 2>/dev/null || true; wait "$BLANK_PID" 2>/dev/null || true; } 2>/dev/null
    else
        bad "blank-args mock did not start"
    fi
    rm -f "$BLANK_PORT_FILE"

    rm -rf "$TMP_OUT"
else
    bad "mock server did not start"
fi

printf '[vllm-exl3-tests] %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
