#!/usr/bin/env bash
# No-GPU tests for the lane-agnostic tools (bench clients, power sampler).
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TOOLS_DIR"

bash -n tests/run-local.sh
python3 -m py_compile bench-glm53.py bench-power.py tests/mock_openai_server.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests/test_bench_power.py
python3 -c '
import json
[json.loads(line) for line in open("benchmark-prompts.jsonl", encoding="utf-8") if line.strip()]
'

printf 'All tools tests passed.\n'
