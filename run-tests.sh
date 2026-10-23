#!/usr/bin/env bash
# Runs every no-GPU test suite: shared tools, then each lane.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

for suite in tools/tests/run-local.sh sglang-nvfp4/tests/run-local.sh vllm-exl3/tests/run-local.sh; do
  printf '\n=== %s ===\n' "$suite"
  "$REPO_DIR/$suite"
done

printf '\nAll suites passed.\n'
