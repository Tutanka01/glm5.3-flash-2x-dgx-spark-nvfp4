#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

for file in ./*.sh ./scripts/*.sh ./tests/*.sh ./tests/mock-start-bin/*; do
  bash -n "$file"
done

PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  tests/test_validate_checkpoint.py tests/test_bench_long_context.py

for profile in 32k 32k-batch4 32k-batch8 64k 128k 128k-batch4 128k-batch4-mtp 128k-batch4-mtp3 128k-batch2-mtp 128k-batch4-8k 128k-batch8 128k-ep1 128k-mtp-ep1 128k-mtp-compile 256k 256k-graphs 256k-mtp 384k-quality 512k-mtp-eager 512k-mtp-cp 32k-mtp 32k-eager; do
  GLM53_ENV_FILE=tests/fixtures/valid.env scripts/validate-env.sh "$profile" >/dev/null
done

./tests/test-compose-command.sh
./tests/test-start-stop.sh
./tests/test-api-clients.sh

python3 -c '
import json
json.load(open("metadata/checkpoint-manifest.json", encoding="utf-8"))
json.load(open("examples/opencode.json", encoding="utf-8"))
[json.loads(line) for line in open("examples/benchmark-prompts.jsonl", encoding="utf-8") if line.strip()]
'

GLM53_ENV_FILE=tests/fixtures/valid.env docker compose \
  --env-file tests/fixtures/valid.env \
  --env-file profiles/32k.env \
  -f docker-compose.glm53.yml config --quiet

printf 'All local recipe tests passed.\n'
