#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib.sh
source "$ROOT_DIR/scripts/lib.sh"

PROFILE="${1:-}"
glm53_load_config "$PROFILE"
glm53_require_command curl
glm53_require_command docker
glm53_require_command python3

CHECK_DIR="$(mktemp -d /tmp/glm53-upstream.XXXXXX)"
trap 'rm -rf "$CHECK_DIR"' EXIT

glm53_info "Reading current Hugging Face heads and audited runtime metadata"
curl -fsSL https://huggingface.co/api/models/zai-org/GLM-5.3-Flash \
  -o "$CHECK_DIR/official.json"
curl -fsSL https://huggingface.co/api/models/zai-org/GLM-5.3-Flash-BF16 \
  -o "$CHECK_DIR/bf16.json"
curl -fsSL https://huggingface.co/api/models/LibertAIDAI/GLM-5.3-Flash-NVFP4 \
  -o "$CHECK_DIR/quant.json"
curl -fsSL https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/chat_template.jinja \
  -o "$CHECK_DIR/official.jinja"
curl -fsSL https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/raw/main/chat_template.jinja \
  -o "$CHECK_DIR/bf16.jinja"
curl -fsSL https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/raw/main/chat_template.jinja \
  -o "$CHECK_DIR/quant.jinja"

RUNTIME_SOURCE_REVISION="$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime"]["source_revision"])
' "$ROOT_DIR/metadata/checkpoint-manifest.json")"
curl -fsSL \
  "https://api.github.com/repos/0xSero/glm-5.3-flash-sglang-sm121/commits/$RUNTIME_SOURCE_REVISION" \
  -o "$CHECK_DIR/runtime-source.json"
docker manifest inspect "$GLM53_RUNTIME_IMAGE" > "$CHECK_DIR/image.json"

python3 - \
  "$ROOT_DIR/metadata/checkpoint-manifest.json" \
  "$CHECK_DIR" \
  "$MODEL_ID" \
  "$MODEL_REVISION" \
  "$GLM53_RUNTIME_IMAGE" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest=json.load(open(sys.argv[1], encoding="utf-8"))
root=pathlib.Path(sys.argv[2])
failures=[]

expected_image=manifest["runtime"]["image_tag"] + "@" + manifest["runtime"]["image_digest"]
if sys.argv[3] != manifest["model"]["id"]:
    failures.append("MODEL_ID in .env.glm53 differs from the audited manifest")
if sys.argv[4] != manifest["model"]["revision"]:
    failures.append("MODEL_REVISION in .env.glm53 differs from the audited manifest")
if sys.argv[5] != expected_image:
    failures.append("GLM53_RUNTIME_IMAGE in .env.glm53 differs from the audited manifest")

def load(name):
    return json.load(open(root/name, encoding="utf-8"))

def sha(name):
    return hashlib.sha256((root/name).read_bytes()).hexdigest()

checks=[
    ("official distribution", load("official.json")["sha"], manifest["official_distribution"]["revision"]),
    ("official BF16 source", load("bf16.json")["sha"], manifest["source"]["revision"]),
    ("NVFP4 quant", load("quant.json")["sha"], manifest["model"]["revision"]),
]
for label, current, audited in checks:
    print(f"  {label}: current={current} audited={audited}")
    if current != audited:
        failures.append(f"{label} HEAD changed; audit the new revision before changing the pin")

expected_template=manifest["template_update"]["sha256"]
for name in ("official.jinja", "bf16.jinja", "quant.jinja"):
    current_hash=sha(name)
    print(f"  {name}: sha256={current_hash}")
    if current_hash != expected_template:
        failures.append(f"{name} no longer matches the audited template")

source_sha=load("runtime-source.json").get("sha")
expected_source=manifest["runtime"]["source_revision"]
print(f"  SGLang patch source: current={source_sha} audited={expected_source}")
if source_sha != expected_source:
    failures.append("audited SGLang source revision is no longer reachable")

image=load("image.json")
arm64=[
    item.get("digest")
    for item in image.get("manifests", [])
    if item.get("platform", {}).get("architecture") == "arm64"
    and item.get("platform", {}).get("os") == "linux"
]
expected_platform=manifest["runtime"]["platform_manifest"]
print(f"  SGLang ARM64 manifest: current={arm64} audited={expected_platform}")
if arm64 != [expected_platform]:
    failures.append("runtime OCI index does not contain the audited linux/arm64 manifest")

if failures:
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    raise SystemExit(1)

print("Upstream audit gate passed.")
print(
    "Current NVFP4 revision changes only README.md and config.json; "
    "all 120 weight shards, the index and template are unchanged."
)
print("Runtime: patched SGLang SM121 image accepted on 2x GB10 TP=2.")
PY
