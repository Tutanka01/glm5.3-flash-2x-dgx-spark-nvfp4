#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE=""
DRAFT_DIR=""

usage() {
  cat <<'EOF'
Usage: ./scripts/check-dflash2-runtime.sh [--image IMAGE] [--draft-dir PATH]

Read-only gate for a candidate SGLang DFlash2 runtime. It never pulls an image.
The draft directory, when supplied, must be a pinned local Hugging Face snapshot.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --image)
      [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      IMAGE="$2"
      shift
      ;;
    --draft-dir)
      [ "$#" -ge 2 ] || { usage >&2; exit 2; }
      DRAFT_DIR="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [ -z "$IMAGE" ]; then
  IMAGE="$(python3 - "$ROOT_DIR/metadata/checkpoint-manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
runtime = manifest["runtime"]
print(runtime["image_tag"] + "@" + runtime["image_digest"])
PY
)"
fi

command -v docker >/dev/null 2>&1 || {
  printf 'DFLASH2 readiness: BLOCKED (docker is unavailable)\n' >&2
  exit 1
}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  printf 'DFLASH2 readiness: BLOCKED\n' >&2
  printf '  image is not present locally (it was not pulled): %s\n' "$IMAGE" >&2
  exit 1
fi

set +e
RUNTIME_REPORT="$(docker run --rm -i --entrypoint python3 "$IMAGE" - <<'PY'
import importlib.util
import json
import pathlib
import sys

spec = importlib.util.find_spec("sglang")
if spec is None or not spec.submodule_search_locations:
    print(json.dumps({"ready": False, "blockers": ["sglang package not found"]}))
    raise SystemExit(1)

root = pathlib.Path(next(iter(spec.submodule_search_locations)))
files = []
for path in root.rglob("*.py"):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    files.append((path, text))

generic_dflash = any("DFLASH" in text and "speculative" in text.lower() for _, text in files)
fa4 = any("fa4" in text.lower() and "attention" in text.lower() for _, text in files)
glm_capture = []
for path, text in files:
    if "set_dflash_layers_to_capture" in text and (
        "Glm5Next" in text or "glm5_next" in str(path).lower()
    ):
        glm_capture.append((path, text))
mhc_contract = any(
    "hc_contract" in text or "hc_mult" in text or "mean(dim=" in text
    for _, text in glm_capture
)

checks = {
    "generic_dflash_algorithm": generic_dflash,
    "glm5_target_capture_hook": bool(glm_capture),
    "glm5_mhc_contraction": mhc_contract,
    "fa4_draft_attention": fa4,
}
blockers = [name for name, passed in checks.items() if not passed]
report = {
    "ready": not blockers,
    "sglang_root": str(root),
    "checks": checks,
    "glm_capture_files": [str(path.relative_to(root)) for path, _ in glm_capture],
    "blockers": blockers,
}
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if not blockers else 1)
PY
)"
RUNTIME_STATUS=$?
set -e

printf '%s\n' "$RUNTIME_REPORT"
if [ "$RUNTIME_STATUS" -ne 0 ]; then
  printf 'DFLASH2 readiness: BLOCKED — this image must not receive DFLASH flags.\n' >&2
  exit 1
fi

if [ -n "$DRAFT_DIR" ]; then
  python3 - "$DRAFT_DIR/config.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"DFLASH2 readiness: BLOCKED (missing {path})")
config = json.load(path.open(encoding="utf-8"))
dflash = config.get("dflash_config", {})
expected_layers = [5, 14, 24, 33, 42]
checks = {
    "architecture": config.get("architectures") == ["DFlash2DraftModel"],
    "block_size": dflash.get("block_size") == 8,
    "target_layer_ids": dflash.get("target_layer_ids") == expected_layers,
    "dtype": config.get("dtype") == "bfloat16",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("DFLASH2 readiness: BLOCKED (draft config: " + ", ".join(failed) + ")")
print(f"Draft config: PASS ({path.parent})")
PY
else
  printf 'Draft config: NOT CHECKED (pass --draft-dir with the pinned snapshot)\n' >&2
fi

printf 'DFLASH2 runtime capability: PASS\n'
printf 'Promotion still requires the cold/context/concurrency gates in docs/DFLASH2.md.\n'
