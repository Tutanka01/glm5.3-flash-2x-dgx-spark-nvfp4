#!/usr/bin/env bash
# c4-chunk-ab.sh — promotion checklist item: GLM53_MIXED_PREFILL_CHUNK A/B at C4.
#
# Boots the lane with the candidate scheduler policy (--policy, default 256),
# runs the exact C4 protocol recorded for the skip baseline
# (bench-glm53.py --runs 3 --concurrency 4 --thinking off), then applies the
# promotion criterion via tests/compare_c4.py: candidate p99 TTFT meaningfully
# below the skip p99 without giving back the aggregate.
#
# Run on the HEAD node while nothing else uses the lane. A boot takes several
# minutes (TP=2 JIT + shape warmup); the bench a few more.
#
#   vllm-exl3/scripts/c4-chunk-ab.sh                       # 256 vs recorded skip baseline
#   vllm-exl3/scripts/c4-chunk-ab.sh --policy skip         # re-record the baseline
#   vllm-exl3/scripts/c4-chunk-ab.sh --baseline <json>     # other skip artifact
#   vllm-exl3/scripts/c4-chunk-ab.sh --no-restart          # lane already on --policy
#
# The served model id defaults to SERVED_MODEL_NAME (as in start.sh) or
# GLM-5.3-Flash-EXL3; bench-glm53.py's own default targets the NVFP4 lane and
# would 404 against this one.
#
# Exit code is compare_c4.py's verdict (0 pass / 3 aggregate regression /
# 4 keep skip); 2 = usage or bench failure.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$LANE_DIR/.." && pwd)"

log()  { printf '\033[1;36m[c4-chunk-ab]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[c4-chunk-ab]\033[0m ERROR: %s\n' "$*" >&2; exit 2; }

POLICY=256
RUNS=3
CONCURRENCY=4
THINKING=off
MODEL="${SERVED_MODEL_NAME:-GLM-5.3-Flash-EXL3}"
NO_RESTART=0
BASELINE="$LANE_DIR/results/glm53-benchmark-c4-chunkskip-20260829-173451.json"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --policy)     POLICY="${2:?}"; shift 2 ;;
        --model)      MODEL="${2:?}"; shift 2 ;;
        --runs)       RUNS="${2:?}"; shift 2 ;;
        --concurrency) CONCURRENCY="${2:?}"; shift 2 ;;
        --baseline)   BASELINE="${2:?}"; shift 2 ;;
        --output)     OUTPUT="${2:?}"; shift 2 ;;
        --no-restart) NO_RESTART=1; shift ;;
        *) die "unknown option: $1 (see header)" ;;
    esac
done
case "$POLICY" in
    skip|0|[1-9]*) ;;
    *) die "--policy must be 'skip', 0, or a positive token cap" ;;
esac

[ -n "$OUTPUT" ] || OUTPUT="$LANE_DIR/results/glm53-benchmark-c4-chunk${POLICY}-${STAMP}.json"
mkdir -p "$LANE_DIR/results"

if [ "$NO_RESTART" -eq 0 ]; then
    log "restarting lane with GLM53_MIXED_PREFILL_CHUNK=$POLICY (boot: several minutes)"
    GLM53_MIXED_PREFILL_CHUNK="$POLICY" "$LANE_DIR/start.sh" restart
else
    log "assuming the running lane already uses GLM53_MIXED_PREFILL_CHUNK=$POLICY"
fi

log "C4 protocol: bench-glm53.py --runs $RUNS --concurrency $CONCURRENCY --thinking $THINKING --model $MODEL"
python3 "$REPO_DIR/bench-glm53.py" \
    --model "$MODEL" \
    --runs "$RUNS" --concurrency "$CONCURRENCY" --thinking "$THINKING" \
    --output "$OUTPUT"
log "artifact: $OUTPUT"

if [ ! -f "$BASELINE" ]; then
    die "skip baseline artifact not found: $BASELINE — record it first with --policy skip"
fi
log "comparing against skip baseline: $BASELINE"
set +e
python3 "$LANE_DIR/tests/compare_c4.py" \
    --baseline "$BASELINE" --candidate "$OUTPUT"
VERDICT=$?
set -e

case "$VERDICT" in
    0) log "PASS — flip the lane default: set GLM53_MIXED_PREFILL_CHUNK=$POLICY in .env (or keep exporting it), then re-record the prefix/long-context rows under the new policy" ;;
    3) log "FAIL — p99 improved but the aggregate regressed; restart plain ./start.sh to return to the documented skip policy" ;;
    4) log "FAIL — keep the skip policy; restart plain ./start.sh to return to it" ;;
    *) log "compare_c4 could not produce a verdict (exit $VERDICT)" ;;
esac
exit "$VERDICT"
