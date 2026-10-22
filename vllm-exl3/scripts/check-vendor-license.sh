#!/usr/bin/env bash
# Licence gate for vendor syncs (policy: root CREDITS.md).
#
# This lane vendors MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks at e25aa70
# (MIT). Upstream relicensed to AGPL-3.0 on 2026-09-07 (cf0f430), and this
# repository is MIT: run this against the ref you are about to sync from.
#
#   scripts/check-vendor-license.sh            # upstream main
#   scripts/check-vendor-license.sh <ref>      # the exact commit to sync
#
# Exit 0 = MIT at that ref (sync allowed), 1 = anything else (licence decision
# needed first), 2 = could not read the licence.
set -euo pipefail

UPSTREAM="${UPSTREAM_REPO:-MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}"
REF="${1:-main}"

first_line="$(curl -fsSL "https://raw.githubusercontent.com/${UPSTREAM}/${REF}/LICENSE" 2>/dev/null \
    | sed -n '/[^[:space:]]/{s/^[[:space:]]*//;p;q;}')" || true
if [ -z "$first_line" ]; then
    echo "[vendor-license] cannot read LICENSE of ${UPSTREAM}@${REF}" >&2
    exit 2
fi

echo "[vendor-license] ${UPSTREAM}@${REF}: ${first_line}"
case "$first_line" in
    "MIT License")
        echo "[vendor-license] MIT: sync allowed into this MIT repository"
        ;;
    *)
        echo "[vendor-license] not MIT: do not sync code from this ref without a licence decision (see CREDITS.md)" >&2
        exit 1
        ;;
esac
