#!/usr/bin/env python3
"""Download or locate the pinned DFlash2 draft and validate its geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


EXPECTED = {
    "architectures": ["DFlash2DraftModel"],
    "dtype": "bfloat16",
}
EXPECTED_DFLASH = {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 14, 24, 33, 42],
}
EXPECTED_WEIGHT_SIZE = 2_342_169_800
EXPECTED_WEIGHT_SHA256 = "8931dc522be0aa31760a7463f8d2f8044fa3e6d40be2e87aa08e9fd17bfd6683"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("--revision must be a lowercase 40-character SHA")
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=args.model_id,
                revision=args.revision,
                token=os.environ.get("HF_TOKEN") or None,
                local_files_only=args.local_only,
                max_workers=args.max_workers,
            )
        )
    except Exception as exc:
        print(f"DFlash2 snapshot preparation failed: {exc}", file=sys.stderr)
        return 1

    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        print("DFlash2 snapshot is missing config.json or model.safetensors", file=sys.stderr)
        return 1
    config = json.loads(config_path.read_text(encoding="utf-8"))
    failures = [
        name
        for name, expected in EXPECTED.items()
        if config.get(name) != expected
    ]
    dflash = config.get("dflash_config", {})
    failures.extend(
        f"dflash_config.{name}"
        for name, expected in EXPECTED_DFLASH.items()
        if dflash.get(name) != expected
    )
    if weights_path.stat().st_size != EXPECTED_WEIGHT_SIZE:
        failures.append("model.safetensors size")
    elif sha256_file(weights_path) != EXPECTED_WEIGHT_SHA256:
        failures.append("model.safetensors sha256")
    if failures:
        print("DFlash2 validation failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    print(f"DFlash2 draft ready: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
