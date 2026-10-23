#!/usr/bin/env python3
"""Download (or locate offline) the pinned snapshot, then validate it."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from validate_checkpoint import validate_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    if len(args.revision) != 40 or any(ch not in "0123456789abcdef" for ch in args.revision):
        print("revision must be a lowercase 40-character commit SHA", file=sys.stderr)
        return 2
    if args.max_workers < 1 or args.max_workers > 16:
        print("max-workers must be between 1 and 16", file=sys.stderr)
        return 2

    token = os.environ.get("HF_TOKEN") or None
    try:
        snapshot = snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            token=token,
            local_files_only=args.local_only,
            max_workers=args.max_workers,
        )
    except Exception as exc:  # huggingface_hub uses several transport exceptions
        mode = "offline lookup" if args.local_only else "download"
        print(f"checkpoint {mode} failed: {exc}", file=sys.stderr)
        return 1

    failures = validate_snapshot(Path(snapshot), args.manifest)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
