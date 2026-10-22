#!/usr/bin/env python3
"""ablit/fetch_transplant.py offline contracts (no torch, no network).

- parse_layers matches the runtime grammar and the published default skips L44;
- --check passes only for a complete manifest of the pinned donor revision
  (stale revision, missing or short tensor -> exit 1, so start.sh fetches);
- manifest keys are strings, as JSON round-trips them (the resume path once
  compared int keys against a reloaded manifest and never skipped anything).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "ablit" / "fetch_transplant.py"


def load(env: dict[str, str]):
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location("fetch_transplant", SRC)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_check(mod, layers) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.check(layers)


def write_tree(out: Path, donor: str, revision: str, layers: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for L in layers:
        raw = bytes([L]) * 64
        (out / f"L{L}.bin").write_bytes(raw)
        entries[str(L)] = {"nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                           "dtype": "BF16", "shape": [8, 4]}
    (out / "MANIFEST.json").write_text(json.dumps(
        {"donor": donor, "donor_revision": revision, "layers": entries}))


def main() -> int:
    mod = load({"ABLIT_LAYERS": "", "ABLIT_DONOR": "", "ABLIT_DONOR_REVISION": ""})
    default = mod.parse_layers(mod.LAYERS_SPEC)
    assert default == list(range(15, 44)) + [45], default
    assert len(mod.REVISION) == 40 and f"/resolve/{mod.REVISION}" in mod.BASE
    assert mod.parse_layers("15,17-19") == [15, 17, 18, 19]
    for bad in ("45-15", "abc", ""):
        try:
            mod.parse_layers(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"parse_layers accepted {bad!r}")

    with tempfile.TemporaryDirectory() as tmp:
        mod.OUT_DIR = Path(tmp) / "transplant"
        assert run_check(mod, default) == 1  # nothing fetched yet

        write_tree(mod.OUT_DIR, mod.DONOR, mod.REVISION, default)
        assert run_check(mod, default) == 0
        assert mod.load_manifest(mod.OUT_DIR / "MANIFEST.json") is not None

        # legacy manifest (no revision) or another revision -> stale -> refetch
        write_tree(mod.OUT_DIR, mod.DONOR, "0" * 40, default)
        assert run_check(mod, default) == 1
        assert mod.load_manifest(mod.OUT_DIR / "MANIFEST.json") is None

        # short tensor -> incomplete
        write_tree(mod.OUT_DIR, mod.DONOR, mod.REVISION, default)
        (mod.OUT_DIR / "L30.bin").write_bytes(b"\0" * 10)
        assert run_check(mod, default) == 1

        # a tree fetched for the default does not cover an explicit L44 request
        write_tree(mod.OUT_DIR, mod.DONOR, mod.REVISION, default)
        assert run_check(mod, mod.parse_layers("15-45")) == 1

    print("ablit fetch contracts OK (default L15-43+45, pinned revision, --check, stale manifest)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
