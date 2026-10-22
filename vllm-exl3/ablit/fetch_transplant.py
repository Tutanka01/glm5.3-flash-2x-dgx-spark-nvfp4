#!/usr/bin/env python3
"""Fetch the abliterated o_proj tensors (L15-43 + MTP 45) from the Dealign donor.

Pulls only the `self_attn.o_proj.weight` byte ranges out of the donor's 120
safetensors shards via HTTP range-requests (~2.6 GiB instead of ~164 GiB),
into ablit/transplant/L{NN}.bin + MANIFEST.json. The donor is pinned to a
Hub commit so every kit transplants the same bytes. Resumable: files whose
sha256 matches the manifest of the same donor revision are skipped.

Run on the head (needs `python3` + network; no torch). start.sh runs it
automatically when ABLIT=1 (ABLIT_AUTO_FETCH=1):
    python3 ablit/fetch_transplant.py           # fetch / resume
    python3 ablit/fetch_transplant.py --check   # offline: exit 0 if complete
    # token: HF_TOKEN env or ~/.cache/huggingface/token (donor is public; the
    # token only helps with rate limits)

The ABLIT runtime (overlay/ablit_runtime.py) picks these up when
ABLIT_METHOD=auto|transplant and replaces the stock o_proj weights at load —
the published "altered Dealign" edit (METHOD.md of
drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-43-mtp-l45: L0-14 and L44 stock).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "ablit" / "transplant"
LAYER_MAP_PATH = REPO_ROOT / "ablit" / "LAYER_MAP.json"

DONOR = os.environ.get("ABLIT_DONOR") or "dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4"
# Donor commit of 2026-09-19; its o_proj bytes are unchanged since the
# 2026-08-29 shard fixes (later commits touch only the card/template).
REVISION = (os.environ.get("ABLIT_DONOR_REVISION")
            or "745aac2ff0f10acf961f396df3f9418598aa7327")
LAYERS_SPEC = os.environ.get("ABLIT_LAYERS") or "15-43,45"
BASE = f"https://huggingface.co/{DONOR}/resolve/{REVISION}"
TOKEN = os.environ.get("HF_TOKEN") or ""
if not TOKEN:
    tok_file = Path.home() / ".cache/huggingface/token"
    if tok_file.is_file():
        TOKEN = tok_file.read_text().strip()

CHUNK = 8 * 1024 * 1024
RETRIES = 4


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def http(url: str, headers: dict | None = None, start: int | None = None, end: int | None = None):
    req = urllib.request.Request(url, headers=dict(headers or {}))
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end if end is not None else ''}")
    return urllib.request.urlopen(req, timeout=120)


def fetch_bytes(url: str, start: int, end: int, headers: dict, what: str) -> bytes:
    """Fetch [start, end] inclusive with retries + resume within the range."""
    got = 0
    total = end - start + 1
    for attempt in range(RETRIES):
        try:
            out = bytearray()
            pos = start + got
            while pos <= end:
                rsp = http(url, headers, pos, end)
                if rsp.status not in (200, 206):
                    raise RuntimeError(f"HTTP {rsp.status}")
                while True:
                    chunk = rsp.read(CHUNK)
                    if not chunk:
                        break
                    out += chunk
                    pos += len(chunk)
                    got = len(out)
                if len(out) == total:
                    return bytes(out)
                rsp.close()
            raise RuntimeError(f"short read {len(out)}/{total}")
        except Exception as exc:  # noqa: BLE001
            wait = 3 * (attempt + 1)
            print(f"  retry {attempt + 1}/{RETRIES} for {what} at {got}/{total} "
                  f"({exc}) — sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"FAILED: {what}")


def safetensors_span(shard_url: str, headers: dict, key: str) -> tuple[int, int, dict]:
    """Resolve (abs_start, abs_end, meta) for one tensor via its header."""
    with http(shard_url, headers, 0, 7) as rsp:
        n = int.from_bytes(rsp.read(8), "little")
    hdr_bytes = fetch_bytes(shard_url, 8, 8 + n - 1, headers, f"header of {shard_url.rsplit('/', 1)[-1]}")
    header = json.loads(hdr_bytes)
    if key not in header:
        raise SystemExit(f"{key} not in {shard_url} header (keys={len(header)})")
    meta = header[key]
    a, b = meta["data_offsets"]
    return 8 + n + a, 8 + n + b - 1, meta


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_layers(spec: str) -> list[int]:
    """Same grammar as ablit_runtime.parse_layers ("15-43,45"), torch-free."""
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        try:
            a, b = int(lo), int(hi or lo)
        except ValueError:
            raise SystemExit(f"ABLIT_LAYERS: bad entry {part!r}")
        if b < a:
            raise SystemExit(f"ABLIT_LAYERS: inverted range {part!r}")
        layers.update(range(a, b + 1))
    if not layers:
        raise SystemExit("ABLIT_LAYERS resolved to an empty set")
    return sorted(layers)


def load_manifest(path: Path) -> dict | None:
    """The manifest of this donor revision, or None when absent or stale."""
    if not path.is_file():
        return None
    old = json.loads(path.read_text())
    if old.get("donor") != DONOR or old.get("donor_revision") != REVISION:
        return None
    return old


def check(layers: list[int]) -> int:
    """Offline completeness check (sizes only; ablit_runtime verifies sha256)."""
    manifest = load_manifest(OUT_DIR / "MANIFEST.json")
    if manifest is None:
        print(f"ablit transplant: no manifest for {DONOR}@{REVISION[:12]}")
        return 1
    have = manifest.get("layers", {})
    missing = [L for L in layers
               if str(L) not in have
               or not (OUT_DIR / f"L{L}.bin").is_file()
               or (OUT_DIR / f"L{L}.bin").stat().st_size != have[str(L)]["nbytes"]]
    if missing:
        print(f"ablit transplant: incomplete, missing/short layers {missing}")
        return 1
    total = sum(have[str(L)]["nbytes"] for L in layers)
    print(f"ablit transplant: complete ({len(layers)} tensors, {total / 1e9:.2f} GB, "
          f"{DONOR}@{REVISION[:12]})")
    return 0


def main() -> int:
    lmap = json.loads(LAYER_MAP_PATH.read_text())
    layers = parse_layers(LAYERS_SPEC)
    if "--check" in sys.argv[1:]:
        return check(layers)
    print(f"donor : {DONOR}@{REVISION}")
    print(f"layers: {LAYERS_SPEC} ({len(layers)} tensors)")

    # authoritative shard map from the donor's own index
    idx_url = f"{BASE}/model.safetensors.index.json"
    for attempt in range(RETRIES):
        try:
            with http(idx_url, auth()) as rsp:
                idx = json.loads(rsp.read())["weight_map"]
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == RETRIES - 1:
                raise SystemExit(f"cannot fetch index: {exc}")
            time.sleep(3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "MANIFEST.json"
    manifest = load_manifest(manifest_path)
    if manifest is not None:
        print("resuming: manifest of this donor revision found")
    else:
        manifest = {
            "donor": DONOR,
            "donor_revision": REVISION,
            "method": "altered-dealign-oproj-transplant (byte-copy, o_proj)",
            "layers": {},
        }

    for L in layers:
        key = f"model.language_model.layers.{L}.self_attn.o_proj.weight"
        shard = idx.get(key)
        if shard is None:
            raise SystemExit(f"donor index has no {key}")
        lmap_shard = next((e["shard"] for e in lmap["layers"] if e["layer"] == L), None)
        if lmap_shard and lmap_shard != shard:
            print(f"  note: L{L} shard differs from LAYER_MAP ({shard} vs {lmap_shard}) — using donor index")
        out = OUT_DIR / f"L{L}.bin"
        if out.is_file() and str(L) in manifest["layers"] \
                and manifest["layers"][str(L)].get("sha256") == sha256(out.read_bytes()):
            print(f"L{L}: already fetched ({out.stat().st_size / 1e6:.0f} MB)")
            continue

        shard_url = f"{BASE}/{shard}"
        print(f"L{L}: {key} @ {shard}")
        start, end, meta = safetensors_span(shard_url, auth(), key)
        nbytes = end - start + 1
        print(f"  {meta['dtype']} {meta['shape']} = {nbytes / 1e6:.0f} MB")
        t0 = time.time()
        data = fetch_bytes(shard_url, start, end, auth(), key)
        dt = time.time() - t0
        print(f"  fetched in {dt:.1f}s ({nbytes / 1e6 / max(dt, 1e-6):.0f} MB/s)")
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(out)
        manifest["layers"][str(L)] = {
            "shard": shard,
            "key": key,
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "nbytes": nbytes,
            "sha256": sha256(data),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    ok = sum(1 for L in layers if str(L) in manifest["layers"])
    total = sum(manifest["layers"][str(L)]["nbytes"] for L in layers
                if str(L) in manifest["layers"])
    print(f"done: {ok}/{len(layers)} tensors, {total / 1e9:.2f} GB -> {OUT_DIR}")
    return 0 if ok == len(layers) else 1


if __name__ == "__main__":
    sys.exit(main())
