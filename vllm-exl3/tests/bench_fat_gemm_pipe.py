#!/usr/bin/env python3
"""GPU bit-exact gate for the opt-in 3-stage cp.async fat-GEMM.

Runs the stock `exl3_fat_gemm` / `_scatter` and the `_pipe` variants on the same
random K4-MCG tensors over a shape matrix, and requires `torch.equal` (the
reference S2b receipt is bit-exact over 56 shapes).

Usage (inside the image, on a GPU node):
  python3 tests/bench_fat_gemm_pipe.py --trials 3 --out results/fat-pipe.json

No-op with exit 0 when CUDA/exllamav3_ext is unavailable, so it is safe to
include in a host test sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _skip(msg: str) -> int:
    print(f"bench_fat_gemm_pipe: SKIP — {msg}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return _skip(f"torch unavailable: {exc!r}")
    if not torch.cuda.is_available():
        return _skip("CUDA unavailable")
    try:
        import exllamav3_ext as ext
    except Exception as exc:  # noqa: BLE001
        return _skip(f"exllamav3_ext unavailable: {exc!r}")
    for sym in ("exl3_fat_gemm", "exl3_fat_gemm_pipe", "exl3_fat_gemm_scatter",
                "exl3_fat_gemm_pipe_scatter"):
        if not hasattr(ext, sym):
            return _skip(f"{sym} missing (image was built without it)")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    shapes = []
    for m in (1, 17, 63, 64, 65, 127, 128, 129, 1024, 3584):
        for k in (16, 128, 1024, 4096):
            for n in (128, 1024, 4096):
                shapes.append((m, k, n))
    # The production TP2 per-rank shape first.
    shapes = [(3584, 4096, 4096)] + shapes
    results = {"ok": True, "trials": args.trials, "shapes": []}
    n_cmp = 0
    for m, k, n in shapes:
        for trial in range(args.trials):
            a = torch.randn((m, k), device=device, dtype=torch.float16)
            packed = torch.randint(
                -30000, 30000, (k // 16, n // 16, 64),
                device=device, dtype=torch.int16,
            )
            svh = (torch.randn((n,), device=device) * 0.5).half()
            out_ref = torch.zeros((m, n), device=device, dtype=torch.float32)
            out_pipe = torch.zeros_like(out_ref)
            ext.exl3_fat_gemm(a, packed, out_ref, svh, 4, True, False)
            ext.exl3_fat_gemm_pipe(a, packed, out_pipe, svh, 4, True, False)
            torch.cuda.synchronize()
            eq = bool(torch.equal(out_ref, out_pipe))
            results["shapes"].append(
                {"m": m, "k": k, "n": n, "trial": trial, "gemm_equal": eq}
            )
            n_cmp += 1
            if not eq:
                results["ok"] = False
                print(f"FAIL gemm m={m} k={k} n={n} trial={trial}")
                break

            # Scatter: a permutation of token ids and random route weights.
            token_idx = torch.randperm(m, device=device, dtype=torch.int64)
            weight = torch.rand((m,), device=device, dtype=torch.float16)
            out_ref.zero_()
            out_pipe.zero_()
            ext.exl3_fat_gemm_scatter(
                a, packed, out_ref, svh, token_idx, weight, 4, True, False
            )
            ext.exl3_fat_gemm_pipe_scatter(
                a, packed, out_pipe, svh, token_idx, weight, 4, True, False
            )
            torch.cuda.synchronize()
            eq = bool(torch.equal(out_ref, out_pipe))
            results["shapes"].append(
                {"m": m, "k": k, "n": n, "trial": trial, "scatter_equal": eq}
            )
            n_cmp += 1
            if not eq:
                results["ok"] = False
                print(f"FAIL scatter m={m} k={k} n={n} trial={trial}")
                break
        if not results["ok"]:
            break

    results["comparisons"] = n_cmp
    print(
        f"bench_fat_gemm_pipe: {'PASS' if results['ok'] else 'FAIL'} "
        f"({n_cmp} comparisons, {len(results['shapes'])} rows)"
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.out}")
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
