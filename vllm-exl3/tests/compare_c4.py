#!/usr/bin/env python3
"""Compare two bench-glm53.py C4 artifacts: MIXED_PREFILL_CHUNK skip vs N.

Reads the artifacts produced by:

    python3 bench-glm53.py --runs 3 --concurrency 4 --thinking off --output <json>

for the baseline policy (`skip`) and the candidate (e.g. `256`), and applies
the promotion criterion from docs/BENCHMARKS.md:

    "p99 TTFT meaningfully below the skip p99 (16.7 s on this kit) without
     giving back the aggregate"

Defaults: --p99-ratio 0.75 (candidate p99 <= 75% of baseline) and
--aggregate-ratio 0.95 (candidate aggregate >= 95% of baseline). The aggregate
uses the artifact's per-target wall clock; artifacts written before
`wall_seconds` was added fall back to max(total_seconds), which underestimates
the wall and is flagged in the output.

Exit codes: 0 = candidate wins both criteria (flip the scheduler default);
3 = p99 improved but the aggregate regressed; 4 = p99 not meaningfully below
(keep `skip`); 2 = malformed/insufficient artifacts. `--self-test` checks the
math on synthetic artifacts and always exits 0/1 itself.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile (matches bench-glm53.py)."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(artifact: dict[str, Any], target: str) -> dict[str, Any]:
    results = [r for r in artifact.get("results", []) if r.get("target") == target]
    ok = [r for r in results if r.get("ok")]
    ttfts = [r["ttft_seconds"] for r in ok if r.get("ttft_seconds") is not None]
    totals = [r["total_seconds"] for r in ok]
    completions = sum(
        r["completion_tokens"] for r in ok if isinstance(r.get("completion_tokens"), int)
    )
    wall = (artifact.get("wall_seconds") or {}).get(target)
    wall_estimated = wall is None
    if wall is None:
        wall = max(totals) if totals else None
    return {
        "attempts": len(results),
        "ok": len(ok),
        "ttft_median": sorted(ttfts)[len(ttfts) // 2] if ttfts else None,
        "ttft_p99": percentile(ttfts, 0.99),
        "total_median": sorted(totals)[len(totals) // 2] if totals else None,
        "completion_tokens": completions,
        "wall_seconds": wall,
        "wall_estimated": wall_estimated,
        "aggregate": completions / wall if wall else None,
    }


def load(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact.get("results"), list):
        raise ValueError(f"{path}: no results[] array")
    return artifact


def compare(
    base: dict[str, Any],
    cand: dict[str, Any],
    target: str,
    p99_ratio_max: float,
    aggregate_ratio_min: float,
) -> tuple[int, str]:
    b, c = summarize(base, target), summarize(cand, target)
    if b["ttft_p99"] is None or c["ttft_p99"] is None or b["ok"] == 0 or c["ok"] == 0:
        return 2, "insufficient successful requests to compare"

    lines = [
        f"baseline (skip)      ok={b['ok']}/{b['attempts']} "
        f"TTFT median={b['ttft_median']:.2f}s p99={b['ttft_p99']:.2f}s "
        f"aggregate={fmt(b['aggregate'])} tok/s "
        f"(wall={'~' if b['wall_estimated'] else ''}{b['wall_seconds']:.1f}s)",
        f"candidate            ok={c['ok']}/{c['attempts']} "
        f"TTFT median={c['ttft_median']:.2f}s p99={c['ttft_p99']:.2f}s "
        f"aggregate={fmt(c['aggregate'])} tok/s "
        f"(wall={'~' if c['wall_estimated'] else ''}{c['wall_seconds']:.1f}s)",
    ]
    p99_ratio = c["ttft_p99"] / b["ttft_p99"]
    lines.append(
        f"p99 ratio cand/base = {p99_ratio:.3f} (want <= {p99_ratio_max}); "
        f"aggregate ratio = {ratio_text(c['aggregate'], b['aggregate'])} "
        f"(want >= {aggregate_ratio_min})"
    )
    markdown = (
        f"| C4 `{target}` | {b['ttft_p99']:.1f} s | {c['ttft_p99']:.1f} s "
        f"({p99_ratio:.0%}) | {fmt(b['aggregate'])} → {fmt(c['aggregate'])} tok/s |"
    )
    lines.append(f"markdown row: {markdown}")

    if b["wall_estimated"] or c["wall_estimated"]:
        lines.append(
            "note: wall clock estimated from max(total_seconds) — re-run the "
            "baseline with a bench-glm53.py that records wall_seconds for an "
            "exact aggregate"
        )

    p99_won = p99_ratio <= p99_ratio_max
    agg_ok = (
        c["aggregate"] is None
        or b["aggregate"] is None
        or (c["aggregate"] / b["aggregate"]) >= aggregate_ratio_min
    )
    if p99_won and agg_ok:
        lines.append("VERDICT: PASS — p99 meaningfully lower, aggregate kept")
        return 0, "\n".join(lines)
    if p99_won:
        lines.append("VERDICT: FAIL — gives back the aggregate; keep `skip`")
        return 3, "\n".join(lines)
    lines.append("VERDICT: FAIL — p99 not meaningfully below the skip baseline")
    return 4, "\n".join(lines)


def fmt(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def ratio_text(cand: float | None, base: float | None) -> str:
    if cand is None or base is None or base == 0:
        return "n/a"
    return f"{cand / base:.3f}"


def self_test() -> int:
    def artifact(ttfts: list[float], completions: list[int], wall: float) -> dict[str, Any]:
        return {
            "wall_seconds": {"local": wall},
            "results": [
                {
                    "target": "local",
                    "ok": True,
                    "ttft_seconds": ttft,
                    "total_seconds": ttft + 2.0,
                    "completion_tokens": tokens,
                }
                for ttft, tokens in zip(ttfts, completions)
            ],
        }

    baseline = artifact([1.0, 2.0, 17.0], [100, 100, 100], 10.0)  # p99 = 17.0, agg 30
    strong = artifact([1.0, 1.5, 4.0], [100, 100, 100], 10.0)     # p99 = 4.0, agg 30
    code, out = compare(baseline, strong, "local", 0.75, 0.95)
    assert code == 0, f"expected PASS, got {code}\n{out}"
    assert "VERDICT: PASS" in out

    regress = artifact([1.0, 1.5, 4.0], [100, 100, 100], 20.0)    # p99 wins, agg halves
    code, out = compare(baseline, regress, "local", 0.75, 0.95)
    assert code == 3, f"expected aggregate-regression (3), got {code}\n{out}"

    slow = artifact([10.0, 12.0, 16.0], [100, 100, 100], 10.0)    # p99 barely moves
    code, out = compare(baseline, slow, "local", 0.75, 0.95)
    assert code == 4, f"expected keep-skip (4), got {code}\n{out}"

    code, _ = compare(baseline, artifact([1.0], [10], 1.0), "ghost", 0.75, 0.95)
    assert code == 2, "expected insufficient-data (2)"
    print("compare_c4 self-test: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, help="skip-policy artifact")
    parser.add_argument("--candidate", type=Path, help="chunk-256 artifact")
    parser.add_argument("--target", default="local", help="target name inside the artifacts")
    parser.add_argument("--p99-ratio", type=float, default=0.75)
    parser.add_argument("--aggregate-ratio", type=float, default=0.95)
    parser.add_argument("--self-test", action="store_true")
    options = parser.parse_args()
    if options.self_test:
        return self_test()
    if not options.baseline or not options.candidate:
        parser.error("--baseline and --candidate are required (or use --self-test)")
    try:
        code, out = compare(
            load(options.baseline),
            load(options.candidate),
            options.target,
            options.p99_ratio,
            options.aggregate_ratio,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"compare_c4 error: {exc}", file=sys.stderr)
        return 2
    print(out)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
