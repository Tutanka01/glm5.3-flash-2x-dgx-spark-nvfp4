#!/usr/bin/env python3
"""Prefix-cache reuse bench for the GLM-5.3-Flash EXL3 vLLM lane (2x DGX Spark).

Protocol (mirrors the upstream MiaAI-Lab kit's prefix-caching table):

  For each independent chat (distinct filler content):
    turn 1 (cold): system + long document + question  -> measures cold TTFT
    turn 2 (warm): same history + a follow-up question. The OpenAI API is
        stateless, so the client resends the full history and vLLM hashes the
        shared prefix. Measures warm TTFT and, when the server exposes
        /metrics, the prefix-cache hit ratio from the
        vllm:prefix_cache_{hits,queries} counter deltas around the turn.

  The OpenAI-side speedup is the headline number (upstream: 9.7 s -> 1.17 s
  TTFT on a ~7.7k-token follow-up); the hit ratio confirms the tokens were
  actually reused rather than the scheduler getting lucky.

Usage:
  python3 tests/bench_prefix_cache.py --runs 3 --prompt-tokens 7680
  python3 tests/bench_prefix_cache.py --concurrent 4 --runs 3   # isolated chats

Stdlib only. Exit 0 when every request succeeds (use --min-hit-ratio to make
a low hit ratio fatal; the check is skipped when /metrics is unavailable).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "http://127.0.0.1:8888/v1"
MODEL = "GLM-5.3-Flash-EXL3"
API_KEY_ENV = "API_KEY"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ~4 chars per token for English filler; the reported prompt_tokens comes from
# the server usage block, so calibration only needs to be roughly right.
_CHARS_PER_TOKEN = 4

_DOC_SENTENCE = (
    "The document states that the {keyword} protocol requires operators to "
    "log the checksum {code} before every deployment window. "
)

_SYSTEM = (
    "You are a careful analyst. Answer only from the supplied document. "
    "Keep answers to a single short sentence."
)

_DOC_INTRO = (
    "Below is a maintenance log for a two-node inference cluster. Read it "
    "carefully; questions will follow.\n\n"
)

_QUESTION = (
    "According to the document, which checksum must operators log before "
    "every deployment window? Reply with the code only."
)

_FOLLOWUP = (
    "Keeping the same document in mind, restate the checksum requirement in "
    "your own words in one short sentence."
)

_METRICS_HITS_RE = re.compile(r"^vllm:prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+(\S+)", re.M)
_METRICS_QUERIES_RE = re.compile(r"^vllm:prefix_cache_queries(?:_total)?(?:\{[^}]*\})?\s+(\S+)", re.M)


def _http_json(url: str, payload: dict | None, api_key: str, timeout: float) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:500]
    except (urllib.error.URLError, OSError) as exc:
        return 0, f"connection error: {exc}"
    try:
        return 200, json.loads(body)
    except json.JSONDecodeError:
        return 200, body


def _build_document(chat_index: int, target_tokens: int) -> str:
    keyword = f"NODE{chat_index}"
    code = f"CKSUM-{chat_index:02d}-{target_tokens:05d}"
    filler_chars = max(256, target_tokens * _CHARS_PER_TOKEN)
    sentence = _DOC_SENTENCE.format(keyword=keyword, code=code)
    body = []
    length = 0
    i = 0
    while length < filler_chars:
        para = " ".join(sentence for _ in range(8)) + "\n\n"
        body.append(para)
        length += len(para)
        i += 1
    return _DOC_INTRO + "".join(body), code


def _metrics_prefix_cache(root_url: str, api_key: str, timeout: float) -> tuple[int | None, int | None]:
    """Return (hits, queries) cumulative counters from vLLM /metrics, or (None, None)."""
    url = root_url.rstrip("/").removesuffix("/v1") + "/metrics"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return None, None
    hits = sum(float(m.group(1)) for m in _METRICS_HITS_RE.finditer(text))
    queries = sum(float(m.group(1)) for m in _METRICS_QUERIES_RE.finditer(text))
    return (int(hits), int(queries)) if queries else (None, None)


def _chat_once(base_url: str, model: str, api_key: str, messages: list, timeout: float) -> dict:
    """Non-streaming completion; returns ttft-ish latency + usage + text."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    code, body = _http_json(url, payload, api_key, timeout)
    wall = time.perf_counter() - t0
    if code != 200 or not isinstance(body, dict):
        return {"ok": False, "http": code, "error": str(body)[:300], "wall_s": round(wall, 3)}
    usage = body.get("usage") or {}
    choice = (body.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content")) or ""
    return {
        "ok": True,
        "http": code,
        "wall_s": round(wall, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "text_head": text[:160],
    }


def run_pair(base_url: str, model: str, api_key: str, chat_index: int,
             target_tokens: int, timeout: float, root_url: str) -> dict:
    doc, code = _build_document(chat_index, target_tokens)
    user1 = doc + f"\nQuestion: {_QUESTION}"
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user1},
    ]
    hits_before, queries_before = _metrics_prefix_cache(root_url, api_key, timeout)

    cold = _chat_once(base_url, model, api_key, messages, timeout)
    if not cold.get("ok"):
        return {"chat": chat_index, "cold": cold, "ok": False}

    assistant_reply = ((cold.get("text_head") or "").strip()) or f"The document requires {code}."
    messages = messages + [
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": _FOLLOWUP},
    ]
    warm = _chat_once(base_url, model, api_key, messages, timeout)

    hits_after, queries_after = _metrics_prefix_cache(root_url, api_key, timeout)
    hit_ratio = None
    if None not in (hits_before, hits_after, queries_before, queries_after):
        d_hits = hits_after - hits_before
        d_queries = queries_after - queries_before
        if d_queries > 0:
            hit_ratio = round(d_hits / d_queries, 4)

    warm_ttft = warm.get("wall_s")
    cold_ttft = cold.get("wall_s")
    speedup = round(cold_ttft / warm_ttft, 3) if (warm_ttft and cold_ttft) else None
    return {
        "chat": chat_index,
        "ok": bool(cold.get("ok") and warm.get("ok")),
        "cold": cold,
        "warm": warm,
        "warm_ttft_s": warm_ttft,
        "cold_ttft_s": cold_ttft,
        "speedup": speedup,
        "prefix_hit_ratio": hit_ratio,
        "needle_code": code,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--api-key-env", default=API_KEY_ENV)
    ap.add_argument("--runs", type=int, default=3, help="cold+warm pairs per chat")
    ap.add_argument("--concurrent", type=int, default=1, help="independent chats in flight")
    ap.add_argument("--prompt-tokens", type=int, default=7680, help="approx cold prompt size")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--min-hit-ratio", type=float, default=0.0,
                    help="fail (exit 1) when measured hit ratio is below this; 0 = report only")
    ap.add_argument("--output")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    root_url = args.base_url
    print(f"prefix-cache bench: {args.runs} pair(s) x {args.concurrent} chat(s), "
          f"~{args.prompt_tokens} tokens, model={args.model}")

    chats = list(range(args.concurrent))
    all_pairs: list[dict] = []
    ok = True
    for run in range(args.runs):
        if args.concurrent == 1:
            results = [run_pair(root_url, args.model, api_key, 0, args.prompt_tokens,
                                args.timeout, root_url)]
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as pool:
                futs = [pool.submit(run_pair, root_url, args.model, api_key, i,
                                    args.prompt_tokens, args.timeout, root_url)
                        for i in chats]
                results = [f.result() for f in futs]
        for r in results:
            all_pairs.append(r)
            if not r.get("ok"):
                ok = False
            ratio = r.get("prefix_hit_ratio")
            print(f"  run {run + 1} chat {r.get('chat')}: "
                  f"cold={r.get('cold_ttft_s')}s warm={r.get('warm_ttft_s')}s "
                  f"speedup={r.get('speedup')} hit={ratio} "
                  f"prompt_tokens={r.get('cold', {}).get('prompt_tokens')}"
                  + ("" if r.get("ok") else f"  FAILED http={r.get('cold', {}).get('http')}"))
        # small settle gap between runs
        time.sleep(0.5)

    colds = [r["cold_ttft_s"] for r in all_pairs if r.get("cold_ttft_s")]
    warms = [r["warm_ttft_s"] for r in all_pairs if r.get("warm_ttft_s")]
    ratios = [r["prefix_hit_ratio"] for r in all_pairs if r.get("prefix_hit_ratio") is not None]
    summary = {
        "pairs": len(all_pairs),
        "cold_ttft_median_s": round(statistics.median(colds), 3) if colds else None,
        "warm_ttft_median_s": round(statistics.median(warms), 3) if warms else None,
        "speedup_median": round(statistics.median(colds) / statistics.median(warms), 3)
        if colds and warms and statistics.median(warms) > 0 else None,
        "prefix_hit_ratio_median": round(statistics.median(ratios), 4) if ratios else None,
        "min_hit_ratio_required": args.min_hit_ratio or None,
    }
    print("summary:", json.dumps(summary))

    if args.min_hit_ratio > 0 and summary["prefix_hit_ratio_median"] is not None:
        if summary["prefix_hit_ratio_median"] < args.min_hit_ratio:
            print(f"FAIL: median hit ratio {summary['prefix_hit_ratio_median']} "
                  f"< required {args.min_hit_ratio}")
            ok = False

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else RESULTS_DIR / \
        f"glm53-exl3-prefix-cache-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "prompt_tokens_target": args.prompt_tokens,
        "runs": args.runs,
        "concurrent_chats": args.concurrent,
        "summary": summary,
        "pairs": all_pairs,
    }
    out.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
