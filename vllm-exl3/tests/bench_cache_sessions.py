#!/usr/bin/env python3
"""Multi-session prefix-cache benchmark for the GLM-5.3-Flash EXL3 lane.

New file in this lane.  The request shape (cold round, then warm repeats,
per-session co-batching) is inspired by the Apache-2.0 Reederey87 kit and
TensorFold's multi-session notes; this client is an independent stdlib-only
implementation.  It does not import or copy their code.

Protocol
--------
A run has ``--rounds`` rounds.  Round 1 is the cold round; rounds 2..N resend
the exact same messages, so the server can reuse cached KV pages.  Every
session is issued concurrently (one thread per session), with a barrier so the
requests really overlap.

Three prompt shapes are available:

* default / ``--sessions`` : each session owns a unique long document.
* ``--shared-prefix``      : all sessions share one long system prompt and
                             differ only in a short user tail.
* ``--sub-page``           : target context is kept below the 3584-token
                             hybrid-MLA page (3000 tokens by default).

``--salt`` (the default) adds a unique run salt at the very start of the
prompt so a new invocation cannot silently reuse pages left by a previous
one.  Use ``--no-salt`` only for deliberate cross-run reuse experiments.

Measurements come from the server, not from local estimates:

* ``prompt_tokens``  -- ``usage.prompt_tokens``
* ``cached_tokens``  -- ``usage.prompt_tokens_details.cached_tokens`` when the
  server exposes it; ``None`` otherwise (the client never invents a hit rate)
* ``ttft_s``         -- first streamed ``delta.content`` token
* ``decode_tok_s``   -- ``(completion_tokens - 1) / (end - first token)``,
  matching tests/bench_decode.py
* ``wall_s``         -- request start to end

Non-SSE servers are accepted: the client falls back to parsing the JSON body,
records ``ttft_s=None`` (there is no first-token timestamp in that shape) and
labels ``decode_tok_s`` as end-to-end.  The stream path is the default.

The verdict uses the median warm-round hit fraction: ``healthy`` at >= 0.90,
``collapsed`` at <= 0.05, otherwise ``partial``.  If the server does not expose
``cached_tokens`` the verdict is ``unknown`` -- it is never silently healthy.
Round 1 is reported separately as the cold round.

Exit status is 0 whenever every HTTP request completed successfully, even if
the verdict is ``collapsed`` (a low hit rate is a measurement, not a client
failure).  HTTP/parse failures exit 1.  ``--dry-run`` prints the planned
requests and exits 0 without any network I/O.

Usage
-----
    python3 tests/bench_cache_sessions.py --sessions 4 --ctx-tokens 60000 --rounds 3
    python3 tests/bench_cache_sessions.py --shared-prefix --sessions 4 --rounds 3
    python3 tests/bench_cache_sessions.py --sub-page --sessions 4 --rounds 3
    python3 tests/bench_cache_sessions.py --dry-run --sessions 2 --ctx-tokens 8000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

DEFAULT_BASE_URL = "http://127.0.0.1:8888"
DEFAULT_MODEL = "GLM-5.3-Flash-EXL3"
DEFAULT_PAGE_TOKENS = 3584
SUB_PAGE_TOKENS = 3000
HEALTHY_HIT_FRACTION = 0.90
COLLAPSED_HIT_FRACTION = 0.05
DEFAULT_CHARS_PER_TOKEN = 4.5
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

_QUESTION = "Reply with the single word OK."
_SYSTEM_HEAD = (
    "You are a careful maintenance analyst. Read the supplied log and answer "
    "only with the requested word."
)
_FILLER = (
    "The maintenance log for {marker} records checksum {checksum} before the "
    "deployment window; operators must verify the checksum and keep the entry "
    "for the audit trail. "
)


@dataclass
class ChatResult:
    """One measured completion request."""

    ok: bool
    http: int
    wall_s: float
    ttft_s: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    decode_tok_s: float | None = None
    decode_basis: str = "none"
    finish_reason: str | None = None
    text_head: str = ""
    usage: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "http": self.http,
            "wall_s": round(self.wall_s, 6),
            "ttft_s": None if self.ttft_s is None else round(self.ttft_s, 6),
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "decode_tok_s": None if self.decode_tok_s is None else round(self.decode_tok_s, 3),
            "decode_basis": self.decode_basis,
            "finish_reason": self.finish_reason,
            "text_head": self.text_head,
            "error": self.error,
        }


@dataclass(frozen=True)
class PlanConfig:
    """Prompt-plan knobs, kept separate from argparse for testability."""

    sessions: int
    ctx_tokens: int
    shared_prefix: bool = False
    sub_page: bool = False
    salt: str = ""
    page_tokens: int = DEFAULT_PAGE_TOKENS
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN


@dataclass(frozen=True)
class SessionPlan:
    """Messages and estimates for one session; reused unchanged every round."""

    session: int
    messages: list[dict[str, str]]
    target_tokens: int
    system_chars: int
    user_chars: int
    label: str
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN

    @property
    def approx_prompt_tokens(self) -> int:
        chars = self.system_chars + self.user_chars
        # Only used by --dry-run; real prompt_tokens come from the usage block.
        return int(round(chars / self.chars_per_token))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def api_root(base_url: str) -> str:
    """Return the server root, accepting either ``:8888`` or ``:8888/v1``."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def chat_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return root + "/chat/completions"
    return root + "/v1/chat/completions"


def validate_base_url(base_url: str) -> None:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"invalid --base-url {base_url!r}: expected http(s)://host[:port]")


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def parse_usage(usage: Any) -> tuple[int | None, int | None, int | None]:
    """Return ``(prompt_tokens, completion_tokens, cached_tokens)`` safely."""
    if not isinstance(usage, dict):
        return None, None, None
    prompt = _int_or_none(usage.get("prompt_tokens"))
    completion = _int_or_none(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details")
    cached = _int_or_none(details.get("cached_tokens")) if isinstance(details, dict) else None
    if cached is None:
        # A few OpenAI-compatible gateways expose the same counter directly.
        cached = _int_or_none(usage.get("cached_tokens"))
    return prompt, completion, cached


def hit_fraction(prompt_tokens: int | None, cached_tokens: int | None) -> float | None:
    """Cached share of the prompt; ``None`` when the server did not report it."""
    if prompt_tokens is None or cached_tokens is None or prompt_tokens <= 0:
        return None
    return cached_tokens / prompt_tokens


def classify_verdict(warm_hit_fractions: Sequence[float | None]) -> str:
    """Classify the warm-round hit fraction.

    ``healthy`` at >= 0.90, ``collapsed`` at <= 0.05, otherwise ``partial``.
    No usable warm samples means ``unknown`` -- never silently healthy.
    """
    values = [
        float(value)
        for value in warm_hit_fractions
        if value is not None and math.isfinite(float(value))
    ]
    if not values:
        return "unknown"
    median = statistics.median(values)
    if median >= HEALTHY_HIT_FRACTION:
        return "healthy"
    if median <= COLLAPSED_HIT_FRACTION:
        return "collapsed"
    return "partial"


def _repeat_filler(marker: str, checksum: str, chars: int) -> str:
    if chars <= 0:
        return ""
    sentence = _FILLER.format(marker=marker, checksum=checksum)
    repeats = max(1, chars // len(sentence) + 1)
    return (sentence * repeats)[:chars]


def _system_prefix(salt: str) -> str:
    return f"Run salt {salt}. " if salt else ""


def build_prompt_plan(config: PlanConfig) -> list[SessionPlan]:
    """Build one stable message list per session.

    The same objects are resent in every round, which is what makes rounds
    2..N warm.  A per-run salt (when enabled) sits at the beginning of the
    prompt so a fresh invocation starts cold.
    """
    if config.sessions < 1:
        raise ValueError("sessions must be >= 1")
    if config.ctx_tokens < 128:
        raise ValueError("ctx_tokens must be >= 128")

    target_tokens = config.ctx_tokens
    label = "sessions"
    if config.sub_page:
        safe = min(SUB_PAGE_TOKENS, config.page_tokens - 64)
        if safe < 128:
            raise ValueError("page_tokens is too small for a sub-page prompt")
        target_tokens = min(target_tokens, safe)
        label = "sub-page"
    if config.shared_prefix:
        label = "shared-prefix" if not config.sub_page else "shared-prefix+sub-page"

    target_chars = max(512, int(target_tokens * config.chars_per_token))
    salt = config.salt.strip()
    plans: list[SessionPlan] = []

    if config.shared_prefix:
        # Most of the context is the common system prompt; the tails differ per
        # session so a hit can only come from the shared prefix.
        system_chars = int(target_chars * 0.70)
        tail_chars = max(256, target_chars - system_chars)
        shared = _system_prefix(salt) + _SYSTEM_HEAD + "\n" + _repeat_filler(
            "shared-prefix", f"SHARED-{salt[-8:] or 'nosalt'}", max(0, system_chars - 160)
        )
        for index in range(config.sessions):
            tail = (
                f"Session {index}: "
                + _repeat_filler(
                    f"session-{index:02d}",
                    f"TAIL-{index:02d}-{salt[-8:] or 'nosalt'}",
                    max(0, tail_chars - 80),
                )
                + "\n"
                + _QUESTION
            )
            messages = [
                {"role": "system", "content": shared},
                {"role": "user", "content": tail},
            ]
            plans.append(
                SessionPlan(
                    session=index,
                    messages=messages,
                    target_tokens=target_tokens,
                    system_chars=len(shared),
                    user_chars=len(tail),
                    label=label,
                    chars_per_token=config.chars_per_token,
                )
            )
        return plans

    for index in range(config.sessions):
        # Each session has its own document, so concurrent sessions cannot
        # share pages in the default mode.
        body = _system_prefix(salt) + _SYSTEM_HEAD + (
            f"\nSession {index:02d}. "
        ) + _repeat_filler(
            f"session-{index:02d}",
            f"DOC-{index:02d}-{salt[-8:] or 'nosalt'}",
            max(0, target_chars - 160),
        )
        messages = [
            {"role": "system", "content": body},
            {"role": "user", "content": _QUESTION},
        ]
        plans.append(
            SessionPlan(
                session=index,
                messages=messages,
                target_tokens=target_tokens,
                system_chars=len(body),
                user_chars=len(_QUESTION),
                label=label,
                chars_per_token=config.chars_per_token,
            )
        )
    return plans


def _parse_json_body(raw: bytes, http: int, wall_s: float) -> ChatResult:
    try:
        body = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        return ChatResult(
            ok=False,
            http=http,
            wall_s=wall_s,
            error=f"malformed JSON response: {exc}",
        )
    if not isinstance(body, dict):
        return ChatResult(ok=False, http=http, wall_s=wall_s, error="response is not a JSON object")
    error = body.get("error")
    if error:
        return ChatResult(ok=False, http=http, wall_s=wall_s, error=f"server error: {error!r}"[:500])
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ChatResult(ok=False, http=http, wall_s=wall_s, error="response has no choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    text = message.get("content") or ""
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
    prompt_tokens, completion_tokens, cached_tokens = parse_usage(usage)
    return ChatResult(
        ok=True,
        http=http,
        wall_s=wall_s,
        ttft_s=None,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        decode_tok_s=(completion_tokens / wall_s) if completion_tokens and wall_s > 0 else None,
        decode_basis="e2e_non_stream",
        finish_reason=choice.get("finish_reason"),
        text_head=str(text)[:160],
        usage=usage,
    )


def _parse_sse(resp: Any, start: float) -> ChatResult:
    """Parse an SSE response; fall back to a plain JSON body if needed."""
    http = int(getattr(resp, "status", 200) or 200)
    chunks: list[str] = []
    raw_parts: list[bytes] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    first_content: float | None = None
    malformed = 0
    server_error: str | None = None

    for raw_line in resp:
        if isinstance(raw_line, str):
            raw_line = raw_line.encode("utf-8", "replace")
        if not raw_line.startswith(b"data:"):
            raw_parts.append(raw_line)
            continue
        payload = raw_line[5:].strip()
        if payload == b"[DONE]":
            continue
        try:
            event = json.loads(payload.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(event, dict):
            malformed += 1
            continue
        if event.get("error") and server_error is None:
            server_error = str(event.get("error"))[:500]
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            if first_content is None:
                first_content = time.perf_counter()
            chunks.append(content)
        if choice.get("finish_reason"):
            finish_reason = str(choice["finish_reason"])

    end = time.perf_counter()
    wall = end - start
    text = "".join(chunks)
    prompt_tokens, completion_tokens, cached_tokens = parse_usage(usage)

    if not chunks and not usage and not server_error:
        # Some gateways return application/json even when stream=true.  Re-parse
        # the buffered body; this path has no first-token timestamp.
        raw = b"".join(raw_parts)
        if raw.strip():
            fallback = _parse_json_body(raw, http, wall)
            fallback.decode_basis = "e2e_non_stream"
            return fallback

    if server_error:
        return ChatResult(ok=False, http=http, wall_s=wall, error=f"server error: {server_error}")
    if malformed and not chunks and not usage:
        return ChatResult(
            ok=False,
            http=http,
            wall_s=wall,
            error=f"malformed SSE stream ({malformed} bad event(s), no content)",
        )
    if not chunks and completion_tokens is None:
        return ChatResult(
            ok=False,
            http=http,
            wall_s=wall,
            error="stream produced no content and no usage",
        )

    ttft = None if first_content is None else (first_content - start)
    decode_tok_s = None
    if ttft is not None and completion_tokens is not None and completion_tokens > 1:
        span = end - first_content  # type: ignore[operator]
        if span > 0:
            decode_tok_s = (completion_tokens - 1) / span
    return ChatResult(
        ok=True,
        http=http,
        wall_s=wall,
        ttft_s=ttft,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        decode_tok_s=decode_tok_s,
        decode_basis="stream" if ttft is not None else "none",
        finish_reason=finish_reason,
        text_head=text[:160],
        usage=usage,
    )


def chat_request(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str = "",
    timeout: float = 900.0,
    max_tokens: int = 16,
    stream: bool = True,
) -> ChatResult:
    """Issue one chat completion and measure it.  Never raises on HTTP errors."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        chat_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    start = time.perf_counter()
    try:
        resp = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        return ChatResult(
            ok=False,
            http=exc.code,
            wall_s=time.perf_counter() - start,
            error=f"HTTP {exc.code}: {body}",
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return ChatResult(
            ok=False,
            http=0,
            wall_s=time.perf_counter() - start,
            error=f"connection error: {exc}",
        )

    with resp:
        if not stream:
            wall_start = start
            try:
                raw = resp.read()
            except (TimeoutError, OSError) as exc:
                return ChatResult(
                    ok=False,
                    http=int(getattr(resp, "status", 0) or 0),
                    wall_s=time.perf_counter() - start,
                    error=f"read error: {exc}",
                )
            return _parse_json_body(raw, int(getattr(resp, "status", 200) or 200),
                                   time.perf_counter() - wall_start)
        return _parse_sse(resp, start)


def _sum_complete(values: Sequence[int | None]) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values if value is not None)


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate_round(round_number: int, results: Sequence[ChatResult], wall_s: float) -> dict[str, Any]:
    """Turn one round's per-session results into comparable round metrics."""
    prompt_tokens = _sum_complete([result.prompt_tokens for result in results])
    cached_tokens = _sum_complete([result.cached_tokens for result in results])
    ttfts = [result.ttft_s for result in results if result.ok and result.ttft_s is not None]
    decodes = [result.decode_tok_s for result in results if result.ok and result.decode_tok_s is not None]
    errors = [result.error for result in results if result.error]
    all_ok = all(result.ok for result in results)
    return {
        "round": round_number,
        "phase": "cold" if round_number == 1 else "warm",
        "ok": all_ok,
        "wall_s": round(wall_s, 6),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "hit_fraction": hit_fraction(prompt_tokens, cached_tokens),
        "ttft_s": round(max(ttfts), 6) if ttfts else None,
        "ttft_median_s": round(_median(ttfts), 6) if ttfts else None,
        "ttft_sessions_measured": len(ttfts),
        "decode_tok_s": round(_median(decodes), 3) if decodes else None,
        "completion_tokens": _sum_complete([result.completion_tokens for result in results]),
        "errors": errors,
        "sessions": [result.to_dict() for result in results],
    }


def run_benchmark(
    args: argparse.Namespace,
    *,
    plan: Sequence[SessionPlan],
    salt: str,
    api_key: str = "",
) -> dict[str, Any]:
    """Run all rounds sequentially; each round co-batches every session."""
    started_at = utc_now()
    mode = _mode(args)
    rounds: list[dict[str, Any]] = []
    warnings: list[str] = []
    if args.rounds < 2:
        warnings.append("rounds < 2: no warm round, verdict will be unknown")
    if args.shared_prefix and args.sessions < 2:
        warnings.append("shared-prefix with one session cannot show cross-session reuse")
    if args.shared_prefix:
        warnings.append(
            "shared-prefix: round 1 is the first exposure of a common prefix; "
            "concurrent sessions may share it by design"
        )
    if args.sub_page:
        warnings.append(
            "sub-page: prompts stay below one 3584-token hybrid-MLA page; a low "
            "hit fraction can reflect page granularity rather than cache failure"
        )

    for round_number in range(1, args.rounds + 1):
        print(
            f"[cache-sessions] round {round_number}/{args.rounds} "
            f"({'cold' if round_number == 1 else 'warm'}) sessions={args.sessions}",
            flush=True,
        )
        barrier = threading.Barrier(len(plan))

        def one(session_plan: SessionPlan, barrier: threading.Barrier = barrier) -> ChatResult:
            try:
                barrier.wait(timeout=min(max(args.timeout, 1.0), 120.0))
            except threading.BrokenBarrierError as exc:
                return ChatResult(ok=False, http=0, wall_s=0.0, error=f"barrier error: {exc}")
            return chat_request(
                args.base_url,
                args.model,
                session_plan.messages,
                api_key=api_key,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                stream=args.stream,
            )

        round_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(plan)) as pool:
            futures = [pool.submit(one, session_plan) for session_plan in plan]
            results: list[ChatResult] = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        ChatResult(
                            ok=False,
                            http=0,
                            wall_s=0.0,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
        record = aggregate_round(round_number, results, time.perf_counter() - round_start)
        rounds.append(record)
        print(
            f"[cache-sessions]   ok={record['ok']} prompt_tokens={record['prompt_tokens']} "
            f"cached={record['cached_tokens']} hit={record['hit_fraction']} "
            f"ttft={record['ttft_s']} decode={record['decode_tok_s']} wall={record['wall_s']}",
            flush=True,
        )

    warm_hits = [
        round_record["hit_fraction"]
        for round_record in rounds
        if round_record["round"] >= 2 and round_record["hit_fraction"] is not None
    ]
    warm_rounds = [round_record for round_record in rounds if round_record["round"] >= 2]
    if warm_rounds and not warm_hits:
        warnings.append(
            "warm rounds exposed no cached_tokens; the server may need "
            "--enable-prompt-tokens-details"
        )
    for round_record in rounds:
        fraction = round_record["hit_fraction"]
        if fraction is not None and fraction > 1.0:
            warnings.append(f"round {round_record['round']} hit fraction > 1.0: {fraction:.3f}")

    verdict = classify_verdict(warm_hits) if warm_rounds else "unknown"
    warm_ttfts = [
        session["ttft_s"]
        for round_record in warm_rounds
        for session in round_record["sessions"]
        if session.get("ttft_s") is not None
    ]
    warm_decodes = [
        session["decode_tok_s"]
        for round_record in warm_rounds
        for session in round_record["sessions"]
        if session.get("decode_tok_s") is not None
    ]
    http_ok = all(round_record["ok"] for round_record in rounds)
    summary = {
        "rounds": len(rounds),
        "sessions": args.sessions,
        "mode": mode,
        "http_ok": http_ok,
        "cold_hit_fraction": rounds[0]["hit_fraction"] if rounds else None,
        "warm_hit_fraction_median": _median(warm_hits) if warm_hits else None,
        "warm_ttft_median_s": _median(warm_ttfts) if warm_ttfts else None,
        "warm_decode_tok_s_median": _median(warm_decodes) if warm_decodes else None,
        "verdict": verdict,
    }
    return {
        "schema": "glm53-exl3-cache-sessions/1",
        "created_at": started_at,
        "finished_at": utc_now(),
        "base_url": args.base_url,
        "endpoint": chat_url(args.base_url),
        "model": args.model,
        "mode": mode,
        "verdict": verdict,
        "http_ok": http_ok,
        "warnings": warnings,
        "config": {
            "sessions": args.sessions,
            "ctx_tokens": args.ctx_tokens,
            "effective_ctx_tokens": plan[0].target_tokens if plan else None,
            "page_tokens": args.page_tokens,
            "rounds": args.rounds,
            "max_tokens": args.max_tokens,
            "stream": args.stream,
            "shared_prefix": args.shared_prefix,
            "sub_page": args.sub_page,
            "salt_enabled": bool(salt),
            "salt": salt or None,
            "chars_per_token": args.chars_per_token,
        },
        "summary": summary,
        "rounds": rounds,
    }


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-session prefix-cache benchmark (stdlib only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Round 1 is cold; rounds 2..N resend identical messages. "
            "Exit 0 on successful HTTP even when the verdict is collapsed."
        ),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="vLLM root URL (with or without /v1)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sessions", type=_positive_int, default=4,
                        help="concurrent sessions (co-batch width)")
    parser.add_argument("--ctx-tokens", type=_positive_int, default=60000,
                        help="approximate prompt tokens per session")
    parser.add_argument("--rounds", type=_positive_int, default=3,
                        help="round 1 cold, rounds 2+ warm")
    parser.add_argument("--max-tokens", type=_positive_int, default=16,
                        help="completion tokens per request")
    parser.add_argument("--shared-prefix", action="store_true",
                        help="same long system prompt, different user tails")
    parser.add_argument("--sub-page", action="store_true",
                        help="keep prompts below the 3584-token hybrid-MLA page")
    parser.add_argument("--page-tokens", type=_positive_int, default=DEFAULT_PAGE_TOKENS,
                        help="hybrid-MLA page size used by --sub-page")
    parser.add_argument("--salt", action=argparse.BooleanOptionalAction, default=True,
                        help="unique run salt in the prompt (use --no-salt to disable)")
    parser.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN,
                        help="filler sizing heuristic; measured tokens come from usage")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                        help="stream to measure TTFT from the first content token")
    parser.add_argument("--api-key", default=None,
                        help="bearer token; defaults to $VLLM_API_KEY or $API_KEY")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="per-request HTTP timeout (seconds)")
    parser.add_argument("--out", default=None,
                        help="JSON artifact path (default: results/glm53-cache-sessions-<stamp>.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned requests and exit without network I/O")
    return parser


def _mode(args: argparse.Namespace) -> str:
    labels = []
    if args.shared_prefix:
        labels.append("shared-prefix")
    if args.sub_page:
        labels.append("sub-page")
    return "+".join(labels) if labels else "sessions"


def _print_dry_run(args: argparse.Namespace, plan: Sequence[SessionPlan], salt: str) -> None:
    print("[dry-run] multi-session prefix-cache benchmark", flush=True)
    print(
        f"[dry-run] endpoint={chat_url(args.base_url)} model={args.model} "
        f"mode={_mode(args)} sessions={args.sessions} rounds={args.rounds} "
        f"ctx_tokens={args.ctx_tokens} max_tokens={args.max_tokens} "
        f"stream={args.stream} salt={'yes' if salt else 'no'}",
        flush=True,
    )
    for round_number in range(1, args.rounds + 1):
        phase = "cold" if round_number == 1 else "warm"
        for session_plan in plan:
            print(
                f"[dry-run] round={round_number} phase={phase} "
                f"session={session_plan.session} "
                f"approx_prompt_tokens={session_plan.approx_prompt_tokens} "
                f"system_chars={session_plan.system_chars} "
                f"user_chars={session_plan.user_chars} "
                f"label={session_plan.label}",
                flush=True,
            )
    print("[dry-run] no requests sent", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_base_url(args.base_url)
        if args.timeout <= 0:
            raise ValueError("--timeout must be > 0")
        if args.chars_per_token <= 0:
            raise ValueError("--chars-per-token must be > 0")
        salt = secrets.token_hex(8) if args.salt else ""
        plan = build_prompt_plan(
            PlanConfig(
                sessions=args.sessions,
                ctx_tokens=args.ctx_tokens,
                shared_prefix=args.shared_prefix,
                sub_page=args.sub_page,
                salt=salt,
                page_tokens=args.page_tokens,
                chars_per_token=args.chars_per_token,
            )
        )
        if args.api_key is None:
            args.api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("API_KEY") or ""
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.sub_page and args.ctx_tokens > min(SUB_PAGE_TOKENS, args.page_tokens - 64):
        print(
            f"note: --sub-page clamps ctx_tokens {args.ctx_tokens} to "
            f"{min(SUB_PAGE_TOKENS, args.page_tokens - 64)}",
            file=sys.stderr,
        )

    if args.dry_run:
        _print_dry_run(args, plan, salt)
        return 0

    artifact = run_benchmark(args, plan=plan, salt=salt, api_key=args.api_key or "")
    if artifact["summary"]["http_ok"]:
        print(
            f"[cache-sessions] verdict={artifact['summary']['verdict']} "
            f"warm_hit={artifact['summary']['warm_hit_fraction_median']} "
            f"warm_ttft={artifact['summary']['warm_ttft_median_s']}s "
            f"warm_decode={artifact['summary']['warm_decode_tok_s_median']} tok/s",
            flush=True,
        )
    else:
        print("[cache-sessions] HTTP failure; artifact records per-session errors", flush=True)

    if args.out:
        out = Path(args.out).expanduser()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = RESULTS_DIR / f"glm53-cache-sessions-{stamp}.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write {out}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0 if artifact["summary"]["http_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
