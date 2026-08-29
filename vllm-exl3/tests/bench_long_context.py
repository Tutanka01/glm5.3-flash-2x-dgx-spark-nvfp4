#!/usr/bin/env python3
"""Cold long-context capacity, retrieval, TTFT and decode benchmark for the
vLLM EXL3 lane (OpenAI-compatible server on :8888, model id GLM-5.3-Flash-EXL3).

The calibrated prompt embeds three retrieval needles at ~5% / ~50% / ~95% of
the body (BEGIN_NEEDLE=ALPHA_7319, MIDDLE_NEEDLE=MIDDLE_2846,
END_NEEDLE=OMEGA_9052); the model must answer with the three codes joined by
"|". Token counts are calibrated through the server's /tokenize endpoint and
the timed request is a single streaming chat completion.

--cold resets the server prefix cache before the timed request (vLLM
/reset_prefix_cache, with /flush_cache endpoints as fallbacks).

Usage:
  # Short sanity run against the local vLLM server:
  python3 vllm-exl3/tests/bench_long_context.py --target-tokens 8192 --label sanity

  # Cold 200k capacity run (--cold resets the prefix cache first):
  python3 vllm-exl3/tests/bench_long_context.py --target-tokens 200000 --cold \
      --label cold-200k --timeout 3600

  # Deliberately unsafe run that skips every context-window gate:
  python3 vllm-exl3/tests/bench_long_context.py --target-tokens 262144 \
      --allow-unsafe-profile --label unsafe-256k
"""

# Adapted from ../bench-long-context.py (SGLang lane) for the vLLM EXL3 lane.

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


NEEDLES = (
    ("BEGIN_NEEDLE", "ALPHA_7319"),
    ("MIDDLE_NEEDLE", "MIDDLE_2846"),
    ("END_NEEDLE", "OMEGA_9052"),
)
EXPECTED = "|".join(value for _, value in NEEDLES)
FILLER = "Neutral archival padding sentence for context capacity verification.\n"
# Unique per invocation so repeated runs never hit the prefix cache left by
# a previous session (this vLLM build exposes no cache-reset endpoint).
_SESSION_ID = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class BenchmarkError(RuntimeError):
    pass


class TokenizerProbeError(BenchmarkError):
    """Tokenizer discovery failed, optionally for a transient server reason."""

    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


RETRYABLE_HTTP_CODES = frozenset((408, 425, 429, 500, 502, 503, 504))
LONG_CONTEXT_THRESHOLD = 131072
# Leave room for the chat template and the requested completion.
REQUEST_HEADROOM_TOKENS = 512


def is_loopback_url(url: str) -> bool:
    hostname = urllib.parse.urlparse(url).hostname
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def urlopen(request: urllib.request.Request, timeout: int):  # type: ignore[no-untyped-def]
    """Open local APIs directly even when the shell exports an HTTP proxy."""

    if is_loopback_url(request.full_url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=timeout
        )
    return urllib.request.urlopen(request, timeout=timeout)


def fetch_max_model_len(
    base_url: str,
    model: str,
    api_key: str | None,
    timeout: int,
) -> tuple[int | None, str]:
    """Read max_model_len for `model` from the OpenAI-compatible /models list."""

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        return None, describe_http_error(exc)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return None, "HTTP 200 with an invalid /models response"
    metadata = next(
        (
            item
            for item in payload["data"]
            if isinstance(item, dict) and item.get("id") == model
        ),
        None,
    )
    if metadata is None:
        ids = [item.get("id") for item in payload["data"] if isinstance(item, dict)]
        return None, f"model {model!r} is not served (served ids: {ids})"
    value = metadata.get("max_model_len")
    if not isinstance(value, int) or isinstance(value, bool):
        return None, f"max_model_len is not reported for {model!r}"
    return value, f"max_model_len={value}"


def enforce_context_capacity(
    base_url: str,
    model: str,
    api_key: str | None,
    target_tokens: int,
    allow_unsafe_profile: bool,
    timeout: int,
) -> None:
    """Fail closed when the target prompt cannot fit the served context window."""

    max_model_len, detail = fetch_max_model_len(base_url, model, api_key, timeout)
    if max_model_len is not None:
        if target_tokens + REQUEST_HEADROOM_TOKENS <= max_model_len:
            print(
                "Context preflight passed: "
                f"{detail}, target={target_tokens:,} tokens"
            )
            return
        if allow_unsafe_profile:
            print(
                f"WARNING: bypassing the context check ({detail}); the "
                f"{target_tokens:,}-token target exceeds the served context window",
                file=sys.stderr,
            )
            return
        raise BenchmarkError(
            f"refusing a {target_tokens:,}-token prompt: the server reports "
            f"{detail} for model {model!r}. Restart the server with a larger "
            "MAX_MODEL_LEN in vllm-exl3/.env (`vllm-exl3/start.sh stop`, then "
            "`vllm-exl3/start.sh`), or lower --target-tokens."
        )
    if allow_unsafe_profile:
        print(
            f"WARNING: bypassing the context check ({detail}); the served "
            "context window could not be verified",
            file=sys.stderr,
        )
        return
    if target_tokens > LONG_CONTEXT_THRESHOLD:
        raise BenchmarkError(
            f"refusing a {target_tokens:,}-token prompt because the served "
            f"context window could not be verified ({detail}). Restart the "
            "server with an explicit MAX_MODEL_LEN in vllm-exl3/.env, or pass "
            "--allow-unsafe-profile only for an intentionally unprotected "
            "experiment."
        )
    print(
        "WARNING: could not verify the served context window "
        f"({detail}); continuing with the {target_tokens:,}-token target at "
        f"or below the {LONG_CONTEXT_THRESHOLD}-token tier",
        file=sys.stderr,
    )


# NOTE: the SGLang lane refuses to run while its startup memory guard
# (.glm53-guard-head.pid) is alive. That guard belongs to the sibling SGLang
# lane and is intentionally not consulted here.


def describe_http_error(exc: urllib.error.HTTPError) -> str:
    detail = f"HTTP {exc.code} {exc.reason}"
    try:
        body = exc.read(512).decode("utf-8", errors="replace").strip()
    except OSError:
        body = ""
    if body:
        compact = " ".join(body.split())
        detail += f": {compact}"
    return detail


def request_json(
    url: str,
    payload: dict[str, Any] | None,
    api_key: str | None,
    timeout: int,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise BenchmarkError(f"{url} returned a non-object JSON response")
    return result


def count_token_payload(payload: dict[str, Any]) -> int:
    for key in ("count", "num_tokens"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    for key in ("tokens", "input_ids"):
        value = payload.get(key)
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                if len(value) != 1:
                    raise BenchmarkError("tokenizer unexpectedly returned a batch")
                return len(value[0])
            return len(value)
    raise BenchmarkError(f"tokenizer response has no token count: keys={sorted(payload)}")


def discover_tokenizer(
    base_url: str,
    model: str,
    api_key: str | None,
    timeout: int,
) -> tuple[str, Callable[[str], dict[str, Any]]]:
    base = base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    candidates: list[tuple[str, Callable[[str], dict[str, Any]]]] = [
        (f"{base}/tokenize", lambda text: {"model": model, "prompt": text}),
        (f"{base}/tokenize", lambda text: {"model": model, "input": text}),
        # vLLM mounts /tokenize at the root and validates `prompt` (or
        # `messages`), not `text` — observed 400 on the glm53-flash image.
        (f"{root}/tokenize", lambda text: {"model": model, "prompt": text}),
        (f"{root}/tokenize", lambda text: {"text": text}),
        (f"{root}/tokenize", lambda text: {"model": model, "text": text}),
    ]
    failures: dict[str, list[str]] = {}
    retryable = False
    for url, make_payload in candidates:
        try:
            response = request_json(url, make_payload("tokenizer probe"), api_key, timeout)
            count_token_payload(response)
            return url, make_payload
        except urllib.error.HTTPError as exc:
            retryable = retryable or exc.code in RETRYABLE_HTTP_CODES
            failures.setdefault(url, []).append(describe_http_error(exc))
        except (OSError, urllib.error.URLError) as exc:
            retryable = True
            failures.setdefault(url, []).append(str(exc))
        except (BenchmarkError, ValueError) as exc:
            failures.setdefault(url, []).append(str(exc))

    details = []
    for url, reasons in failures.items():
        unique_reasons = list(dict.fromkeys(reasons))
        details.append(f"{url}: {' | '.join(unique_reasons)}")
    if retryable:
        message = "the server tokenizer/engine is temporarily unavailable; tried: "
    else:
        message = "no compatible tokenizer endpoint found on the server; tried: "
    raise TokenizerProbeError(
        message + "; ".join(details),
        retryable=retryable,
    )


def model_api_status(
    base_url: str,
    model: str | None,
    api_key: str | None,
    timeout: int,
) -> tuple[bool, str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        return False, describe_http_error(exc)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return False, str(exc)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return False, "HTTP 200 with an invalid /models response"
    ids = [item.get("id") for item in payload["data"] if isinstance(item, dict)]
    if model is not None and model not in ids:
        return False, f"HTTP 200, but served model ids are {ids}"
    return True, f"HTTP 200, served model ids are {ids}"


def wait_for_tokenizer(
    base_url: str,
    model: str,
    api_key: str | None,
    request_timeout: int,
    readiness_timeout: int,
) -> tuple[str, Callable[[str], dict[str, Any]]]:
    started = time.monotonic()
    deadline = started + readiness_timeout
    last_error: TokenizerProbeError | None = None
    last_model_status = "not checked"
    announced = False

    while True:
        try:
            return discover_tokenizer(base_url, model, api_key, request_timeout)
        except TokenizerProbeError as exc:
            last_error = exc
            _, last_model_status = model_api_status(
                base_url, model, api_key, min(request_timeout, 10)
            )
            remaining = deadline - time.monotonic()
            if not exc.retryable or remaining <= 0:
                break
            if not announced:
                print(
                    "the server answered but its tokenizer/engine is not ready; "
                    f"waiting up to {readiness_timeout}s (/models: {last_model_status})",
                    file=sys.stderr,
                )
                announced = True
            time.sleep(min(2.0, remaining))

    assert last_error is not None
    elapsed = time.monotonic() - started
    if last_error.retryable:
        raise BenchmarkError(
            f"the server did not become usable after {elapsed:.1f}s. "
            f"/models: {last_model_status}. Tokenizer: {last_error}. "
            "HTTP 503 means the HTTP front end is reachable but the inference "
            "engine is loading, stopping, or no longer alive; inspect "
            "`vllm-exl3/start.sh logs` before retrying."
        )
    raise BenchmarkError(f"{last_error}. /models: {last_model_status}")


def make_context(repeats: int) -> str:
    # Approximate needle positions: 5%, 50%, and 95% of the message body.
    segments = (
        repeats * 5 // 100,
        repeats * 45 // 100,
        repeats * 45 // 100,
    )
    used = sum(segments)
    tail = repeats - used
    return "".join(
        (
            "GLM53_LONG_CONTEXT_CAPABILITY_TEST\n"
            f"SESSION {_SESSION_ID}. "  # unique per invocation: true colds
            "Read the full archival text. Three labelled retrieval codes occur once each.\n",
            FILLER * segments[0],
            f"\n{NEEDLES[0][0]} stores {NEEDLES[0][1]}.\n",
            FILLER * segments[1],
            f"\n{NEEDLES[1][0]} stores {NEEDLES[1][1]}.\n",
            FILLER * segments[2],
            f"\n{NEEDLES[2][0]} stores {NEEDLES[2][1]}.\n",
            FILLER * tail,
            "\nReturn the values stored by BEGIN_NEEDLE, MIDDLE_NEEDLE, and "
            "END_NEEDLE, in that order, separated only by |. Do not add any other text.\n",
        )
    )


def calibrate_context(
    target_tokens: int,
    tokenize: Callable[[str], int],
) -> tuple[str, int, int]:
    unit_tokens = tokenize(FILLER)
    if unit_tokens < 1:
        raise BenchmarkError("filler tokenized to zero tokens")
    overhead_tokens = tokenize(make_context(0))
    repeats = max(1, (target_tokens - overhead_tokens) // unit_tokens)
    actual = 0
    context = ""
    for _ in range(4):
        context = make_context(repeats)
        actual = tokenize(context)
        delta = target_tokens - actual
        if abs(delta) <= max(64, target_tokens // 2000):
            break
        repeats = max(1, repeats + round(delta / unit_tokens))
    return context, actual, repeats


def reset_backend_cache(base_url: str, api_key: str | None, timeout: int) -> str | None:
    """Reset the server prefix cache, trying every known reset endpoint.

    Returns the endpoint that worked, or None when the server offers none.
    On this glm53-flash vLLM build no reset endpoint exists (404 on every
    candidate, observed 2026-08-29): --cold then relies on the unique
    SESSION filler so repeated invocations still measure true cold
    prefills. A lane restart is the stricter guarantee (fresh boot = empty
    cache).
    """

    base = base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    candidates = (
        f"{root}/reset_prefix_cache",
        f"{base}/reset_prefix_cache",
        f"{base}/flush_cache",
        f"{root}/flush_cache",
    )
    failures: list[str] = []
    for url in candidates:
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            request = urllib.request.Request(
                url, data=b"{}", headers=headers, method="POST"
            )
            with urlopen(request, timeout=timeout) as response:
                response.read()
            return url
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 501):
                failures.append(f"{url}: endpoint not offered ({describe_http_error(exc)})")
                continue
            failures.append(f"{url}: {describe_http_error(exc)}")
        except (OSError, urllib.error.URLError) as exc:
            failures.append(f"{url}: {exc}")
    print(
        "WARNING: no cache-reset endpoint responded; proceeding without a "
        "reset. Cold-ness is preserved by the unique SESSION filler."
    )
    return None


def stream_chat(
    base_url: str,
    model: str,
    api_key: str | None,
    context: str,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": context}],
        "temperature": 0,
        "max_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    with urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                content_piece = delta.get("content") or ""
                reasoning_piece = delta.get("reasoning_content") or ""
                tool_piece = delta.get("tool_calls") or []
                if first_token_at is None and (content_piece or reasoning_piece or tool_piece):
                    first_token_at = time.perf_counter()
                if content_piece:
                    content_parts.append(content_piece)
                if reasoning_piece:
                    reasoning_parts.append(reasoning_piece)
    finished = time.perf_counter()
    if first_token_at is None:
        raise BenchmarkError("stream contained no content, reasoning, or tool delta")
    completion_tokens = usage.get("completion_tokens")
    decode_rate = None
    decode_window = finished - first_token_at if finished > first_token_at else 0.0
    # A sub-100 ms window on a short completion makes (n-1)/window explode
    # into absurd tok/s (observed 516638 on a 40-token answer) — the SSE tail
    # can flush in one read. Report None below a minimum measurement window.
    if (
        isinstance(completion_tokens, int)
        and completion_tokens > 1
        and decode_window >= 0.25
    ):
        decode_rate = (completion_tokens - 1) / decode_window
    prompt_tokens = usage.get("prompt_tokens")
    return {
        "content": "".join(content_parts),
        "reasoning_chars": len("".join(reasoning_parts)),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": first_token_at - started,
        "total_seconds": finished - started,
        "decode_tokens_per_second": decode_rate,
        "prefill_tokens_per_second_e2e": (
            prompt_tokens / (first_token_at - started)
            if isinstance(prompt_tokens, int) and first_token_at > started
            else None
        ),
    }


def api_is_healthy(base_url: str, api_key: str | None, timeout: int) -> bool:
    healthy, _ = model_api_status(base_url, None, api_key, timeout)
    return healthy


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a real cold long prompt with three retrieval needles."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--readiness-timeout",
        type=int,
        default=60,
        help="seconds to retry transient tokenizer/API failures before aborting",
    )
    parser.add_argument(
        "--cold", action="store_true", help="reset the server prefix cache first"
    )
    parser.add_argument(
        "--allow-unsafe-profile",
        action="store_true",
        help="bypass every context-window gate (may crash the engine or fail the request)",
    )
    parser.add_argument("--label", default="long-context")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1024 <= args.target_tokens <= 1048000:
        parser.error("--target-tokens must be between 1024 and 1048000")
    if args.timeout < 30:
        parser.error("--timeout must be at least 30 seconds")
    if args.readiness_timeout < 0:
        parser.error("--readiness-timeout must be non-negative")

    api_key = os.environ.get(args.api_key_env)
    enforce_context_capacity(
        args.base_url,
        args.model,
        api_key,
        args.target_tokens,
        args.allow_unsafe_profile,
        min(args.timeout, 30),
    )
    tokenizer_url, make_tokenizer_payload = wait_for_tokenizer(
        args.base_url,
        args.model,
        api_key,
        min(args.timeout, 30),
        args.readiness_timeout,
    )
    model_ready, model_status = model_api_status(
        args.base_url,
        args.model,
        api_key,
        min(args.timeout, 30),
    )
    if not model_ready:
        raise BenchmarkError(f"server preflight failed: {model_status}")

    def tokenize(text: str) -> int:
        response = request_json(
            tokenizer_url,
            make_tokenizer_payload(text),
            api_key,
            min(args.timeout, 600),
        )
        return count_token_payload(response)

    print(f"Calibrating a {args.target_tokens:,}-token prompt through {tokenizer_url}")
    context, raw_tokens, repeats = calibrate_context(args.target_tokens, tokenize)
    print(
        f"Built {raw_tokens:,} raw tokens ({len(context):,} chars, "
        f"{repeats:,} padding records)"
    )
    flushed_at = reset_backend_cache(args.base_url, api_key, 60) if args.cold else None
    if flushed_at:
        print(f"Reset prefix cache through {flushed_at}")

    failure: str | None = None
    response: dict[str, Any] = {}
    try:
        response = stream_chat(args.base_url, args.model, api_key, context, args.timeout)
    except (BenchmarkError, OSError, ValueError, urllib.error.URLError) as exc:
        failure = str(exc)
    healthy_after = api_is_healthy(args.base_url, api_key, 30)
    content = str(response.get("content", ""))
    normalized = re.sub(r"[`\s]", "", content)
    retrieval_ok = EXPECTED in normalized
    ok = failure is None and retrieval_ok and healthy_after

    artifact = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "target_tokens": args.target_tokens,
        "raw_message_tokens": raw_tokens,
        "actual_prompt_tokens": response.get("prompt_tokens"),
        "cold_cache": args.cold,
        "tokenizer_endpoint": tokenizer_url,
        "needle_positions": [0.05, 0.50, 0.95],
        "expected_sha256": hashlib.sha256(EXPECTED.encode("utf-8")).hexdigest(),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_chars": len(content),
        "retrieval_ok": retrieval_ok,
        "api_healthy_after": healthy_after,
        "ok": ok,
        "error": failure,
        "ttft_seconds": response.get("ttft_seconds"),
        "total_seconds": response.get("total_seconds"),
        "prefill_tokens_per_second_e2e": response.get("prefill_tokens_per_second_e2e"),
        "decode_tokens_per_second": response.get("decode_tokens_per_second"),
        "completion_tokens": response.get("completion_tokens"),
        "reasoning_chars": response.get("reasoning_chars"),
    }
    output = args.output
    if output is None:
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", args.label).strip("-") or "long"
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        # Lane-local vllm-exl3/results/, like the other EXL3 benches.
        output = (
            Path(__file__).resolve().parents[1]
            / "results"
            / f"glm53-long-context-{safe_label}-{stamp}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    print(
        "Long-context result: "
        f"ok={ok} retrieval={retrieval_ok} healthy_after={healthy_after} "
        f"prompt_tokens={artifact['actual_prompt_tokens']} "
        f"TTFT={artifact['ttft_seconds']}s "
        f"prefill={artifact['prefill_tokens_per_second_e2e']} tok/s "
        f"decode={artifact['decode_tokens_per_second']} tok/s"
    )
    if failure:
        print(f"Failure: {failure}", file=sys.stderr)
    elif not retrieval_ok:
        print("Failure: one or more long-context retrieval needles were missed", file=sys.stderr)
    elif not healthy_after:
        print("Failure: API was not healthy after the request", file=sys.stderr)
    print(f"Wrote {output}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"long-context benchmark error: {exc}", file=sys.stderr)
        raise SystemExit(2)
