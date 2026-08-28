#!/usr/bin/env python3
"""Cold long-context capacity, retrieval, TTFT and decode benchmark."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


NEEDLES = (
    ("BEGIN_NEEDLE", "ALPHA_7319"),
    ("MIDDLE_NEEDLE", "MIDDLE_2846"),
    ("END_NEEDLE", "OMEGA_9052"),
)
EXPECTED = "|".join(value for _, value in NEEDLES)
FILLER = "Neutral archival padding sentence for context capacity verification.\n"


class BenchmarkError(RuntimeError):
    pass


class TokenizerProbeError(BenchmarkError):
    """Tokenizer discovery failed, optionally for a transient server reason."""

    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


RETRYABLE_HTTP_CODES = frozenset((408, 425, 429, 500, 502, 503, 504))
LONG_CONTEXT_THRESHOLD = 131072
LONG_CONTEXT_MAX_PREFILL_CHUNK = 2048
LONG_CONTEXT_MAX_STATIC_FRACTION = 0.88


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


def inspect_local_runtime(base_url: str) -> tuple[dict[str, str] | None, str]:
    """Read the effective environment of the local head container, if any."""

    if not is_loopback_url(base_url):
        return None, "the benchmark API is not loopback"
    project = os.environ.get("GLM53_PROJECT_NAME", "glm53")
    try:
        probe = subprocess.run(
            [
                "docker",
                "ps",
                "-q",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--filter",
                "label=com.docker.compose.service=sglang-glm53",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"docker inspection failed: {exc}"
    container_ids = probe.stdout.split()
    if not container_ids:
        return None, "no running local glm53 head container was found"
    try:
        inspected = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{json .Config.Env}}",
                container_ids[0],
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = json.loads(inspected.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return None, f"docker environment inspection failed: {exc}"
    if inspected.returncode != 0 or not isinstance(values, list):
        detail = inspected.stderr.strip() or "invalid docker inspect response"
        return None, detail
    runtime: dict[str, str] = {}
    for item in values:
        if isinstance(item, str) and "=" in item:
            key, value = item.split("=", 1)
            runtime[key] = value
    return runtime, f"container={container_ids[0]}"


def long_context_safety_issues(
    runtime: dict[str, str], target_tokens: int
) -> list[str]:
    """Return fail-closed issues for cold prompts above the proven 128k tier."""

    if target_tokens <= LONG_CONTEXT_THRESHOLD:
        return []
    issues: list[str] = []

    def integer(name: str) -> int | None:
        try:
            return int(runtime[name])
        except (KeyError, ValueError):
            issues.append(f"{name} is unknown")
            return None

    max_model_len = integer("MAX_MODEL_LEN")
    max_num_seqs = integer("MAX_NUM_SEQS")
    max_prefill = integer("MAX_NUM_BATCHED_TOKENS")
    disable_graphs = integer("DISABLE_CUDA_GRAPH")
    mtp_tokens = integer("MTP_NUM_TOKENS")
    try:
        static_fraction = float(runtime["MEM_FRACTION_STATIC"])
    except (KeyError, ValueError):
        static_fraction = None
        issues.append("MEM_FRACTION_STATIC is unknown")

    # Leave room for the chat template and the requested completion.
    if max_model_len is not None and target_tokens + 512 > max_model_len:
        issues.append(
            f"MAX_MODEL_LEN={max_model_len} leaves less than 512 tokens of request headroom"
        )
    if max_num_seqs is not None and max_num_seqs != 1:
        issues.append(f"MAX_NUM_SEQS={max_num_seqs}, expected 1")
    if max_prefill is not None and max_prefill > LONG_CONTEXT_MAX_PREFILL_CHUNK:
        issues.append(
            f"MAX_NUM_BATCHED_TOKENS={max_prefill}, safe ceiling is "
            f"{LONG_CONTEXT_MAX_PREFILL_CHUNK}"
        )
    if disable_graphs is not None and disable_graphs != 1:
        issues.append("CUDA graphs are enabled")
    if mtp_tokens is not None and mtp_tokens != 0:
        issues.append(f"MTP_NUM_TOKENS={mtp_tokens}, expected 0")
    if (
        static_fraction is not None
        and static_fraction > LONG_CONTEXT_MAX_STATIC_FRACTION + 1e-9
    ):
        issues.append(
            f"MEM_FRACTION_STATIC={static_fraction:g}, safe ceiling is "
            f"{LONG_CONTEXT_MAX_STATIC_FRACTION:g}"
        )
    return issues


def enforce_long_context_safety(
    base_url: str, target_tokens: int, allow_unsafe_profile: bool
) -> None:
    if target_tokens <= LONG_CONTEXT_THRESHOLD:
        return
    runtime, runtime_detail = inspect_local_runtime(base_url)
    if runtime is None:
        if allow_unsafe_profile:
            print(
                f"WARNING: long-context profile could not be verified ({runtime_detail})",
                file=sys.stderr,
            )
            return
        raise BenchmarkError(
            "refusing a cold prompt above 128k because the effective server profile "
            f"could not be verified ({runtime_detail}). Run this client on the head "
            "node, or pass --allow-unsafe-profile only for an intentionally "
            "unprotected experiment."
        )
    issues = long_context_safety_issues(runtime, target_tokens)
    profile = runtime.get("GLM53_PROFILE_NAME", "unknown")
    if issues and not allow_unsafe_profile:
        raise BenchmarkError(
            f"refusing {target_tokens:,} cold tokens with running profile={profile}: "
            + "; ".join(issues)
            + ". The reliability profile is '256k': stop both ranks, run "
            "./start-glm53.sh 256k, then retry."
        )
    if issues:
        print(
            f"WARNING: bypassing long-context safety checks for profile={profile}: "
            + "; ".join(issues),
            file=sys.stderr,
        )
    else:
        print(
            "Long-context preflight passed: "
            f"profile={profile}, context={runtime.get('MAX_MODEL_LEN')}, "
            f"prefill_chunk={runtime.get('MAX_NUM_BATCHED_TOKENS')}, "
            f"static={runtime.get('MEM_FRACTION_STATIC')}, eager, MTP=off"
        )


def refuse_live_startup_guard() -> None:
    pid_file = Path(__file__).resolve().parent / ".glm53-guard-head.pid"
    try:
        raw_pid = pid_file.read_text(encoding="utf-8").splitlines()[0]
        pid = int(raw_pid)
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        pass

    command = ""
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        pass
    if not command or "oom-guard-node.sh" in command:
        raise BenchmarkError(
            f"startup memory guard pid={pid} is still active. It may stop the "
            "server when the long prefill consumes memory. Wait for "
            "start-glm53.sh to finish or run ./scripts/stop-guard-node.sh head, "
            "then confirm with ./status-glm53.sh before benchmarking."
        )


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
        message = "SGLang tokenizer/engine is temporarily unavailable; tried: "
    else:
        message = "no compatible SGLang tokenizer endpoint found; tried: "
    raise TokenizerProbeError(
        message + "; ".join(details),
        retryable=retryable,
    )


def model_api_status(
    base_url: str,
    model: str | None,
    api_key: str | None,
    timeout: int,
    required_context: int | None = None,
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
    if model is not None and required_context is not None:
        metadata = next(
            (
                item
                for item in payload["data"]
                if isinstance(item, dict) and item.get("id") == model
            ),
            None,
        )
        actual_context = metadata.get("max_model_len") if metadata else None
        if isinstance(actual_context, int) and actual_context < required_context:
            return (
                False,
                f"model context is {actual_context}, but this request needs at least "
                f"{required_context}",
            )
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
                    "SGLang answered but its tokenizer/engine is not ready; "
                    f"waiting up to {readiness_timeout}s (/models: {last_model_status})",
                    file=sys.stderr,
                )
                announced = True
            time.sleep(min(2.0, remaining))

    assert last_error is not None
    elapsed = time.monotonic() - started
    if last_error.retryable:
        raise BenchmarkError(
            f"SGLang did not become usable after {elapsed:.1f}s. "
            f"/models: {last_model_status}. Tokenizer: {last_error}. "
            "HTTP 503 means the HTTP front end is reachable but the inference "
            "engine is loading, stopping, or no longer alive; inspect "
            "./status-glm53.sh and both rank logs before retrying."
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


def flush_cache(base_url: str, api_key: str | None, timeout: int) -> str:
    base = base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    failures: list[str] = []
    for url in (f"{root}/flush_cache", f"{base}/flush_cache"):
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
        except (OSError, urllib.error.URLError) as exc:
            failures.append(f"{url}: {exc}")
    raise BenchmarkError("cold run requested but cache flush failed: " + "; ".join(failures))


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
    if isinstance(completion_tokens, int) and completion_tokens > 1 and finished > first_token_at:
        decode_rate = (completion_tokens - 1) / (finished - first_token_at)
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
    parser.add_argument("--model", default="glm-5.3-flash-nvfp4")
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--readiness-timeout",
        type=int,
        default=60,
        help="seconds to retry transient tokenizer/API failures before aborting",
    )
    parser.add_argument("--cold", action="store_true", help="flush SGLang radix cache first")
    parser.add_argument(
        "--allow-unsafe-profile",
        action="store_true",
        help="bypass the fail-closed profile check above 128k (may crash the engine)",
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
    refuse_live_startup_guard()
    enforce_long_context_safety(
        args.base_url, args.target_tokens, args.allow_unsafe_profile
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
        required_context=args.target_tokens + 512,
    )
    if not model_ready:
        raise BenchmarkError(f"server context preflight failed: {model_status}")

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
    flushed_at = flush_cache(args.base_url, api_key, 60) if args.cold else None
    if flushed_at:
        print(f"Flushed radix cache through {flushed_at}")

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
        output = Path("results") / f"glm53-long-context-{safe_label}-{stamp}.json"
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
