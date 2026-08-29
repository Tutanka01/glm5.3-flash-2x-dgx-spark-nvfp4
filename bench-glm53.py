#!/usr/bin/env python3
"""Small OpenAI-compatible TTFT/throughput comparison harness (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    {
        "name": "sanity",
        "messages": [{"role": "user", "content": "Reply with exactly BENCH_OK."}],
        "max_tokens": 64,
    },
    {
        "name": "coding",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a correct Python function merge_intervals(intervals) that merges "
                    "overlapping closed integer intervals. Return only one fenced code block."
                ),
            }
        ],
        "max_tokens": 1024,
    },
    {
        "name": "reasoning",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A service has p99 latency 800 ms. Three independent changes reduce it by "
                    "20%, 15%, and 10% multiplicatively. Compute the final latency and explain briefly."
                ),
            }
        ],
        "max_tokens": 512,
    },
]


@dataclass
class Result:
    target: str
    prompt: str
    run: int
    ok: bool
    error: str | None
    ttft_seconds: float | None
    total_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    decode_tokens_per_second: float | None
    content_sha256: str | None
    content_chars: int
    content: str | None = None  # only with --save-content (A/B quality grading)


def urlopen(request: urllib.request.Request, timeout: int):  # type: ignore[no-untyped-def]
    """Keep loopback benchmark traffic out of inherited HTTP proxies."""

    host = urllib.parse.urlparse(request.full_url).hostname
    if host in {"127.0.0.1", "localhost", "::1"}:
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=timeout
        )
    return urllib.request.urlopen(request, timeout=timeout)


def load_prompts(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or "name" not in value or "messages" not in value:
                raise ValueError(f"invalid prompt on line {line_number}")
            prompts.append(value)
    if not prompts:
        raise ValueError("prompt file is empty")
    return prompts


def stream_once(
    *,
    target_name: str,
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: dict[str, Any],
    run_number: int,
    timeout: int,
    thinking: str = "default",
    save_content: bool = False,
    extra_body: dict[str, Any] | None = None,
) -> Result:
    payload = {
        "model": model,
        "messages": prompt["messages"],
        "temperature": prompt.get("temperature", 0),
        "max_tokens": prompt.get("max_tokens", 512),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if thinking in ("on", "off"):
        # GLM chat templates gate thinking via chat_template_kwargs. Lanes
        # that default thinking ON burn small max_tokens budgets inside
        # <think> and can stream no content deltas at all.
        payload["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
    if extra_body:
        # Per-target body overrides — e.g. OpenRouter's unified reasoning
        # switch, since chat_template_kwargs is vLLM-specific and ignored
        # there. Applied last so it can override anything above.
        payload.update(extra_body)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    started = time.perf_counter()
    first_token_at: float | None = None
    content_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    try:
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
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    piece = delta.get("content") or ""
                    reasoning_piece = delta.get("reasoning_content") or ""
                    tool_piece = delta.get("tool_calls") or []
                    if first_token_at is None and (piece or reasoning_piece or tool_piece):
                        first_token_at = time.perf_counter()
                    if piece:
                        content_parts.append(piece)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        total = time.perf_counter() - started
        error = str(exc)
        if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
            error += (
                " — unknown route or model id on this server; check --model "
                "(and --compare-model) against GET /v1/models"
            )
        return Result(
            target=target_name,
            prompt=prompt["name"],
            run=run_number,
            ok=False,
            error=error,
            ttft_seconds=None,
            total_seconds=total,
            prompt_tokens=None,
            completion_tokens=None,
            decode_tokens_per_second=None,
            content_sha256=None,
            content_chars=0,
        )

    finished = time.perf_counter()
    content = "".join(content_parts)
    ttft = (first_token_at - started) if first_token_at is not None else None
    completion_tokens = usage.get("completion_tokens")
    decode_rate = None
    if (
        isinstance(completion_tokens, int)
        and completion_tokens > 1
        and first_token_at is not None
        and finished > first_token_at
    ):
        # TTFT already includes the first token, so steady-state decode throughput
        # is based on the remaining completion tokens.
        decode_rate = float(completion_tokens - 1) / (finished - first_token_at)
    return Result(
        target=target_name,
        prompt=prompt["name"],
        run=run_number,
        ok=first_token_at is not None,
        error=None if first_token_at is not None else (
            "stream contained no content/reasoning/tool delta"
            + (f" (finish_reason={finish_reason})" if finish_reason else "")
            + " — try --thinking off if the lane defaults thinking on"
        ),
        ttft_seconds=ttft,
        total_seconds=finished - started,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=completion_tokens,
        decode_tokens_per_second=decode_rate,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        content_chars=len(content),
        content=content if save_content else None,
    )


def add_target(
    targets: list[tuple[str, str, str, str | None, dict[str, Any] | None]],
    name: str,
    base_url: str | None,
    model: str | None,
    key_env: str | None,
    extra_body: dict[str, Any] | None = None,
) -> None:
    if not base_url and not model:
        return
    if not base_url or not model:
        raise ValueError(f"{name}: base URL and model must be provided together")
    targets.append(
        (name, base_url, model, os.environ.get(key_env) if key_env else None, extra_body)
    )


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; returns None when there is no data."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="glm-5.3-flash-nvfp4")
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--compare-base-url")
    parser.add_argument("--compare-model")
    parser.add_argument("--compare-api-key-env", default="ZAI_API_KEY")
    parser.add_argument(
        "--compare-extra-body",
        type=json.loads,
        default=None,
        help="extra JSON body params for the compare target only, e.g. "
             '\'{"reasoning": {"enabled": false}}\' to disable reasoning on '
             "OpenRouter (chat_template_kwargs is vLLM-specific and is ignored "
             "there). Applied after --thinking, so it can override it.",
    )
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="discarded single-stream warmups per target (important for JIT speculative kernels)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="number of requests kept in flight per target (sub-agent simulation)",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--thinking",
        choices=["default", "off", "on"],
        default="default",
        help="control enable_thinking via chat_template_kwargs. Use 'off' for "
             "lanes whose chat template defaults thinking on (GLM vLLM lane), "
             "otherwise small max_tokens budgets stream no content deltas.",
    )
    parser.add_argument(
        "--save-content",
        action="store_true",
        help="store raw completions in the artifact (needed for the A/B "
             "quality grader; large artifacts otherwise)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.runs < 1 or args.runs > 20:
        parser.error("--runs must be between 1 and 20")
    if args.warmup_runs < 0 or args.warmup_runs > 5:
        parser.error("--warmup-runs must be between 0 and 5")
    if args.concurrency < 1 or args.concurrency > 64:
        parser.error("--concurrency must be between 1 and 64")
    prompts = load_prompts(args.prompts)
    targets: list[tuple[str, str, str, str | None, dict[str, Any] | None]] = []
    add_target(targets, "local", args.base_url, args.model, args.api_key_env)
    add_target(
        targets,
        "official",
        args.compare_base_url,
        args.compare_model,
        args.compare_api_key_env,
        args.compare_extra_body,
    )

    results: list[Result] = []
    warmup_results: list[Result] = []
    wall_seconds: dict[str, float] = {}
    for target_name, base_url, model, api_key, extra_body in targets:
        for warmup_number in range(1, args.warmup_runs + 1):
            prompt = prompts[(warmup_number - 1) % len(prompts)]
            print(
                f"[{target_name}] warmup {warmup_number}/{args.warmup_runs} "
                f"({prompt['name']}; discarded)",
                flush=True,
            )
            warmup = stream_once(
                target_name=target_name,
                base_url=base_url,
                model=model,
                api_key=api_key,
                prompt=prompt,
                run_number=0,
                timeout=args.timeout,
                thinking=args.thinking,
                save_content=args.save_content,
                extra_body=extra_body,
            )
            warmup_results.append(warmup)
            if not warmup.ok:
                print(
                    f"[{target_name}] warmup failed: {warmup.error}",
                    file=sys.stderr,
                )
                return 1
        tasks = [
            (prompt, run_number)
            for prompt in prompts
            for run_number in range(1, args.runs + 1)
        ]
        target_results: list[Result] = []
        phase_started = time.perf_counter()
        if args.concurrency == 1:
            for prompt, run_number in tasks:
                print(f"[{target_name}] {prompt['name']} run {run_number}/{args.runs}", flush=True)
                target_results.append(
                    stream_once(
                        target_name=target_name,
                        base_url=base_url,
                        model=model,
                        api_key=api_key,
                        prompt=prompt,
                        run_number=run_number,
                        timeout=args.timeout,
                        thinking=args.thinking,
                        save_content=args.save_content,
                        extra_body=extra_body,
                    )
                )
        else:
            print(
                f"[{target_name}] running {len(tasks)} requests with "
                f"concurrency {args.concurrency}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                pending = {
                    pool.submit(
                        stream_once,
                        target_name=target_name,
                        base_url=base_url,
                        model=model,
                        api_key=api_key,
                        prompt=prompt,
                        run_number=run_number,
                        timeout=args.timeout,
                        thinking=args.thinking,
                        save_content=args.save_content,
                        extra_body=extra_body,
                    ): (prompt["name"], run_number)
                    for prompt, run_number in tasks
                }
                for future in as_completed(pending):
                    prompt_name, run_number = pending[future]
                    result = future.result()
                    target_results.append(result)
                    print(
                        f"[{target_name}] {prompt_name} run {run_number}/{args.runs} "
                        f"ok={result.ok} ttft={result.ttft_seconds} "
                        f"total={result.total_seconds:.3f}s "
                        f"tok/s={result.decode_tokens_per_second}",
                        flush=True,
                    )
        wall_seconds[target_name] = time.perf_counter() - phase_started
        results.extend(target_results)

    print("\nMedian summary")
    for target_name, *_ in targets:
        subset = [result for result in results if result.target == target_name and result.ok]
        attempts = sum(r.target == target_name for r in results)
        ttfts = [result.ttft_seconds for result in subset if result.ttft_seconds is not None]
        rates = [
            result.decode_tokens_per_second
            for result in subset
            if result.decode_tokens_per_second is not None
        ]
        totals = [result.total_seconds for result in subset]
        line = (
            f"  {target_name}: success={len(subset)}/{attempts} "
            f"TTFT={median(ttfts)}s (p99={percentile(ttfts, 0.99)}s) "
            f"total={median(totals)}s decode={median(rates)} tok/s"
        )
        # Aggregate goodput only means something when requests actually overlap.
        if args.concurrency > 1 and subset:
            completion_tokens = sum(
                result.completion_tokens
                for result in subset
                if isinstance(result.completion_tokens, int)
            )
            wall = wall_seconds[target_name]
            aggregate = completion_tokens / wall if wall > 0 else 0.0
            line += f" aggregate={aggregate:.1f} tok/s over {wall:.2f}s"
        print(line)

    output = args.output
    if output is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output = Path("results") / f"glm53-benchmark-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "concurrency": args.concurrency,
        "warmup_runs": args.warmup_runs,
        "targets": [
            {"name": name, "base_url": url, "model": model, "extra_body": extra}
            for name, url, model, _, extra in targets
        ],
        "wall_seconds": wall_seconds,
        "results": [asdict(result) for result in results],
        "warmups_discarded": [asdict(result) for result in warmup_results],
    }
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        raise SystemExit(2)
