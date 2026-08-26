#!/usr/bin/env python3
"""Small OpenAI-compatible TTFT/throughput comparison harness (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
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
) -> Result:
    payload = {
        "model": model,
        "messages": prompt["messages"],
        "temperature": prompt.get("temperature", 0),
        "max_tokens": prompt.get("max_tokens", 512),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
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
    usage: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
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
                    piece = delta.get("content") or ""
                    reasoning_piece = delta.get("reasoning_content") or ""
                    tool_piece = delta.get("tool_calls") or []
                    if first_token_at is None and (piece or reasoning_piece or tool_piece):
                        first_token_at = time.perf_counter()
                    if piece:
                        content_parts.append(piece)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        total = time.perf_counter() - started
        return Result(
            target=target_name,
            prompt=prompt["name"],
            run=run_number,
            ok=False,
            error=str(exc),
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
        error=None if first_token_at is not None else "stream contained no content/reasoning/tool delta",
        ttft_seconds=ttft,
        total_seconds=finished - started,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=completion_tokens,
        decode_tokens_per_second=decode_rate,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        content_chars=len(content),
    )


def add_target(
    targets: list[tuple[str, str, str, str | None]],
    name: str,
    base_url: str | None,
    model: str | None,
    key_env: str | None,
) -> None:
    if not base_url and not model:
        return
    if not base_url or not model:
        raise ValueError(f"{name}: base URL and model must be provided together")
    targets.append((name, base_url, model, os.environ.get(key_env) if key_env else None))


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="glm-5.3-flash-nvfp4")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--compare-base-url")
    parser.add_argument("--compare-model")
    parser.add_argument("--compare-api-key-env", default="ZAI_API_KEY")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.runs < 1 or args.runs > 20:
        parser.error("--runs must be between 1 and 20")
    prompts = load_prompts(args.prompts)
    targets: list[tuple[str, str, str, str | None]] = []
    add_target(targets, "local", args.base_url, args.model, args.api_key_env)
    add_target(
        targets,
        "official",
        args.compare_base_url,
        args.compare_model,
        args.compare_api_key_env,
    )

    results: list[Result] = []
    for target_name, base_url, model, api_key in targets:
        for prompt in prompts:
            for run_number in range(1, args.runs + 1):
                print(f"[{target_name}] {prompt['name']} run {run_number}/{args.runs}", flush=True)
                result = stream_once(
                    target_name=target_name,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    prompt=prompt,
                    run_number=run_number,
                    timeout=args.timeout,
                )
                results.append(result)
                print(
                    f"  ok={result.ok} ttft={result.ttft_seconds} "
                    f"total={result.total_seconds:.3f}s tok/s={result.decode_tokens_per_second}",
                    flush=True,
                )

    print("\nMedian summary")
    for target_name, *_ in targets:
        subset = [result for result in results if result.target == target_name and result.ok]
        ttfts = [result.ttft_seconds for result in subset if result.ttft_seconds is not None]
        rates = [
            result.decode_tokens_per_second
            for result in subset
            if result.decode_tokens_per_second is not None
        ]
        totals = [result.total_seconds for result in subset]
        print(
            f"  {target_name}: success={len(subset)}/{sum(r.target == target_name for r in results)} "
            f"TTFT={median(ttfts)}s total={median(totals)}s decode={median(rates)} tok/s"
        )

    output = args.output
    if output is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output = Path("results") / f"glm53-benchmark-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "targets": [
            {"name": name, "base_url": url, "model": model} for name, url, model, _ in targets
        ],
        "results": [asdict(result) for result in results],
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
