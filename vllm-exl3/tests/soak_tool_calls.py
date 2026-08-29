#!/usr/bin/env python3
"""Tool-calling soak under concurrent load (upstream issue #10).

Under heavy concurrent load (multi-tool-call turns + large cold prefills)
vLLM sometimes returns `tool_calls` with a function name but **empty required
arguments**. Not reproducible standalone; upstream's own
`GLM53_MIXED_PREFILL_CHUNK=skip` does not eliminate it, so the promotion
checklist requires a client-side validation + retry soak on this kit.

Traffic model (OpenCode-like, stdlib only):
- `--agents` concurrent agents, each running `--turns` sequential tool turns;
- each turn offers an OpenCode-flavored toolset (read/edit/bash/glob) and
  instructs one deterministic call, so schema-conformance is checkable;
- a per-agent salted filler prefix recreates the "large cold prefill mixed
  with interactive tool traffic" trigger from docs/EXL3-KNOWN-ISSUES.md;
- every response is validated client-side: tool known, arguments parse as a
  JSON object, every schema-required key present and non-blank;
- a failing turn is retried with the identical payload (`--retries`) — the
  silent retry reproduced clean output upstream.

Exit codes: 0 = all turns valid (blank-arg events, if any, all recovered by
retry — the accepted mitigation until upstream fixes #10); 2 = transport
errors; 3 = unrecovered blank/invalid tool args (issue #10 reproduced);
4 = internal error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern", "path"],
            },
        },
    },
]
OFFERED = {tool["function"]["name"]: tool["function"] for tool in TOOLS}

# One deterministic scripted call per turn — cycles over this list.
TURN_SCRIPTS: list[tuple[str, dict[str, str]]] = [
    ("read_file", {"path": "/repo/src/agent.py"}),
    ("bash", {"command": "pytest -q tests/test_agent.py"}),
    ("edit_file", {"path": "/repo/src/agent.py", "old_string": "TODO", "new_string": "done"}),
    ("glob", {"pattern": "**/*.py", "path": "/repo/src"}),
]

SYSTEM = (
    "You are a coding agent. When asked to act, respond with exactly one "
    "tool call using the offered tools. Fill every required argument with "
    "the exact values given by the user. SESSION {salt}."
)


def args_to_text(args: dict[str, str]) -> str:
    return ", ".join(f"{key}='{value}'" for key, value in args.items())


def post_chat(
    base_url: str,
    model: str,
    api_key: str | None,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def validate(message: dict[str, Any]) -> str:
    """Return "" when every tool call conforms to its schema, else the reason."""
    calls = message.get("tool_calls") or []
    if not calls:
        return "response carried no tool_calls"
    for call in calls:
        fn = call.get("function") or {}
        name = fn.get("name")
        if name not in OFFERED:
            return f"unknown tool {name!r}"
        raw = fn.get("arguments")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return f"{name}: blank arguments (issue #10 signature)"
        if isinstance(raw, str):
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                return f"{name}: arguments are not valid JSON"
        else:
            args = raw
        if not isinstance(args, dict):
            return f"{name}: arguments are not an object"
        for key in OFFERED[name].get("parameters", {}).get("required", []):
            value = args.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                return f"{name}: blank/missing required argument {key!r}"
    return ""


def run_agent(
    agent: int,
    options: argparse.Namespace,
) -> dict[str, Any]:
    """One agent: `--turns` sequential validated tool turns with retry."""
    salt = f"SOAK-A{agent:02d}-X{options.salt}-{agent * 7919 + options.salt}"
    system = SYSTEM.format(salt=salt)
    filler = ("ctx " * options.filler_words).rstrip()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{system} {filler}".strip()}
    ]
    records: list[dict[str, Any]] = []
    for turn in range(options.turns):
        tool_name, tool_args = TURN_SCRIPTS[turn % len(TURN_SCRIPTS)]
        task = (
            f"Turn {turn}: call the {tool_name} tool with {args_to_text(tool_args)}."
        )
        messages.append({"role": "user", "content": task})
        payload = {
            "model": options.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": options.max_tokens,
            "tools": TOOLS,
            "tool_choice": "auto",
        }
        if options.thinking in ("on", "off"):
            payload["chat_template_kwargs"] = {"enable_thinking": options.thinking == "on"}

        attempts, reason, latency = 0, None, 0.0
        while attempts <= options.retries:
            started = time.perf_counter()
            attempts += 1
            try:
                response = post_chat(
                    options.base_url, options.model, options.api_key, payload, options.timeout
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                reason = f"transport: {type(exc).__name__}: {exc}"
                latency += time.perf_counter() - started
                records.append(
                    {
                        "turn": turn,
                        "tool": tool_name,
                        "ok": False,
                        "reason": reason,
                        "attempts": attempts,
                        "seconds": latency,
                    }
                )
                break
            latency += time.perf_counter() - started
            message = (response.get("choices") or [{}])[0].get("message") or {}
            reason = validate(message)
            if not reason:
                break
        ok = not reason
        records.append(
            {
                "turn": turn,
                "tool": tool_name,
                "ok": ok,
                "reason": reason,
                "attempts": attempts,
                "seconds": round(latency, 3),
            }
        )
        if ok:
            # Grow the conversation like a real agent session would.
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{agent}_{turn}",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{agent}_{turn}",
                    "content": f"ok ({len(filler) + turn} bytes)",
                }
            )
        else:
            break  # unrecovered this turn; stop the agent here
    return {"agent": agent, "records": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--agents", type=int, default=4, help="concurrent agent sessions")
    parser.add_argument("--turns", type=int, default=8, help="tool turns per agent")
    parser.add_argument("--retries", type=int, default=2, help="retries per failing turn")
    parser.add_argument("--filler-words", type=int, default=8000,
                        help="salted filler tokens per agent (cold-prefill trigger)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--salt", type=int, default=1,
                        help="bump to guarantee cold prefixes across re-runs")
    parser.add_argument("--thinking", choices=["default", "off", "on"], default="off")
    parser.add_argument("--out", type=Path, default=Path("results/soak-tool-calls.json"))
    options = parser.parse_args()

    if options.agents < 1 or options.turns < 1:
        parser.error("--agents and --turns must be >= 1")
    options.api_key = os.environ.get(options.api_key_env)

    started = time.perf_counter()
    agents: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=options.agents) as pool:
        futures = [pool.submit(run_agent, agent, options) for agent in range(options.agents)]
        for future in as_completed(futures):
            agents.append(future.result())

    records = [record for agent in agents for record in agent["records"]]
    attempts = sum(record["attempts"] for record in records)
    failed = [record for record in records if not record["ok"]]
    recovered = sum(1 for record in records if record["ok"] and record["attempts"] > 1)
    transport_errors = sum(1 for record in failed if record["reason"].startswith("transport:"))
    invalid_args = sum(1 for record in failed if not record["reason"].startswith("transport:"))
    # Validation-failure events: every failed attempt on unrecovered turns,
    # plus every retried (discarded) attempt on recovered turns.
    blank_arg_events = sum(record["attempts"] for record in failed if not record["reason"].startswith("transport:")) + sum(
        record["attempts"] - 1 for record in records if record["ok"]
    )
    latencies = [record["seconds"] for record in records if record["ok"]]

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": options.base_url,
        "model": options.model,
        "agents": options.agents,
        "turns_per_agent": options.turns,
        "filler_words": options.filler_words,
        "retries_allowed": options.retries,
        "turns_total": len(records),
        "attempts_total": attempts,
        "turns_ok": len(records) - len(failed),
        "blank_arg_events": blank_arg_events,
        "recovered_after_retry": recovered,
        "transport_errors": transport_errors,
        "invalid_args_unrecovered": invalid_args,
        "median_turn_seconds": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "wall_seconds": round(time.perf_counter() - started, 2),
        "agents_detail": agents,
    }

    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"[soak-tool-calls] turns={summary['turns_total']} ok={summary['turns_ok']} "
        f"attempts={attempts} blank_arg_events={summary['blank_arg_events']} "
        f"recovered={recovered} transport_errors={transport_errors} "
        f"unrecovered_invalid_args={invalid_args}"
    )
    print(f"[soak-tool-calls] wrote {options.out}")
    if transport_errors:
        return 2
    if invalid_args:
        print(
            "[soak-tool-calls] FAIL: unrecovered blank/invalid tool arguments — "
            "upstream issue #10 reproduced; keep the client retry in place",
            file=sys.stderr,
        )
        return 3
    if summary["blank_arg_events"]:
        print(
            "[soak-tool-calls] PASS-WITH-RETRIES: blank arguments occurred but every "
            "turn recovered via the client retry (accepted mitigation)"
        )
    else:
        print("[soak-tool-calls] PASS: no blank required arguments")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(4)
