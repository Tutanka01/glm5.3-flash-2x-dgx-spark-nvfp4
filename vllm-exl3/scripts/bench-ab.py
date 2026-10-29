#!/usr/bin/env python3
"""A/B benchmark and qualification harness for the 2-node vLLM cluster.

New file in this lane.  The protocol (restart per arm, health gate, discarded
warmup, optional prefix-cache reset, boot-identity checks) is inspired by the
Apache-2.0 Reederey87 kit and TensorFold's A/B notes; this implementation is
independent and uses only the Python standard library.

An *arm* is a baseline or candidate configuration expressed as environment
overrides.  The first ``--arm`` is the reference.  Every benchmark artifact is
collected under ``<out-stem>/`` and the comparison is emitted as JSON plus a
Markdown table.

Registered benchmarks
---------------------
``decode_structured``  tests/bench_decode.py --phase structured --structured
``decode_prose``       tests/bench_decode.py --phase prose
``prefix_cache``       tests/bench_prefix_cache.py
``cache_sessions``     tests/bench_cache_sessions.py (new multi-session bench)
``short_chat``         direct HTTP, three short streaming prompts

Add another benchmark by constructing a ``BenchSpec`` and calling
``register_bench``; the CLI, dry-run, artifact collection and comparison pick
it up automatically.  By default every registered benchmark runs; use
``--bench`` (repeatable) to narrow the set.  The two decode benchmarks call the
existing client, which hardcodes ``http://127.0.0.1:8888``; select only the
other benchmarks when using a different ``--base-url``.

Protocol / fail-closed rules
----------------------------
* With restarts enabled (default), each arm runs ``./start.sh restart`` with
  the arm environment merged over the current process environment.
* ``/health`` must return 200 before any measurement.
* If ``GLM53_EXPOSE_CACHE_RESET=1``, ``POST /reset_prefix_cache`` is required;
  a 404 is logged as "endpoint absent" and accepted, anything else fails the
  arm.
* One warmup completion is sent and discarded before the benchmarks.
* Missing health, a non-zero benchmark exit, a missing artifact or malformed
  JSON fail the arm.  Failed arms are never silently compared.
* The engine boot id is read from ``/metrics`` (process start time or a
  ``boot_id`` label), falling back to the host boot id.  If the restart
  protocol is enabled but arms share a boot, or no boot id can be established,
  the report warns loudly.

Usage
-----
    python3 scripts/bench-ab.py \\
        --arm baseline \\
        --arm candidate:MAX_NUM_SEQS=6,GLM53_MIXED_PREFILL_CHUNK=448 \\
        --base-url http://127.0.0.1:8888 --workdir vllm-exl3 \\
        --runs 3 --out results/ab-<stamp>.json --markdown results/ab-<stamp>.md

    python3 scripts/bench-ab.py --dry-run --bench short_chat --no-restart
    python3 scripts/bench-ab.py --list-benches
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_BASE_URL = "http://127.0.0.1:8888"
DEFAULT_MODEL = "GLM-5.3-Flash-EXL3"
DEFAULT_WORKDIR = "vllm-exl3"
DEFAULT_RUNS = 3
DEFAULT_TIMEOUT = 900.0
DEFAULT_HEALTH_TIMEOUT = 1800.0
DEFAULT_HEALTH_INTERVAL = 5.0
DEFAULT_RESTART_TIMEOUT = 7200.0
DEFAULT_SESSIONS = 4
DEFAULT_CTX_TOKENS = 60000
SHORT_CHAT_PROMPTS = (
    "Reply with exactly: OK",
    "What is 2 + 2? Answer with one digit.",
    "Name the capital of France in one word.",
)

SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_DIR = SCRIPT_DIR.parent
REPO_ROOT = VLLM_DIR.parent

_SENSITIVE_ENV_RE = re.compile(
    r"(API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.I
)
_RELEVANT_ENV_PREFIXES = (
    "GLM53_",
    "VLLM_",
    "EXL3_",
    "MTP_",
    "MAX_",
    "SERVED_",
    "SPEC_",
    "ENFORCE_",
    "GPU_",
    "MODEL",
    "IMAGE",
    "HEAD_IP",
    "WORKER_IP",
    "PORT",
    "READY_",
    "EXTRA_ARGS",
    "ABLIT",
)
_ARM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROM_LINE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|NaN|[+-]?Inf)"
    r"(?:\s+[+-]?\d+(?:\.\d+)?)?\s*$"
)
_BOOT_LABEL_RE = re.compile(r'\bboot_id="([^"]+)"')
_START_METRIC_RE = re.compile(
    r"(?:process|engine|server).*start_time_seconds$|"
    r"start_time_seconds.*(?:process|engine|server)$"
)


class BenchAbError(RuntimeError):
    """Configuration or protocol failure in the A/B harness."""


class ArmSpecError(BenchAbError):
    """Invalid --arm specification."""


class BenchArtifactError(BenchAbError):
    """Missing, non-zero or malformed benchmark artifact."""


class ArmRunError(BenchAbError):
    """A fatal per-arm protocol failure (restart, health, warmup, reset)."""


# ---------------------------------------------------------------------------
# arm parsing / environment handling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    name: str
    env: dict[str, str]
    spec: str


def parse_arm(spec: str) -> Arm:
    """Parse ``NAME`` or ``NAME:KEY=VALUE,KEY=VALUE``.

    Commas and newlines are not supported inside values; this matches the
    documented CLI shape and keeps the arm line unambiguous.
    """
    raw = spec.strip()
    if not raw:
        raise ArmSpecError("empty --arm specification")
    name, separator, env_text = raw.partition(":")
    name = name.strip()
    if not name:
        raise ArmSpecError(f"missing arm name in {spec!r}")
    if not _ARM_NAME_RE.match(name):
        raise ArmSpecError(f"invalid arm name {name!r}")
    if separator and not env_text.strip():
        raise ArmSpecError(f"arm {name!r} has an empty environment override list")

    env: dict[str, str] = {}
    if separator:
        for item in env_text.split(","):
            piece = item.strip()
            if not piece:
                raise ArmSpecError(f"arm {name!r} has an empty override segment")
            key, equals, value = piece.partition("=")
            key = key.strip()
            if not equals:
                raise ArmSpecError(f"arm {name!r}: override {piece!r} is missing '='")
            if not _ENV_KEY_RE.match(key):
                raise ArmSpecError(f"arm {name!r}: invalid environment key {key!r}")
            if key in env:
                raise ArmSpecError(f"arm {name!r}: duplicate override for {key}")
            if "\n" in value or "\r" in value:
                raise ArmSpecError(f"arm {name!r}: newline in value for {key}")
            env[key] = value
    return Arm(name=name, env=env, spec=raw)


def parse_arms(specs: Sequence[str]) -> list[Arm]:
    """Parse arms in order; the first one is the baseline."""
    raw_specs = list(specs) or ["baseline"]
    arms = [parse_arm(spec) for spec in raw_specs]
    seen: set[str] = set()
    for arm in arms:
        if arm.name in seen:
            raise ArmSpecError(f"duplicate arm name {arm.name!r}")
        seen.add(arm.name)
    return arms


def merge_env(base: Mapping[str, str], overrides: Mapping[str, str]) -> dict[str, str]:
    """Return a new environment with overrides applied (base is untouched)."""
    merged = {str(key): str(value) for key, value in base.items()}
    merged.update({str(key): str(value) for key, value in overrides.items()})
    return merged


def is_sensitive_key(key: str) -> bool:
    return bool(_SENSITIVE_ENV_RE.search(key))


def redact_env(env: Mapping[str, str]) -> dict[str, str]:
    return {
        key: ("<redacted>" if is_sensitive_key(key) else value)
        for key, value in env.items()
    }


def filter_effective_env(env: Mapping[str, str]) -> dict[str, str]:
    """Keep only bench-relevant knobs; secret values are redacted."""
    selected = {
        key: value
        for key, value in env.items()
        if key in ("MODEL", "IMAGE") or key.startswith(_RELEVANT_ENV_PREFIXES)
    }
    return redact_env(selected)


def _secret_values(env: Mapping[str, str], api_key: str = "") -> list[str]:
    values = {
        value
        for key, value in env.items()
        if is_sensitive_key(key) and len(value) >= 4
    }
    if len(api_key) >= 4:
        values.add(api_key)
    return sorted(values, key=len, reverse=True)


def redact_text(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


@dataclass
class HttpResponse:
    code: int
    body: bytes
    error: str | None = None

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8", "replace"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BenchAbError(f"malformed JSON response: {exc}") from exc


def api_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def chat_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return root + "/chat/completions"
    return root + "/v1/chat/completions"


def http_request(
    url: str,
    *,
    method: str = "GET",
    payload: Any = None,
    timeout: float = 30.0,
    api_key: str = "",
) -> HttpResponse:
    """Small urllib wrapper: HTTP error codes are returned, not raised."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(code=int(response.status), body=response.read())
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except OSError:
            body = b""
        return HttpResponse(code=int(exc.code), body=body)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return HttpResponse(code=0, body=b"", error=f"connection error: {exc}")


def wait_for_health(
    base_url: str,
    *,
    timeout: float,
    interval: float,
    api_key: str = "",
) -> dict[str, Any]:
    """Poll /health until 200 or the deadline.  Never raises on connection errors."""
    if timeout <= 0:
        raise BenchAbError("health timeout must be > 0")
    if interval <= 0:
        raise BenchAbError("health interval must be > 0")
    url = api_root(base_url) + "/health"
    deadline = time.monotonic() + timeout
    attempts = 0
    last_code = 0
    last_error = ""
    started = time.monotonic()
    while True:
        attempts += 1
        result = http_request(url, timeout=min(10.0, max(1.0, timeout)), api_key=api_key)
        last_code = result.code
        last_error = result.error or ""
        if result.code == 200:
            return {
                "ok": True,
                "url": url,
                "code": 200,
                "attempts": attempts,
                "seconds": round(time.monotonic() - started, 3),
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
    return {
        "ok": False,
        "url": url,
        "code": last_code,
        "attempts": attempts,
        "seconds": round(time.monotonic() - started, 3),
        "error": last_error or f"last HTTP status {last_code}",
    }


def reset_prefix_cache(base_url: str, *, timeout: float, api_key: str = "") -> dict[str, Any]:
    """POST /reset_prefix_cache; 404 means the route is absent (accepted)."""
    url = api_root(base_url) + "/reset_prefix_cache"
    result = http_request(url, method="POST", payload={}, timeout=timeout, api_key=api_key)
    if result.code in (200, 204):
        return {"status": "ok", "endpoint": url, "http": result.code}
    if result.code == 404:
        return {
            "status": "absent",
            "endpoint": url,
            "http": 404,
            "note": "endpoint absent (404); cold measurements rely on unique prompts",
        }
    return {
        "status": "error",
        "endpoint": url,
        "http": result.code,
        "error": result.error or result.body.decode("utf-8", "replace")[:500],
    }


def warmup_request(
    base_url: str,
    *,
    model: str,
    timeout: float,
    api_key: str = "",
) -> dict[str, Any]:
    """Send and discard one short completion."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    result = http_request(
        chat_url(base_url), method="POST", payload=payload, timeout=timeout, api_key=api_key
    )
    wall = time.perf_counter() - started
    if result.code != 200:
        return {
            "ok": False,
            "http": result.code,
            "wall_s": round(wall, 3),
            "error": result.error or result.body.decode("utf-8", "replace")[:300],
        }
    try:
        body = result.json()
    except BenchAbError as exc:
        return {"ok": False, "http": result.code, "wall_s": round(wall, 3), "error": str(exc)}
    usage = body.get("usage") if isinstance(body, dict) else None
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return {
            "ok": False,
            "http": result.code,
            "wall_s": round(wall, 3),
            "error": "warmup response has no choices",
        }
    return {
        "ok": True,
        "http": result.code,
        "wall_s": round(wall, 3),
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
    }


# ---------------------------------------------------------------------------
# boot identity / git / host
# ---------------------------------------------------------------------------


def host_boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def probe_boot_id(base_url: str, *, timeout: float, api_key: str = "") -> dict[str, Any]:
    """Identify the engine boot from /metrics, falling back to the host boot id."""
    url = api_root(base_url) + "/metrics"
    result = http_request(url, timeout=timeout, api_key=api_key)
    metric_boot: str | None = None
    process_start: float | None = None
    error = result.error or ""
    if result.code == 200:
        text = result.body.decode("utf-8", "replace")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            label_match = _BOOT_LABEL_RE.search(line)
            if label_match and metric_boot is None:
                metric_boot = label_match.group(1)
            match = _PROM_LINE_RE.match(line)
            if not match:
                continue
            name, _labels, raw_value = match.groups()
            if not _START_METRIC_RE.search(name):
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            if math.isfinite(value) and process_start is None:
                process_start = value
        if metric_boot:
            return {"id": f"metrics-boot:{metric_boot}", "source": "metrics-label"}
        if process_start is not None:
            return {
                "id": f"process-start:{int(process_start)}",
                "source": "metrics-process-start",
            }
    else:
        error = error or f"HTTP {result.code}"

    host_id = host_boot_id()
    if host_id:
        return {
            "id": f"host-boot:{host_id}",
            "source": "host-boot",
            "note": "engine start metric unavailable; host boot id cannot prove an engine restart",
            "error": error,
        }
    return {
        "id": None,
        "source": "unavailable",
        "error": error or "no process_start_time_seconds and no host boot id",
    }


def boot_warnings(arms: Sequence[dict[str, Any]], *, restart_required: bool) -> list[str]:
    """Warn when boot identity cannot support the configured protocol."""
    warnings: list[str] = []
    boots = []
    for arm in arms:
        boot = arm.get("boot") if isinstance(arm.get("boot"), dict) else {}
        boots.append((str(arm.get("name", "?")), boot.get("id"), boot.get("source")))

    if restart_required:
        missing = [name for name, boot_id, _source in boots if not boot_id]
        if missing:
            warnings.append(
                "engine boot id unavailable for arm(s) "
                + ", ".join(missing)
                + "; cannot prove each arm used a distinct restart"
            )
        host_only = [
            name for name, boot_id, source in boots if boot_id and source == "host-boot"
        ]
        if host_only:
            warnings.append(
                "only host boot id available for arm(s) "
                + ", ".join(host_only)
                + "; an engine restart inside the same host boot is invisible"
            )
        known = [(name, boot_id) for name, boot_id, _source in boots if boot_id]
        if len(known) >= 2 and len({boot_id for _name, boot_id in known}) == 1:
            warnings.append(
                "all arms report the same boot identity ("
                + str(known[0][1])
                + "); the restart protocol may not have taken effect"
            )
    else:
        warnings.append(
            "--no-restart: arm environment overrides are not applied by the harness; "
            "measurements are only valid if the server was provisioned externally"
        )
        known_ids = {boot_id for _name, boot_id, _source in boots if boot_id}
        if len(known_ids) > 1:
            warnings.append(
                "--no-restart: the engine boot changed between arms outside harness control"
            )
    return warnings


def git_info(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        status = run("status", "--porcelain")
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "branch": None, "dirty": None, "error": str(exc)}
    if commit.returncode != 0:
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
            "error": (commit.stderr or "git rev-parse failed").strip()[:500],
        }
    status_lines = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "dirty": bool(status_lines),
        "dirty_count": len(status_lines),
        "status_head": status_lines[:50],
    }


def host_info() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "boot_id": host_boot_id(),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# benchmark registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchContext:
    name: str
    base_url: str
    model: str
    runs: int
    timeout: float
    artifact_path: Path
    tests_dir: Path
    workdir: Path
    sessions: int
    ctx_tokens: int
    cache_mode: str
    api_key: str


@dataclass
class BenchMetrics:
    samples: dict[str, list[float]]
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "summary": self.summary,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class BenchSpec:
    name: str
    description: str
    extract: Callable[[dict[str, Any]], BenchMetrics]
    direction: dict[str, str]
    build_argv: Callable[[BenchContext], list[str]] | None = None
    run_direct: Callable[[BenchContext], dict[str, Any]] | None = None
    dry_run_hint: str = ""
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.build_argv is None) == (self.run_direct is None):
            raise ValueError(f"{self.name}: exactly one of build_argv/run_direct is required")


BENCHMARKS: dict[str, BenchSpec] = {}


def register_bench(spec: BenchSpec) -> BenchSpec:
    if spec.name in BENCHMARKS:
        raise ValueError(f"duplicate benchmark name {spec.name!r}")
    BENCHMARKS[spec.name] = spec
    return spec


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _numbers(records: Sequence[Any], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        number = _number(record.get(key))
        if number is not None:
            values.append(number)
    return values


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _require_list(artifact: dict[str, Any], key: str, bench: str) -> list[Any]:
    value = artifact.get(key)
    if not isinstance(value, list) or not value:
        raise BenchArtifactError(f"{bench}: artifact has no non-empty {key!r} list")
    return value


def _decode_build(structured: bool) -> Callable[[BenchContext], list[str]]:
    def build(context: BenchContext) -> list[str]:
        if api_root(context.base_url) != api_root(DEFAULT_BASE_URL):
            raise BenchAbError(
                "tests/bench_decode.py hardcodes http://127.0.0.1:8888; "
                "use --base-url http://127.0.0.1:8888 for decode benches"
            )
        argv = [
            sys.executable,
            str(context.tests_dir / "bench_decode.py"),
            "--phase",
            "structured" if structured else "prose",
            "--runs",
            str(context.runs),
            "--max-tokens",
            "400",
            "--skip-coherence",
            "--out",
            str(context.artifact_path),
        ]
        if structured:
            argv.append("--structured")
        return argv

    return build


def _extract_decode(artifact: dict[str, Any]) -> BenchMetrics:
    runs = _require_list(artifact, "runs", "decode")
    tok_s = _numbers(runs, "tok_s")
    if not tok_s:
        raise BenchArtifactError("decode: no numeric tok_s samples in artifact")
    ttft = _numbers(runs, "ttft_s")
    completion = _numbers(runs, "completion_tokens")
    notes: list[str] = []
    if any(bool(run.get("nan")) for run in runs if isinstance(run, dict)):
        notes.append("decode artifact reports NaN/locklock text")
    summary = {
        "decode_tok_s_median": _median(tok_s),
        "decode_tok_s_min": min(tok_s),
        "decode_tok_s_max": max(tok_s),
        "ttft_median_s": _median(ttft),
        "completion_tokens_median": _median(completion),
        "any_nan": bool(artifact.get("any_nan")),
        "runs": len(runs),
    }
    return BenchMetrics(
        samples={"decode_tok_s": tok_s, "ttft_s": ttft},
        summary=summary,
        notes=notes,
    )


def _v1_base(base_url: str) -> str:
    """Return a base ending in /v1 for clients that append /chat/completions."""
    root = api_root(base_url)
    return root + "/v1"


def _prefix_build(context: BenchContext) -> list[str]:
    return [
        sys.executable,
        str(context.tests_dir / "bench_prefix_cache.py"),
        "--base-url",
        _v1_base(context.base_url),
        "--model",
        context.model,
        "--runs",
        str(context.runs),
        "--timeout",
        str(context.timeout),
        "--output",
        str(context.artifact_path),
    ]


def _pair_metric(pair: dict[str, Any], key: str) -> float | None:
    value = _number(pair.get(key))
    if value is not None:
        return value
    nested = pair.get(key.removesuffix("_ttft_s"))
    if isinstance(nested, dict):
        return _number(nested.get("wall_s"))
    return None


def _extract_prefix(artifact: dict[str, Any]) -> BenchMetrics:
    pairs = _require_list(artifact, "pairs", "prefix_cache")
    dict_pairs = [pair for pair in pairs if isinstance(pair, dict)]
    if not dict_pairs:
        raise BenchArtifactError("prefix_cache: pairs are not objects")
    cold = [value for pair in dict_pairs if (value := _pair_metric(pair, "cold_ttft_s")) is not None]
    warm = [value for pair in dict_pairs if (value := _pair_metric(pair, "warm_ttft_s")) is not None]
    ratios = [
        value
        for pair in dict_pairs
        if (value := _number(pair.get("prefix_hit_ratio"))) is not None
    ]
    efficiency = _numbers(dict_pairs, "hit_efficiency")
    speedup = _numbers(dict_pairs, "speedup")
    if not (cold or warm or ratios):
        raise BenchArtifactError("prefix_cache: no numeric timing or hit-ratio samples")
    notes: list[str] = []
    if not ratios:
        notes.append("prefix_hit_ratio unavailable (server did not expose cache metrics)")
    return BenchMetrics(
        samples={
            "cold_ttft_s": cold,
            "warm_ttft_s": warm,
            "prefix_hit_ratio": ratios,
            "hit_efficiency": efficiency,
            "speedup": speedup,
        },
        summary={
            "pairs": len(dict_pairs),
            "cold_ttft_median_s": _median(cold),
            "warm_ttft_median_s": _median(warm),
            "prefix_hit_ratio_median": _median(ratios),
            "hit_efficiency_median": _median(efficiency),
        },
        notes=notes,
    )


def _cache_sessions_build(context: BenchContext) -> list[str]:
    mode_flags = {
        "sessions": [],
        "shared-prefix": ["--shared-prefix"],
        "sub-page": ["--sub-page"],
    }
    if context.cache_mode not in mode_flags:
        raise BenchAbError(f"unknown cache-sessions mode {context.cache_mode!r}")
    return [
        sys.executable,
        str(context.tests_dir / "bench_cache_sessions.py"),
        "--base-url",
        context.base_url,
        "--model",
        context.model,
        "--sessions",
        str(context.sessions),
        "--ctx-tokens",
        str(context.ctx_tokens),
        "--rounds",
        str(context.runs),
        "--timeout",
        str(context.timeout),
        "--out",
        str(context.artifact_path),
        *mode_flags[context.cache_mode],
    ]


def _extract_cache_sessions(artifact: dict[str, Any]) -> BenchMetrics:
    rounds = _require_list(artifact, "rounds", "cache_sessions")
    warm_hits: list[float] = []
    cold_hits: list[float] = []
    warm_ttft: list[float] = []
    warm_decode: list[float] = []
    warm_wall: list[float] = []
    for position, round_record in enumerate(rounds):
        if not isinstance(round_record, dict):
            raise BenchArtifactError("cache_sessions: round entry is not an object")
        number = _number(round_record.get("round"))
        round_number = int(number) if number is not None else position + 1
        hit = _number(round_record.get("hit_fraction"))
        if round_number == 1:
            if hit is not None:
                cold_hits.append(hit)
            continue
        if hit is not None:
            warm_hits.append(hit)
        wall = _number(round_record.get("wall_s"))
        if wall is not None:
            warm_wall.append(wall)
        sessions = round_record.get("sessions")
        if not isinstance(sessions, list):
            raise BenchArtifactError("cache_sessions: round has no sessions list")
        warm_ttft.extend(_numbers(sessions, "ttft_s"))
        warm_decode.extend(_numbers(sessions, "decode_tok_s"))
    if not (cold_hits or warm_hits or warm_ttft or warm_decode):
        raise BenchArtifactError("cache_sessions: no numeric round metrics")
    summary = artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
    notes = [
        str(warning)
        for warning in (artifact.get("warnings") or [])
        if str(warning).strip()
    ]
    if not warm_hits:
        notes.append("no warm cached_tokens; hit-based verdict is unknown")
    return BenchMetrics(
        samples={
            "cold_hit_fraction": cold_hits,
            "warm_hit_fraction": warm_hits,
            "warm_ttft_s": warm_ttft,
            "warm_decode_tok_s": warm_decode,
            "warm_wall_s": warm_wall,
        },
        summary={
            "verdict": summary.get("verdict") or artifact.get("verdict"),
            "cold_hit_fraction_median": _median(cold_hits),
            "warm_hit_fraction_median": summary.get("warm_hit_fraction_median"),
            "warm_ttft_median_s": summary.get("warm_ttft_median_s"),
            "warm_decode_tok_s_median": summary.get("warm_decode_tok_s_median"),
            "http_ok": summary.get("http_ok"),
        },
        notes=notes,
    )


def _load_cache_sessions_module(tests_dir: Path) -> Any:
    path = tests_dir / "bench_cache_sessions.py"
    if not path.is_file():
        raise BenchAbError(f"missing benchmark client {path}")
    name = "bench_ab_cache_sessions_runtime"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BenchAbError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.12 dataclasses look the defining module up in sys.modules.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _short_chat_run(context: BenchContext) -> dict[str, Any]:
    module = _load_cache_sessions_module(context.tests_dir)
    requests: list[dict[str, Any]] = []
    for index, prompt in enumerate(SHORT_CHAT_PROMPTS):
        result = module.chat_request(
            context.base_url,
            context.model,
            [{"role": "user", "content": prompt}],
            api_key=context.api_key,
            timeout=context.timeout,
            max_tokens=64,
            stream=True,
        )
        record = result.to_dict()
        record["prompt"] = prompt
        requests.append(record)
        if not result.ok:
            raise BenchAbError(
                f"short_chat prompt {index + 1} failed: {result.error or result.http}"
            )
    ttft = _numbers(requests, "ttft_s")
    decode = _numbers(requests, "decode_tok_s")
    if not ttft and not decode:
        raise BenchAbError("short_chat produced no TTFT or decode samples")
    return {
        "schema": "glm53-exl3-short-chat/1",
        "created_at": utc_now(),
        "base_url": context.base_url,
        "model": context.model,
        "requests": requests,
        "ttft_median_s": _median(ttft),
        "decode_tok_s_median": _median(decode),
    }


def _extract_short_chat(artifact: dict[str, Any]) -> BenchMetrics:
    requests = _require_list(artifact, "requests", "short_chat")
    ttft = _numbers(requests, "ttft_s")
    decode = _numbers(requests, "decode_tok_s")
    if not ttft and not decode:
        raise BenchArtifactError("short_chat: no numeric TTFT/decode samples")
    notes: list[str] = []
    if not ttft:
        notes.append("server returned non-SSE bodies; no first-token TTFT")
    return BenchMetrics(
        samples={"ttft_s": ttft, "decode_tok_s": decode},
        summary={
            "ttft_median_s": _median(ttft),
            "decode_tok_s_median": _median(decode),
            "requests": len(requests),
        },
        notes=notes,
    )


register_bench(
    BenchSpec(
        name="decode_structured",
        description="warmed structured decode (count 1..200), 400 tokens",
        build_argv=_decode_build(True),
        extract=_extract_decode,
        direction={"decode_tok_s": "higher", "ttft_s": "lower"},
        requires=("bench_decode.py",),
    )
)
register_bench(
    BenchSpec(
        name="decode_prose",
        description="warmed prose decode (hash-map explanation), 400 tokens",
        build_argv=_decode_build(False),
        extract=_extract_decode,
        direction={"decode_tok_s": "higher", "ttft_s": "lower"},
        requires=("bench_decode.py",),
    )
)
register_bench(
    BenchSpec(
        name="prefix_cache",
        description="cold/warm prefix-cache pairs and hit ratio",
        build_argv=_prefix_build,
        extract=_extract_prefix,
        direction={
            "cold_ttft_s": "lower",
            "warm_ttft_s": "lower",
            "prefix_hit_ratio": "higher",
            "hit_efficiency": "higher",
            "speedup": "higher",
        },
        requires=("bench_prefix_cache.py",),
    )
)
register_bench(
    BenchSpec(
        name="cache_sessions",
        description="multi-session co-batched prefix-cache rounds",
        build_argv=_cache_sessions_build,
        extract=_extract_cache_sessions,
        direction={
            "cold_hit_fraction": "neutral",
            "warm_hit_fraction": "higher",
            "warm_ttft_s": "lower",
            "warm_decode_tok_s": "higher",
            "warm_wall_s": "lower",
        },
        requires=("bench_cache_sessions.py",),
    )
)
register_bench(
    BenchSpec(
        name="short_chat",
        description="three short streaming prompts, direct HTTP",
        run_direct=_short_chat_run,
        extract=_extract_short_chat,
        direction={"ttft_s": "lower", "decode_tok_s": "higher"},
        dry_run_hint="direct HTTP: 3 short streaming prompts, 64 tokens each",
        requires=("bench_cache_sessions.py",),
    )
)


def parse_bench_selection(names: Sequence[str]) -> list[str]:
    if not names:
        return list(BENCHMARKS)
    selected: list[str] = []
    for name in names:
        if name not in BENCHMARKS:
            raise BenchAbError(f"unknown benchmark {name!r}")
        if name not in selected:
            selected.append(name)
    return selected


# ---------------------------------------------------------------------------
# artifact loading / statistics
# ---------------------------------------------------------------------------


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BenchArtifactError(f"benchmark artifact missing: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchArtifactError(f"cannot read benchmark artifact {path}: {exc}") from exc
    try:
        artifact = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchArtifactError(
            f"malformed JSON in {path}: line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(artifact, dict):
        raise BenchArtifactError(f"benchmark artifact {path} is not a JSON object")
    return artifact


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise BenchArtifactError(f"cannot write benchmark artifact {path}: {exc}") from exc


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Nearest-rank empirical percentile (p99 == max for n < 100)."""
    if not values:
        return None
    if not 0 < quantile <= 100:
        raise ValueError("quantile must be in (0, 100]")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(quantile / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def numeric_stats(values: Sequence[float]) -> dict[str, Any]:
    clean = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    if not clean:
        return {
            "n": 0,
            "median": None,
            "p99": None,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": len(clean),
        "median": statistics.median(clean),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.fmean(clean),
    }


def _delta_pct(baseline: float | None, candidate: float | None, direction: str) -> float | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    if direction == "lower":
        return (baseline - candidate) / abs(baseline) * 100.0
    return (candidate - baseline) / abs(baseline) * 100.0


def build_comparison(arms: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build per-metric medians/p99 and candidate deltas.

    Only successful arms contribute.  The baseline is the first arm; if it
    failed, deltas are suppressed rather than comparing against a substitute.
    """
    baseline_arm = arms[0] if arms else None
    baseline_name = str(baseline_arm.get("name")) if baseline_arm else None
    baseline_ok = bool(baseline_arm and baseline_arm.get("ok"))
    per_metric: dict[str, dict[str, Any]] = {}
    directions: dict[str, str] = {}

    for arm in arms:
        if not arm.get("ok"):
            continue
        arm_name = str(arm.get("name"))
        for bench in arm.get("benches", []):
            if not isinstance(bench, dict) or not bench.get("ok"):
                continue
            bench_name = str(bench.get("name"))
            spec = BENCHMARKS.get(bench_name)
            metrics = bench.get("metrics")
            if spec is None or not isinstance(metrics, dict):
                continue
            samples = metrics.get("samples")
            if not isinstance(samples, dict):
                continue
            for sample_name, values in samples.items():
                if not isinstance(values, list):
                    continue
                clean = [
                    float(value)
                    for value in values
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ]
                if not clean:
                    continue
                key = f"{bench_name}.{sample_name}"
                directions.setdefault(key, spec.direction.get(sample_name, "neutral"))
                per_metric.setdefault(key, {})[arm_name] = numeric_stats(clean)

    rows = []
    for key in sorted(per_metric):
        arm_stats = per_metric[key]
        direction = directions.get(key, "neutral")
        baseline_stats = arm_stats.get(baseline_name) if baseline_name else None
        deltas: dict[str, float | None] = {}
        for arm in arms:
            arm_name = str(arm.get("name"))
            stats = arm_stats.get(arm_name)
            if (
                baseline_ok
                and arm_name != baseline_name
                and baseline_stats is not None
                and stats is not None
            ):
                deltas[arm_name] = _delta_pct(
                    baseline_stats.get("median"), stats.get("median"), direction
                )
        rows.append(
            {
                "metric": key,
                "direction": direction,
                "baseline": baseline_stats,
                "arms": arm_stats,
                "deltas": deltas,
            }
        )
    return {
        "baseline": baseline_name,
        "baseline_ok": baseline_ok,
        "metrics": per_metric,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# dry run / markdown
# ---------------------------------------------------------------------------


def artifact_dir_for(out: Path) -> Path:
    return out.parent / out.stem if out.suffix else out.parent / f"{out.name}-artifacts"


def _benchmark_argv(spec: BenchSpec, context: BenchContext) -> list[str] | None:
    if spec.build_argv is None:
        return None
    return spec.build_argv(context)


def print_dry_run(
    arms: Sequence[Arm],
    benches: Sequence[str],
    args: argparse.Namespace,
    *,
    workdir: Path,
    tests_dir: Path,
    out: Path,
    markdown: Path,
    api_key: str,
) -> None:
    print("[bench-ab] dry run - no restart, no HTTP, no artifact writes")
    print(f"[bench-ab] workdir={workdir}")
    print(f"[bench-ab] out={out}")
    print(f"[bench-ab] markdown={markdown}")
    print(f"[bench-ab] base_url={args.base_url} model={args.model} runs={args.runs}")
    print(f"[bench-ab] auth={'configured' if api_key else 'none'}")
    print(f"[bench-ab] benchmarks={','.join(benches)}")
    artifact_dir = artifact_dir_for(out)
    for index, arm in enumerate(arms):
        role = "baseline" if index == 0 else "candidate"
        print(f"\n[bench-ab] arm {index + 1}/{len(arms)} {arm.name} ({role})")
        print(f"  overrides={json.dumps(redact_env(arm.env), sort_keys=True)}")
        print(
            "  restart="
            + ("skipped (--no-restart)" if args.no_restart else "./start.sh restart")
        )
        print(f"  health=GET {api_root(args.base_url)}/health")
        print(
            f"  reset=POST {api_root(args.base_url)}/reset_prefix_cache "
            "(only when GLM53_EXPOSE_CACHE_RESET=1)"
        )
        print(f"  warmup=POST {chat_url(args.base_url)} (discarded)")
        for bench in benches:
            spec = BENCHMARKS[bench]
            context = BenchContext(
                name=bench,
                base_url=args.base_url,
                model=args.model,
                runs=args.runs,
                timeout=args.timeout,
                artifact_path=artifact_dir / f"{_slug(arm.name)}__{bench}.json",
                tests_dir=tests_dir,
                workdir=workdir,
                sessions=args.sessions,
                ctx_tokens=args.ctx_tokens,
                cache_mode=args.cache_mode,
                api_key=api_key,
            )
            argv = _benchmark_argv(spec, context)
            if argv is None:
                print(f"  bench {bench}: {spec.dry_run_hint}")
            else:
                print(f"  bench {bench}: {shlex.join(argv)}")


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return slug or "arm"


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4g}"


def _fmt_delta(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1f}%"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    comparison = report.get("comparison") or {}
    arms = report.get("arms") or []
    baseline = comparison.get("baseline") or "baseline"
    candidates = [str(arm.get("name")) for arm in arms[1:]]
    git = report.get("git") or {}
    protocol = report.get("protocol") or {}

    lines = [
        f"# A/B benchmark - {report.get('stamp', '')}",
        "",
        f"- created: `{report.get('created_at')}`",
        f"- base URL: `{report.get('base_url')}`",
        f"- workdir: `{report.get('workdir')}`",
        f"- git: `{git.get('commit')}` ({git.get('branch')}), "
        f"dirty={git.get('dirty')}",
        f"- runs: {protocol.get('runs')}; restart per arm: {protocol.get('restart')}; "
        f"warmup discarded: {protocol.get('warmup')}",
        "",
        "## Warnings",
        "",
    ]
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(f"- {_cell(warning)}" for warning in warnings)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Comparison",
            "",
            "Medians and empirical p99 (nearest rank; p99 equals max for n < 100). "
            "Deltas are relative to the baseline median; positive means better for "
            "higher/lower metrics, neutral rows are informational.",
            "",
        ]
    )
    rows = comparison.get("rows") or []
    if not rows:
        lines.append("_No comparable metrics: no successful arm produced numeric samples._")
    else:
        header = ["metric", "direction", f"{baseline} median", f"{baseline} p99"]
        for candidate in candidates:
            header.extend([f"{candidate} median", f"{candidate} p99", f"{candidate} delta"])
        lines.append("| " + " | ".join(_cell(item) for item in header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for row in rows:
            stats = row.get("baseline") or {}
            metric = str(row.get("metric"))
            cell = f"{metric} (n={stats.get('n', 0)})"
            values = [
                cell,
                str(row.get("direction")),
                _fmt(stats.get("median")),
                _fmt(stats.get("p99")),
            ]
            for candidate in candidates:
                candidate_stats = (row.get("arms") or {}).get(candidate) or {}
                values.extend(
                    [
                        _fmt(candidate_stats.get("median")),
                        _fmt(candidate_stats.get("p99")),
                        _fmt_delta((row.get("deltas") or {}).get(candidate)),
                    ]
                )
            lines.append("| " + " | ".join(_cell(item) for item in values) + " |")

    lines.extend(
        [
            "",
            "## Arms",
            "",
            "| arm | role | ok | restart | boot id | health | benches | warnings |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for arm in arms:
        benches = arm.get("benches") or []
        bench_ok = sum(1 for bench in benches if bench.get("ok"))
        health = arm.get("health") or {}
        restart = arm.get("restart") or {}
        boot = arm.get("boot") or {}
        arm_warnings = arm.get("warnings") or []
        if arm.get("error"):
            arm_warnings = [*arm_warnings, str(arm.get("error"))]
        lines.append(
            "| "
            + " | ".join(
                _cell(item)
                for item in (
                    arm.get("name"),
                    arm.get("role"),
                    arm.get("ok"),
                    restart.get("exit_code", "skipped"),
                    boot.get("id") or "unavailable",
                    health.get("code", "-"),
                    f"{bench_ok}/{len(benches)}",
                    "; ".join(str(item) for item in arm_warnings) or "-",
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| arm | benchmark | ok | artifact |",
            "|---|---|---|---|",
        ]
    )
    for arm in arms:
        for bench in arm.get("benches") or []:
            lines.append(
                "| "
                + " | ".join(
                    _cell(item)
                    for item in (
                        arm.get("name"),
                        bench.get("name"),
                        bench.get("ok"),
                        bench.get("artifact"),
                    )
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# benchmark execution
# ---------------------------------------------------------------------------


def _run_process(
    argv: Sequence[str],
    *,
    workdir: Path,
    env: Mapping[str, str],
    timeout: float,
    log_path: Path,
    secrets: Sequence[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=workdir,
            env=dict(env),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        exit_code = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr += f"\n[bench-ab] benchmark exceeded {timeout:.0f}s and was killed\n"
    duration = time.perf_counter() - started
    log_text = (
        f"$ {shlex.join(list(argv))}\n"
        f"exit_code={exit_code} duration_s={duration:.3f}\n\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
    )
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(redact_text(log_text, secrets), encoding="utf-8")
    except OSError:
        pass
    return {
        "exit_code": exit_code,
        "duration_s": round(duration, 3),
        "log": str(log_path),
        "stdout_tail": redact_text(stdout[-2000:], secrets),
        "stderr_tail": redact_text(stderr[-4000:], secrets),
    }


def run_arm(
    index: int,
    arm: Arm,
    args: argparse.Namespace,
    benches: Sequence[str],
    *,
    workdir: Path,
    tests_dir: Path,
    out_dir: Path,
    api_key: str,
) -> dict[str, Any]:
    """Run one arm's restart/health/warmup/bench sequence.

    Fatal protocol failures are recorded in ``error``.  Benchmark failures are
    recorded per benchmark and continue with the remaining benchmarks so the
    operator gets a complete diagnosis.
    """
    merged = merge_env(os.environ, arm.env)
    if api_key:
        merged["VLLM_API_KEY"] = api_key
        merged["API_KEY"] = api_key
    secrets = _secret_values(merged, api_key)
    result: dict[str, Any] = {
        "index": index,
        "name": arm.name,
        "role": "baseline" if index == 0 else "candidate",
        "spec": arm.name,
        "env": redact_env(arm.env),
        "effective_env": filter_effective_env(merged),
        "started_at": utc_now(),
        "finished_at": None,
        "restart": {"attempted": False},
        "health": None,
        "boot": None,
        "boot_end": None,
        "cache_reset": None,
        "warmup": None,
        "benches": [],
        "warnings": [],
        "error": None,
        "ok": False,
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        result["error"] = f"cannot create artifact directory {out_dir}: {exc}"
        result["finished_at"] = utc_now()
        return result

    boot_changed = False
    try:
        if args.no_restart:
            result["restart"] = {"attempted": False, "reason": "--no-restart"}
            if arm.env:
                result["warnings"].append(
                    "environment overrides were not applied (--no-restart)"
                )
        else:
            start_script = workdir / "start.sh"
            if not start_script.is_file():
                raise ArmRunError(f"start script not found: {start_script}")
            log_path = out_dir / f"restart-{_slug(arm.name)}.log"
            started = time.perf_counter()
            restart_argv = ["./start.sh", "restart"]
            try:
                completed = subprocess.run(
                    restart_argv,
                    cwd=workdir,
                    env=merged,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=args.restart_timeout,
                    check=False,
                )
                exit_code = completed.returncode
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
            except subprocess.TimeoutExpired as exc:
                exit_code = 124
                stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr = exc.stderr if isinstance(exc.stderr, str) else ""
                stderr += (
                    f"\n[bench-ab] start.sh restart exceeded "
                    f"{args.restart_timeout:.0f}s and was killed\n"
                )
            duration = time.perf_counter() - started
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    redact_text(
                        f"$ ./start.sh restart\n"
                        f"exit_code={exit_code} duration_s={duration:.3f}\n\n"
                        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
                        secrets,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass
            result["restart"] = {
                "attempted": True,
                "command": restart_argv,
                "exit_code": exit_code,
                "duration_s": round(duration, 3),
                "log": str(log_path),
            }
            if exit_code != 0:
                raise ArmRunError(
                    f"./start.sh restart failed with exit {exit_code}; see {log_path}"
                )

        health = wait_for_health(
            args.base_url,
            timeout=args.health_timeout,
            interval=args.health_interval,
            api_key=api_key,
        )
        result["health"] = health
        if not health["ok"]:
            raise ArmRunError(
                f"/health did not return 200 within {args.health_timeout:.0f}s "
                f"(last HTTP {health.get('code')})"
            )

        result["boot"] = probe_boot_id(args.base_url, timeout=args.timeout, api_key=api_key)
        if not result["boot"].get("id"):
            result["warnings"].append("engine boot id unavailable")
        elif result["boot"].get("source") == "host-boot":
            result["warnings"].append("engine boot id fell back to host boot id")

        if merged.get("GLM53_EXPOSE_CACHE_RESET") == "1":
            reset = reset_prefix_cache(
                args.base_url, timeout=args.timeout, api_key=api_key
            )
            result["cache_reset"] = reset
            if reset["status"] == "absent":
                result["warnings"].append(str(reset.get("note")))
                print(
                    "[bench-ab]   reset endpoint absent (404); continuing",
                    flush=True,
                )
            elif reset["status"] != "ok":
                raise ArmRunError(
                    f"cache reset failed: HTTP {reset.get('http')} "
                    f"{reset.get('error', '')}".strip()
                )
        else:
            result["cache_reset"] = {
                "status": "skipped",
                "reason": "GLM53_EXPOSE_CACHE_RESET != 1",
            }

        warmup = warmup_request(
            args.base_url, model=args.model, timeout=args.timeout, api_key=api_key
        )
        result["warmup"] = warmup
        if not warmup.get("ok"):
            raise ArmRunError(f"warmup request failed: {warmup.get('error') or warmup.get('http')}")

        if args.settle > 0:
            time.sleep(args.settle)

        for bench_name in benches:
            spec = BENCHMARKS[bench_name]
            artifact_path = out_dir / f"{_slug(arm.name)}__{bench_name}.json"
            log_path = out_dir / f"{_slug(arm.name)}__{bench_name}.log"
            record: dict[str, Any] = {
                "name": bench_name,
                "started_at": utc_now(),
                "finished_at": None,
                "artifact": str(artifact_path),
                "log": str(log_path),
                "ok": False,
                "exit_code": None,
                "duration_s": None,
                "metrics": None,
                "error": None,
            }
            context = BenchContext(
                name=bench_name,
                base_url=args.base_url,
                model=args.model,
                runs=args.runs,
                timeout=args.timeout,
                artifact_path=artifact_path,
                tests_dir=tests_dir,
                workdir=workdir,
                sessions=args.sessions,
                ctx_tokens=args.ctx_tokens,
                cache_mode=args.cache_mode,
                api_key=api_key,
            )
            bench_started = time.perf_counter()
            try:
                argv = _benchmark_argv(spec, context)
                if argv is not None:
                    record["command"] = argv
                    child_timeout = max(60.0, args.timeout) * max(args.runs, 1) + 300.0
                    process = _run_process(
                        argv,
                        workdir=workdir,
                        env=merged,
                        timeout=child_timeout,
                        log_path=log_path,
                        secrets=secrets,
                    )
                    record.update(process)
                    if process["exit_code"] != 0:
                        raise BenchArtifactError(
                            f"{bench_name} exited {process['exit_code']} "
                            f"(see {log_path})"
                        )
                    artifact = load_artifact(artifact_path)
                else:
                    artifact = spec.run_direct(context)  # type: ignore[misc]
                    write_artifact(artifact_path, artifact)
                    record["exit_code"] = 0
                metrics = spec.extract(artifact)
                record["metrics"] = metrics.to_dict()
                record["ok"] = True
            except (BenchAbError, OSError, ValueError) as exc:
                record["error"] = str(exc)
            except Exception as exc:  # defensive: never lose the arm report
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["duration_s"] = round(time.perf_counter() - bench_started, 3)
            record["finished_at"] = utc_now()
            result["benches"].append(record)
            status = "OK" if record["ok"] else "FAIL"
            print(
                f"[bench-ab]   {status} {bench_name} "
                f"({record['duration_s']:.1f}s)"
                + ("" if record["ok"] else f": {record['error']}"),
                flush=True,
            )

        boot_end = probe_boot_id(
            args.base_url,
            timeout=min(max(args.timeout, 1.0), 30.0),
            api_key=api_key,
        )
        result["boot_end"] = boot_end
        start_boot = (result.get("boot") or {}).get("id")
        end_boot = boot_end.get("id")
        start_source = (result.get("boot") or {}).get("source")
        end_source = boot_end.get("source")
        if start_boot and end_boot and start_boot != end_boot:
            if start_source == end_source:
                boot_changed = True
                result["warnings"].append(
                    "engine boot changed during the arm; measurements mix boot states"
                )
            else:
                result["warnings"].append(
                    "boot identity source changed during the arm; cannot prove a stable boot"
                )
    except ArmRunError as exc:
        result["error"] = str(exc)
    except Exception as exc:  # defensive: keep partial arm data
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["finished_at"] = utc_now()
    result["ok"] = (
        result["error"] is None
        and not boot_changed
        and bool(result["benches"])
        and all(bench.get("ok") for bench in result["benches"])
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_workdir(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if (cwd_candidate / "start.sh").is_file():
        return cwd_candidate
    repo_candidate = REPO_ROOT / path
    if (repo_candidate / "start.sh").is_file():
        return repo_candidate
    return cwd_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "A/B benchmark harness for the 2-node vLLM cluster: restart per arm, "
            "health gate, warmup, pluggable benchmarks, medians/p99 comparison."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/bench-ab.py --arm baseline \\\n"
            "      --arm candidate:MAX_NUM_SEQS=6,GLM53_MIXED_PREFILL_CHUNK=448 \\\n"
            "      --runs 3 --out results/ab-<stamp>.json\n"
            "  python3 scripts/bench-ab.py --dry-run --bench short_chat --no-restart\n"
            "\nRegistered benchmarks: " + ", ".join(BENCHMARKS)
        ),
    )
    parser.add_argument("--arm", action="append", default=[],
                        metavar="NAME[:K=V,...]",
                        help="repeatable; the first arm is the baseline "
                             "(override values must not contain commas)")
    parser.add_argument("--bench", action="append", choices=list(BENCHMARKS), default=None,
                        help="repeatable; default runs every registered benchmark")
    parser.add_argument("--list-benches", action="store_true",
                        help="list registered benchmarks and exit")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR,
                        help="vllm-exl3 directory containing start.sh and tests/")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help="runs per benchmark (also cache_sessions rounds)")
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS,
                        help="cache_sessions co-batch width")
    parser.add_argument("--ctx-tokens", type=int, default=DEFAULT_CTX_TOKENS,
                        help="cache_sessions approximate prompt size")
    parser.add_argument("--cache-mode", choices=("sessions", "shared-prefix", "sub-page"),
                        default="sessions")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="per-request/benchmark HTTP timeout (seconds)")
    parser.add_argument("--health-timeout", type=float, default=DEFAULT_HEALTH_TIMEOUT)
    parser.add_argument("--health-interval", type=float, default=DEFAULT_HEALTH_INTERVAL)
    parser.add_argument("--restart-timeout", type=float, default=DEFAULT_RESTART_TIMEOUT)
    parser.add_argument("--settle", type=float, default=0.0,
                        help="extra sleep after health before the warmup")
    parser.add_argument("--no-restart", action="store_true",
                        help="use the already-running server; arm env overrides are NOT applied")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan, commands and artifact paths; no I/O")
    parser.add_argument("--out", default=None,
                        help="result JSON (default: <workdir>/results/ab-<stamp>.json)")
    parser.add_argument("--markdown", default=None,
                        help="Markdown table path (default: sibling of --out)")
    parser.add_argument("--api-key", default=None,
                        help="bearer token; defaults to $VLLM_API_KEY or $API_KEY")
    return parser


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _validate_inputs(args: argparse.Namespace, benches: Sequence[str], tests_dir: Path) -> None:
    if args.runs < 1:
        raise BenchAbError("--runs must be >= 1")
    if args.sessions < 1:
        raise BenchAbError("--sessions must be >= 1")
    if args.ctx_tokens < 128:
        raise BenchAbError("--ctx-tokens must be >= 128")
    if args.timeout <= 0 or args.health_timeout <= 0 or args.restart_timeout <= 0:
        raise BenchAbError("timeouts must be > 0")
    if args.health_interval <= 0:
        raise BenchAbError("--health-interval must be > 0")
    if args.settle < 0:
        raise BenchAbError("--settle must be >= 0")
    if any(name.startswith("decode_") for name in benches) and (
        api_root(args.base_url) != api_root(DEFAULT_BASE_URL)
    ):
        raise BenchAbError(
            "tests/bench_decode.py hardcodes http://127.0.0.1:8888; "
            "select only prefix_cache/cache_sessions/short_chat for another --base-url"
        )
    for bench in benches:
        for filename in BENCHMARKS[bench].requires:
            if not (tests_dir / filename).is_file():
                raise BenchAbError(f"missing benchmark client {tests_dir / filename}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_benches:
        for name, spec in BENCHMARKS.items():
            print(f"{name}: {spec.description}")
        return 0

    try:
        arms = parse_arms(args.arm)
        benches = parse_bench_selection(args.bench or [])
        workdir = resolve_workdir(args.workdir)
        tests_dir = workdir / "tests"
        _validate_inputs(args, benches, tests_dir)
    except BenchAbError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    api_key = args.api_key
    if api_key is None:
        api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("API_KEY") or ""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.out:
        out = Path(args.out).expanduser()
    else:
        out = workdir / "results" / f"ab-{stamp}.json"
    if args.markdown:
        markdown_path = Path(args.markdown).expanduser()
    else:
        markdown_path = out.with_suffix(".md") if out.suffix else out.with_name(out.name + ".md")
    out_dir = artifact_dir_for(out)

    if args.dry_run:
        try:
            print_dry_run(
                arms,
                benches,
                args,
                workdir=workdir,
                tests_dir=tests_dir,
                out=out,
                markdown=markdown_path,
                api_key=api_key,
            )
        except BenchAbError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0

    if not args.no_restart and not (workdir / "start.sh").is_file():
        print(f"error: start script not found: {workdir / 'start.sh'}", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "schema": "glm53-exl3-bench-ab/1",
        "stamp": stamp,
        "created_at": utc_now(),
        "finished_at": None,
        "base_url": args.base_url,
        "model": args.model,
        "workdir": str(workdir),
        "git": git_info(),
        "host": host_info(),
        "protocol": {
            "arms": [{"name": arm.name, "env": redact_env(arm.env)} for arm in arms],
            "benchmarks": list(benches),
            "runs": args.runs,
            "restart": not args.no_restart,
            "health_timeout_s": args.health_timeout,
            "warmup": True,
            "cache_reset_flag": "GLM53_EXPOSE_CACHE_RESET",
            "out_dir": str(out_dir),
        },
        "arms": [],
        "warnings": [],
        "comparison": {},
    }

    print(
        f"[bench-ab] {len(arms)} arm(s), {len(benches)} benchmark(s), "
        f"workdir={workdir}",
        flush=True,
    )
    for index, arm in enumerate(arms):
        print(
            f"[bench-ab] arm {index + 1}/{len(arms)} {arm.name} "
            f"({'baseline' if index == 0 else 'candidate'})",
            flush=True,
        )
        result = run_arm(
            index,
            arm,
            args,
            benches,
            workdir=workdir,
            tests_dir=tests_dir,
            out_dir=out_dir,
            api_key=api_key,
        )
        report["arms"].append(result)
        print(
            f"[bench-ab] arm {arm.name}: {'OK' if result['ok'] else 'FAILED'}"
            + (f" ({result['error']})" if result.get("error") else ""),
            flush=True,
        )

    report["finished_at"] = utc_now()
    report["comparison"] = build_comparison(report["arms"])
    warnings = boot_warnings(report["arms"], restart_required=not args.no_restart)
    git = report["git"]
    if git.get("dirty"):
        warnings.append(
            f"git tree is dirty ({git.get('dirty_count')} entries); "
            "record the exact working-tree state before publishing results"
        )
    if api_key and any(name.startswith("decode_") for name in benches):
        warnings.append(
            "bench_decode.py does not send an Authorization header; decode arms may "
            "fail against a server that requires VLLM_API_KEY"
        )
    if args.model != DEFAULT_MODEL and any(
        name.startswith("decode_") for name in benches
    ):
        warnings.append(
            f"bench_decode.py hardcodes model {DEFAULT_MODEL!r}; "
            f"--model {args.model!r} does not affect decode benches"
        )
    for arm in report["arms"]:
        if not arm.get("ok"):
            failed = [
                bench.get("name")
                for bench in arm.get("benches", [])
                if not bench.get("ok")
            ]
            if failed:
                warnings.append(
                    f"arm {arm.get('name')} failed benchmarks: "
                    + ", ".join(str(item) for item in failed)
                )
            else:
                warnings.append(f"arm {arm.get('name')} failed")
        for warning in arm.get("warnings", []):
            warnings.append(f"arm {arm.get('name')}: {warning}")
    if not report["comparison"].get("baseline_ok"):
        warnings.append("baseline arm failed; candidate deltas were suppressed")
    report["warnings"] = _dedupe(warnings)

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write results: {exc}", file=sys.stderr)
        return 1

    print("\n" + render_markdown(report))
    print(f"\n[bench-ab] wrote {out}")
    print(f"[bench-ab] wrote {markdown_path}")
    all_ok = bool(report["arms"]) and all(arm.get("ok") for arm in report["arms"])
    return 0 if all_ok and report["comparison"].get("baseline_ok") else 1


if __name__ == "__main__":
    sys.exit(main())
