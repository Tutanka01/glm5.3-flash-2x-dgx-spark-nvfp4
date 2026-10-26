#!/usr/bin/env python3
"""Prefill-throughput degradation watchdog for the GLM-5.3-Flash EXL3 lane.

New file in this lane.  The idea (a small out-of-band observer that never
restarts the server itself) is inspired by the Apache-2.0 Reederey87 kit and
TensorFold's prefill watch notes; this implementation is independent and uses
only the Python standard library.

What it watches
---------------
* ``GET /metrics`` -- ``vllm:prompt_tokens_total`` deltas combined with the
  ``vllm:time_to_first_token_seconds{_sum,_count}`` histogram.  The reported
  prefill proxy is ``delta(prompt_tokens) / delta(ttft_seconds_sum)``; when the
  histogram is absent it falls back to tokens per wall-clock second.
* ``logs/head.log`` -- vLLM's periodic ``Avg prompt throughput: N tokens/s``
  lines and explicit ``prefill ... tok/s`` lines.

A *sustained* drop is detected against the median of the older observations
(a robust baseline), not against a single noisy sample.  The default window is
3 recent observations and the default threshold is a 30% drop.

Exit codes
----------
0   healthy / no evidence of degradation
1   unreadable or insufficient input (watchdog cannot make a claim)
2   sustained degradation

The watchdog never restarts anything.  Fixtures are first-class:
``--metrics-file`` may be repeated (one Prometheus snapshot per file, oldest
first) and ``--log-file`` replaces the default log path.

Usage
-----
    python3 scripts/prefill-watchdog.py
    python3 scripts/prefill-watchdog.py --json --window 5 --threshold 0.25
    python3 scripts/prefill-watchdog.py --metrics-file s0.txt --metrics-file s1.txt ...
    python3 scripts/prefill-watchdog.py --log-file logs/head.log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_BASE_URL = "http://127.0.0.1:8888"
SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_DIR = SCRIPT_DIR.parent
DEFAULT_LOG_FILE = VLLM_DIR / "logs" / "head.log"
DEFAULT_WINDOW = 3
DEFAULT_THRESHOLD = 0.30
DEFAULT_BASELINE_WINDOW = 20
DEFAULT_TAIL_LINES = 5000
DEFAULT_INTERVAL = 5.0
HTTP_TIMEOUT = 10.0

_METRIC_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|NaN|[+-]?Inf)"
    r"(?:\s+[+-]?\d+(?:\.\d+)?)?\s*$"
)
_BOOT_LABEL_RE = re.compile(r'\bboot_id="([^"]+)"')
_START_METRIC_RE = re.compile(
    r"(?:process|engine|server).*start_time_seconds$|"
    r"start_time_seconds.*(?:process|engine|server)$"
)

# vLLM's scheduler logs "Avg prompt throughput: 123.4 tokens/s"; benchmark and
# overlay logs may print "prefill=934.2 tok/s" or "prefill_tok_s: 934.2".
_LOG_PATTERNS = (
    re.compile(r"Avg prompt throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens?/s", re.I),
    re.compile(r"\bprefill(?:_tok_s|_tokens_per_second)?\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"\bprefill\b[^\n]{0,80}?([0-9]+(?:\.[0-9]+)?)\s*(?:tok|tokens?)/s", re.I),
    re.compile(r'"prefill_tok_s"\s*:\s*([0-9]+(?:\.[0-9]+)?)', re.I),
)


class WatchdogError(RuntimeError):
    """Unreadable or unusable watchdog input."""


@dataclass(frozen=True)
class MetricsSnapshot:
    source: str
    ts: float
    prompt_tokens: float | None = None
    ttft_sum: float | None = None
    ttft_count: float | None = None
    process_start: float | None = None
    boot_label: str | None = None


@dataclass(frozen=True)
class Observation:
    tok_s: float | None
    ttft_s: float | None
    prompt_tokens: float | None
    dt_s: float | None


@dataclass(frozen=True)
class SeriesAnalysis:
    status: str  # "ok" | "degraded" | "insufficient"
    n: int
    baseline: float | None
    recent: float | None
    drop_fraction: float | None
    cutoff: float | None
    recent_below_cutoff: int
    window: int
    threshold: float

    @property
    def degraded(self) -> bool:
        return self.status == "degraded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "degraded": self.degraded,
            "n": self.n,
            "baseline": _round_or_none(self.baseline),
            "recent": _round_or_none(self.recent),
            "drop_fraction": _round_or_none(self.drop_fraction),
            "cutoff": _round_or_none(self.cutoff),
            "recent_below_cutoff": self.recent_below_cutoff,
            "window": self.window,
            "threshold": self.threshold,
        }


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _parse_value(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_metrics_text(text: str, *, source: str, ts: float) -> MetricsSnapshot:
    """Parse one Prometheus scrape.

    Label sets are aggregated (summed), which is what the vLLM metrics expose
    for a single engine behind one API server.  Missing metrics stay ``None``
    so callers can distinguish "zero" from "not exported".
    """
    prompt_tokens: float | None = None
    ttft_sum: float | None = None
    ttft_count: float | None = None
    process_start: float | None = None
    boot_label: str | None = None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        label_match = _BOOT_LABEL_RE.search(line)
        if label_match and boot_label is None:
            boot_label = label_match.group(1)
        match = _METRIC_RE.match(line)
        if not match:
            continue
        name, _labels, raw_value = match.groups()
        value = _parse_value(raw_value)
        if value is None:
            continue
        if name == "vllm:prompt_tokens_total":
            prompt_tokens = (prompt_tokens or 0.0) + value
        elif name == "vllm:time_to_first_token_seconds_sum":
            ttft_sum = (ttft_sum or 0.0) + value
        elif name == "vllm:time_to_first_token_seconds_count":
            ttft_count = (ttft_count or 0.0) + value
        elif _START_METRIC_RE.search(name) and process_start is None:
            process_start = value

    return MetricsSnapshot(
        source=source,
        ts=ts,
        prompt_tokens=prompt_tokens,
        ttft_sum=ttft_sum,
        ttft_count=ttft_count,
        process_start=process_start,
        boot_label=boot_label,
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise WatchdogError(f"cannot read {path}: {exc}") from exc


def snapshot_from_file(path: Path) -> MetricsSnapshot:
    try:
        ts = path.stat().st_mtime
    except OSError as exc:
        raise WatchdogError(f"cannot stat {path}: {exc}") from exc
    return parse_metrics_text(_read_text(path), source=str(path), ts=ts)


def fetch_metrics_snapshot(
    base_url: str,
    *,
    api_key: str = "",
    timeout: float = HTTP_TIMEOUT,
) -> MetricsSnapshot:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    url = root + "/metrics"
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WatchdogError(f"cannot scrape {url}: {exc}") from exc
    return parse_metrics_text(body, source=url, ts=time.time())


def observations_from_snapshots(snapshots: Sequence[MetricsSnapshot]) -> list[Observation]:
    """Turn cumulative counters into per-interval observations.

    Counter resets (a restart between snapshots) break the interval instead of
    producing a negative throughput.  Idle intervals (no prompt tokens) are
    skipped: they carry no prefill signal.
    """
    observations: list[Observation] = []
    for previous, current in zip(snapshots, snapshots[1:], strict=False):
        if previous.prompt_tokens is None or current.prompt_tokens is None:
            continue
        if (
            previous.process_start is not None
            and current.process_start is not None
            and previous.process_start != current.process_start
        ):
            # A different engine process means the old baseline is invalid.
            observations.clear()
            continue
        d_prompt = current.prompt_tokens - previous.prompt_tokens
        d_ttft_sum: float | None = None
        if previous.ttft_sum is not None and current.ttft_sum is not None:
            d_ttft_sum = current.ttft_sum - previous.ttft_sum
        d_ttft_count = (
            current.ttft_count - previous.ttft_count
            if previous.ttft_count is not None and current.ttft_count is not None
            else None
        )
        if d_prompt < 0 or (d_ttft_sum is not None and d_ttft_sum < 0):
            observations.clear()  # counter reset / restart
            continue
        if d_prompt <= 0:
            continue
        dt = current.ts - previous.ts if current.ts > previous.ts else None
        tok_s: float | None = None
        if d_ttft_sum is not None and d_ttft_sum > 0:
            tok_s = d_prompt / d_ttft_sum
        elif dt is not None and dt > 0:
            tok_s = d_prompt / dt
        ttft = (
            d_ttft_sum / d_ttft_count
            if d_ttft_sum is not None
            and d_ttft_count is not None
            and d_ttft_count > 0
            else None
        )
        observations.append(
            Observation(
                tok_s=tok_s,
                ttft_s=ttft,
                prompt_tokens=d_prompt,
                dt_s=dt,
            )
        )
    return observations


def analyze_series(
    values: Sequence[float | None],
    *,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
) -> SeriesAnalysis:
    """Compare the recent window against a robust baseline (median).

    ``values`` are throughput-like numbers, oldest first.  A sustained drop
    means the median of the last ``window`` values is below
    ``baseline * (1 - threshold)``.  At least one older observation and a full
    recent window are required; otherwise the result is ``insufficient``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if baseline_window < 1:
        raise ValueError("baseline_window must be >= 1")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")

    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value)) and float(value) > 0
    ]
    if len(clean) < window + 1:
        return SeriesAnalysis(
            status="insufficient",
            n=len(clean),
            baseline=None,
            recent=None,
            drop_fraction=None,
            cutoff=None,
            recent_below_cutoff=0,
            window=window,
            threshold=threshold,
        )

    recent_values = clean[-window:]
    pool_start = max(0, len(clean) - window - baseline_window)
    baseline_values = clean[pool_start:-window]
    if not baseline_values:
        return SeriesAnalysis(
            status="insufficient",
            n=len(clean),
            baseline=None,
            recent=None,
            drop_fraction=None,
            cutoff=None,
            recent_below_cutoff=0,
            window=window,
            threshold=threshold,
        )

    baseline = statistics.median(baseline_values)
    recent = statistics.median(recent_values)
    cutoff = baseline * (1.0 - threshold)
    drop = 1.0 - recent / baseline if baseline > 0 else None
    below = sum(1 for value in recent_values if value <= cutoff)
    degraded = recent <= cutoff
    return SeriesAnalysis(
        status="degraded" if degraded else "ok",
        n=len(clean),
        baseline=baseline,
        recent=recent,
        drop_fraction=drop,
        cutoff=cutoff,
        recent_below_cutoff=below,
        window=window,
        threshold=threshold,
    )


def parse_log_values(lines: Iterable[str]) -> tuple[list[float], int]:
    """Extract prefill tok/s values from log lines.

    Returns ``(values, zero_lines_skipped)``.  Zero-throughput lines are vLLM's
    idle heartbeat and would otherwise look like a collapse; they are counted
    but not used as observations.
    """
    values: list[float] = []
    zero_lines = 0
    for line in lines:
        for pattern in _LOG_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            value = _parse_value(match.group(1))
            if value is None:
                continue
            if value <= 0:
                zero_lines += 1
            else:
                values.append(value)
            break
    return values, zero_lines


def read_tail(path: Path, max_lines: int) -> list[str]:
    """Read the last ``max_lines`` lines without loading a huge log whole."""
    if max_lines < 1:
        raise WatchdogError("--tail-lines must be >= 1")
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            block = 64 * 1024
            data = b""
            while end > 0 and data.count(b"\n") <= max_lines:
                step = min(block, end)
                end -= step
                handle.seek(end)
                data = handle.read(step) + data
    except OSError as exc:
        raise WatchdogError(f"cannot read {path}: {exc}") from exc
    text = data.decode("utf-8", "replace")
    return text.splitlines()[-max_lines:]


def analyze_log_file(
    path: Path,
    *,
    window: int,
    threshold: float,
    baseline_window: int,
    tail_lines: int,
) -> dict[str, Any]:
    lines = read_tail(path, tail_lines)
    values, zero_lines = parse_log_values(lines)
    analysis = analyze_series(
        values,
        window=window,
        threshold=threshold,
        baseline_window=baseline_window,
    )
    return {
        "source": "log",
        "path": str(path),
        "lines_read": len(lines),
        "values": [round(value, 4) for value in values],
        "zero_lines_skipped": zero_lines,
        "analysis": analysis.to_dict(),
        "status": analysis.status,
        "degraded": analysis.degraded,
    }


def analyze_metrics_snapshots(
    snapshots: Sequence[MetricsSnapshot],
    *,
    window: int,
    threshold: float,
    baseline_window: int,
) -> dict[str, Any]:
    observations = observations_from_snapshots(snapshots)
    values = [
        observation.tok_s
        for observation in observations
        if observation.tok_s is not None
    ]
    analysis = analyze_series(
        values,
        window=window,
        threshold=threshold,
        baseline_window=baseline_window,
    )
    return {
        "source": "metrics",
        "scrapes": len(snapshots),
        "observations": len(observations),
        "values": [round(value, 4) for value in values if value is not None],
        "ttft_values": [
            round(observation.ttft_s, 6)
            for observation in observations
            if observation.ttft_s is not None
        ],
        "analysis": analysis.to_dict(),
        "status": analysis.status,
        "degraded": analysis.degraded,
        "method": (
            "prompt_tokens/ttft_sum"
            if any(observation.ttft_s is not None for observation in observations)
            else "prompt_tokens/wall_time"
            if values
            else "none"
        ),
    }


def _safe_analysis(record: dict[str, Any]) -> SeriesAnalysis | None:
    analysis = record.get("analysis")
    if not isinstance(analysis, dict):
        return None
    try:
        return SeriesAnalysis(
            status=str(analysis["status"]),
            n=int(analysis["n"]),
            baseline=None if analysis.get("baseline") is None else float(analysis["baseline"]),
            recent=None if analysis.get("recent") is None else float(analysis["recent"]),
            drop_fraction=None
            if analysis.get("drop_fraction") is None
            else float(analysis["drop_fraction"]),
            cutoff=None if analysis.get("cutoff") is None else float(analysis["cutoff"]),
            recent_below_cutoff=int(analysis.get("recent_below_cutoff", 0)),
            window=int(analysis.get("window", DEFAULT_WINDOW)),
            threshold=float(analysis.get("threshold", DEFAULT_THRESHOLD)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def collect_metrics_files(paths: Sequence[Path]) -> list[MetricsSnapshot]:
    if not paths:
        raise WatchdogError("--metrics-file needs at least one snapshot")
    return [snapshot_from_file(path) for path in paths]


def collect_http_snapshots(
    base_url: str,
    *,
    samples: int,
    interval: float,
    timeout: float,
    api_key: str,
) -> list[MetricsSnapshot]:
    snapshots = []
    for index in range(samples):
        if index:
            time.sleep(interval)
        snapshots.append(fetch_metrics_snapshot(base_url, api_key=api_key, timeout=timeout))
    return snapshots


def _result(
    *,
    status: str,
    exit_code: int,
    source: str,
    analyses: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    errors: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    primary_record = next(
        (record for record in analyses if record.get("status") == "degraded"),
        None,
    )
    if primary_record is None:
        primary_record = next(
            (record for record in analyses if record.get("status") == "ok"),
            analyses[0] if analyses else None,
        )
    primary: SeriesAnalysis | None = _safe_analysis(primary_record) if primary_record else None
    payload: dict[str, Any] = {
        "status": status,
        "degraded": status == "DEGRADED",
        "exit_code": exit_code,
        "source": source,
        "window": args.window,
        "threshold": args.threshold,
        "baseline_window": args.baseline_window,
        "baseline_tok_s": None if primary is None else _round_or_none(primary.baseline),
        "recent_tok_s": None if primary is None else _round_or_none(primary.recent),
        "drop_fraction": None if primary is None else _round_or_none(primary.drop_fraction),
        "cutoff_tok_s": None if primary is None else _round_or_none(primary.cutoff),
        "sources": list(analyses),
        "errors": list(errors),
        "notes": list(notes),
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report sustained prefill-throughput degradation. Never restarts anything.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="vLLM root URL for /metrics")
    parser.add_argument("--metrics-file", action="append", type=Path, default=[],
                        help="Prometheus snapshot fixture; repeat oldest-first "
                             "(two or more snapshots are needed for deltas)")
    parser.add_argument("--log-file", type=Path, default=None,
                        help=f"server log to inspect (default: {DEFAULT_LOG_FILE} when used)")
    parser.add_argument("--tail-lines", type=int, default=DEFAULT_TAIL_LINES)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="recent observations that must all/mostly be low")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="fractional drop from baseline that counts as degraded")
    parser.add_argument("--baseline-window", type=int, default=DEFAULT_BASELINE_WINDOW,
                        help="maximum older observations used for the median baseline")
    parser.add_argument("--samples", type=int, default=0,
                        help="HTTP scrapes (0 = window + 4)")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help="seconds between HTTP scrapes")
    parser.add_argument("--api-key", default=None,
                        help="bearer token; defaults to $VLLM_API_KEY or $API_KEY")
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    return parser


def _print_result(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    status = payload["status"]
    if status == "ERROR":
        detail = "; ".join(payload["errors"]) or "no usable input"
        print(f"ERROR {detail}")
        return
    print(
        f"{status} source={payload['source']} "
        f"baseline_tok_s={payload['baseline_tok_s']} "
        f"recent_tok_s={payload['recent_tok_s']} "
        f"drop={payload['drop_fraction']} threshold={payload['threshold']} "
        f"window={payload['window']}"
    )
    for note in payload["notes"]:
        print(f"note: {note}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.window < 1:
        print("error: --window must be >= 1", file=sys.stderr)
        return 1
    if args.baseline_window < 1:
        print("error: --baseline-window must be >= 1", file=sys.stderr)
        return 1
    if not 0 < args.threshold < 1:
        print("error: --threshold must be between 0 and 1", file=sys.stderr)
        return 1
    if args.samples < 0:
        print("error: --samples must be >= 0", file=sys.stderr)
        return 1
    if args.interval <= 0:
        print("error: --interval must be > 0", file=sys.stderr)
        return 1

    api_key = args.api_key
    if api_key is None:
        api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("API_KEY") or ""

    analyses: list[dict[str, Any]] = []
    errors: list[str] = []
    notes: list[str] = []
    explicit_metrics = bool(args.metrics_file)
    explicit_log = args.log_file is not None
    metrics_insufficient = False

    if explicit_metrics:
        try:
            snapshots = collect_metrics_files(args.metrics_file)
            record = analyze_metrics_snapshots(
                snapshots,
                window=args.window,
                threshold=args.threshold,
                baseline_window=args.baseline_window,
            )
            analyses.append(record)
            metrics_insufficient = record["status"] == "insufficient"
        except WatchdogError as exc:
            errors.append(str(exc))

    # Scrape only when the caller did not hand us fixtures, or when fixtures
    # produced no usable series.
    if not explicit_metrics and not explicit_log:
        samples = args.samples or (args.window + 4)
        samples = max(samples, 2)
        try:
            snapshots = collect_http_snapshots(
                args.base_url,
                samples=samples,
                interval=args.interval,
                timeout=HTTP_TIMEOUT,
                api_key=api_key,
            )
            record = analyze_metrics_snapshots(
                snapshots,
                window=args.window,
                threshold=args.threshold,
                baseline_window=args.baseline_window,
            )
            analyses.append(record)
            metrics_insufficient = record["status"] == "insufficient"
        except WatchdogError as exc:
            notes.append(f"metrics unavailable: {exc}")

    log_path = args.log_file
    if log_path is None and not explicit_metrics:
        log_path = DEFAULT_LOG_FILE
    use_log = explicit_log or (not explicit_metrics and (not analyses or metrics_insufficient))
    if use_log and log_path is not None:
        if log_path.is_file():
            try:
                record = analyze_log_file(
                    log_path,
                    window=args.window,
                    threshold=args.threshold,
                    baseline_window=args.baseline_window,
                    tail_lines=args.tail_lines,
                )
                analyses.append(record)
            except WatchdogError as exc:
                errors.append(str(exc))
        elif explicit_log:
            errors.append(f"log file not found: {log_path}")
        else:
            notes.append(f"log file not found: {log_path}")

    if errors:
        payload = _result(
            status="ERROR",
            exit_code=1,
            source="none" if not analyses else "+".join(
                record["source"] for record in analyses
            ),
            analyses=analyses,
            args=args,
            errors=errors,
            notes=notes,
        )
        _print_result(payload, args.json)
        return 1

    usable = [record for record in analyses if record.get("status") in ("ok", "degraded")]
    if not usable:
        if not errors:
            errors.append("no usable prefill samples (metrics and log were empty or idle)")
        payload = _result(
            status="ERROR",
            exit_code=1,
            source="none" if not analyses else "+".join(
                record["source"] for record in analyses
            ),
            analyses=analyses,
            args=args,
            errors=errors,
            notes=notes,
        )
        _print_result(payload, args.json)
        return 1

    degraded = any(record.get("status") == "degraded" for record in usable)
    status = "DEGRADED" if degraded else "OK"
    payload = _result(
        status=status,
        exit_code=2 if degraded else 0,
        source="+".join(record["source"] for record in usable),
        analyses=analyses,
        args=args,
        notes=notes,
    )
    _print_result(payload, args.json)
    return payload["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
