#!/usr/bin/env python3
"""GPU/system power and energy sampler for the glm53 benches (stdlib only).

Two modes:
  ./bench-power.py [options] -- COMMAND [args...]   sample around COMMAND
  ./bench-power.py --watch [options]                sample until Ctrl+C

Writes results/glm53-power-<label>-<stamp>.json (summary), .jsonl (raw
samples/markers/events) and .svg (chart). GPU power comes from NVML
(pynvml preferred, nvidia-smi fallback); Linux RAPL package energy is
added when the kernel exposes it. This must run on the GB10 nodes.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import xml.sax.saxutils as saxutils
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1
SMI_TIMEOUT = 5.0
NA_VALUES = frozenset(("", "n/a", "[n/a]", "[not supported]", "not supported", "n/a ]"))
MIN_INTERVAL = 0.05
MAX_INTERVAL = 30.0
MAX_GAP = 5.0  # trapezoid segments wider than this (s) are excluded from energy
KILL_GRACE = 5.0  # seconds between SIGTERM and SIGKILL for the wrapped command
LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")


class CollectorError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def now_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "Z"


def sanitize_label(label: str) -> str:
    cleaned = LABEL_RE.sub("_", label).strip("._-")[:64]
    return cleaned or "run"


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; returns None when there is no data."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _num(text: str) -> float | None:
    """Parse one nvidia-smi CSV field; '[N/A]' and friends become None."""
    value = text.strip().lower()
    if value in NA_VALUES:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def series_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"samples": 0}
    return {
        "samples": len(values),
        "mean_w": round(statistics.fmean(values), 3),
        "median_w": round(statistics.median(values), 3),
        "min_w": round(min(values), 3),
        "max_w": round(max(values), 3),
        "p95_w": round(percentile(values, 0.95), 3),
    }


def trapezoid_energy(
    points: list[tuple[float, float]], max_gap: float = MAX_GAP
) -> tuple[float, float, int]:
    """Integrate watts over seconds; segments wider than max_gap are excluded.

    Returns (energy_j, excluded_gap_seconds, excluded_segment_count).
    """
    energy = 0.0
    excluded_seconds = 0.0
    excluded_segments = 0
    for (t0, w0), (t1, w1) in zip(points, points[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        if dt > max_gap:
            excluded_seconds += dt
            excluded_segments += 1
            continue
        energy += (w0 + w1) / 2.0 * dt
    return energy, excluded_seconds, excluded_segments


def counter_energy(points: list[tuple[float, float]]) -> float | None:
    """Energy (J) from a monotonic millijoule counter sampled at (t, mj)."""
    usable = [mj for _, mj in points if mj is not None]
    if len(usable) < 2:
        return None
    delta = usable[-1] - usable[0]
    if delta < 0:  # counter reset or wrap without known range: refuse
        return None
    return delta / 1000.0


# ---------------------------------------------------------------------------
# collectors — each yields flat device records:
#   {"key": "gpu0" | "sys-package-0", "group": "gpu" | "system",
#    "w": watts | None, "mj": counter | None, ...extras}
# ---------------------------------------------------------------------------


class BaseCollector:
    group = "gpu"
    backend = "abstract"

    def probe(self) -> dict[str, Any]:
        return {}

    def sample(self) -> list[dict[str, Any]] | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NvmlCollector(BaseCollector):
    """Preferred backend: pynvml, including the exact mJ energy counter."""

    backend = "pynvml"

    def __init__(self, wanted: set[int] | None) -> None:
        import pynvml  # noqa: PLC0415 — optional dependency, resolved at runtime

        self._nvml = pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count < 1:
            raise CollectorError("pynvml reports zero NVIDIA GPUs")
        self.indices = sorted(wanted) if wanted else list(range(count))
        unknown = [i for i in self.indices if i >= count]
        if unknown:
            raise CollectorError(
                f"GPU index(es) {unknown} do not exist (device count: {count})"
            )
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex_v2(i) for i in self.indices]
        self._energy_fn = getattr(pynvml, "nvmlDeviceGetTotalEnergyConsumption", None)
        self._energy_supported = {i: self._energy_fn is not None for i in self.indices}
        self._not_supported = getattr(pynvml, "NVMLErrorNotSupported", pynvml.NVMLError)

    def probe(self) -> dict[str, Any]:
        gpus = []
        for index, handle in zip(self.indices, self.handles):
            entry: dict[str, Any] = {
                "key": f"gpu{index}",
                "index": index,
                "name": _text(self._nvml.nvmlDeviceGetName(handle)),
                "uuid": _text(self._nvml.nvmlDeviceGetUUID(handle)),
            }
            for field, getter in (
                ("power_limit_w", self._nvml.nvmlDeviceGetPowerManagementLimit),
                (
                    "power_default_limit_w",
                    getattr(
                        self._nvml, "nvmlDeviceGetPowerManagementDefaultLimit", None
                    ),
                ),
            ):
                try:
                    if getter is not None:
                        entry[field] = getter(handle) / 1000.0
                except self._nvml.NVMLError:
                    pass
            if self._energy_fn is not None:
                try:
                    self._energy_fn(handle)
                    entry["energy_counter"] = "supported"
                except self._nvml.NVMLError:
                    entry["energy_counter"] = "unsupported"
                    self._energy_supported[index] = False
            gpus.append(entry)
        return {"driver": _text(self._nvml.nvmlSystemGetDriverVersion()), "gpus": gpus}

    def sample(self) -> list[dict[str, Any]] | None:
        devices: list[dict[str, Any]] = []
        failed = 0
        for index, handle in zip(self.indices, self.handles):
            device: dict[str, Any] = {"key": f"gpu{index}", "group": "gpu"}
            getters = (
                ("w", lambda h: self._nvml.nvmlDeviceGetPowerUsage(h) / 1000.0),
                ("util", lambda h: self._nvml.nvmlDeviceGetUtilizationRates(h).gpu),
                (
                    "temp",
                    lambda h: self._nvml.nvmlDeviceGetTemperature(
                        h, self._nvml.NVML_TEMPERATURE_GPU
                    ),
                ),
                ("mem_mib", lambda h: self._nvml.nvmlDeviceGetMemoryInfo(h).used / 2**20),
                ("sm_mhz", lambda h: self._nvml.nvmlDeviceGetClockInfo(h, self._nvml.NVML_CLOCK_SM)),
                ("mem_mhz", lambda h: self._nvml.nvmlDeviceGetClockInfo(h, self._nvml.NVML_CLOCK_MEM)),
            )
            ok = False
            for field, getter in getters:
                try:
                    device[field] = getter(handle)
                    ok = ok or field == "w"
                except self._nvml.NVMLError:
                    device[field] = None
            if self._energy_fn is not None and self._energy_supported.get(index):
                try:
                    device["mj"] = float(self._energy_fn(handle))
                except self._not_supported:
                    self._energy_supported[index] = False
                except self._nvml.NVMLError:
                    pass
            if not ok:
                failed += 1
            devices.append(device)
        return None if failed == len(self.indices) else devices

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 — shutdown must never break the summary
            pass


class SmiCollector(BaseCollector):
    """Fallback backend: poll `nvidia-smi --query-gpu=...` per sample."""

    backend = "nvidia-smi"

    _STATIC_FIELDS = (
        "index,name,uuid,power.limit,power.default_limit,driver_version",
        "index,name,uuid,power.limit,driver_version",
        "index,name,uuid",
    )
    _POWER_FIELDS = ("power.draw.average", "power.draw")

    def __init__(self, wanted: set[int] | None, binary: str = "nvidia-smi") -> None:
        self.smi = shutil.which(binary)
        if not self.smi:
            raise CollectorError("nvidia-smi is not on PATH")
        self.power_field = self._probe_power_field()
        self.query = ",".join(
            ["index", self.power_field, "utilization.gpu", "memory.used",
             "temperature.gpu", "clocks.sm", "clocks.mem"]
        )
        rows = self._run_static()
        self.by_index = {int(row["index"]): row for row in rows if row.get("index") is not None}
        indices = sorted(self.by_index)
        if not indices:
            raise CollectorError("nvidia-smi returned no GPU rows")
        self.indices = sorted(wanted) if wanted else indices
        unknown = [i for i in self.indices if i not in self.by_index]
        if unknown:
            raise CollectorError(
                f"GPU index(es) {unknown} do not exist (detected: {indices})"
            )

    def _run(self, args: list[str]) -> str:
        result = subprocess.run(
            [self.smi, *args],
            capture_output=True,
            text=True,
            timeout=SMI_TIMEOUT,
            check=False,
        )
        if result.returncode != 0:
            raise CollectorError(
                f"nvidia-smi exited {result.returncode}: {result.stderr.strip()[:200]}"
            )
        return result.stdout

    def _probe_power_field(self) -> str:
        """Prefer the driver's 1-second average when it exists (smoother integration)."""
        for field in self._POWER_FIELDS:
            try:
                out = self._run(
                    ["--query-gpu=" + field, "--format=csv,noheader,nounits"]
                )
            except CollectorError:
                continue
            values = [_num(cell) for line in out.splitlines() for cell in line.split(",")]
            if values and any(value is not None for value in values):
                return field
        raise CollectorError("nvidia-smi exposes no power.draw field on this driver")

    def _run_static(self) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for fields in self._STATIC_FIELDS:
            try:
                out = self._run(
                    [f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
                )
            except CollectorError as exc:
                last_error = exc
                continue
            names = fields.split(",")
            return [
                dict(zip(names, [cell.strip() for cell in line.split(",")]))
                for line in out.splitlines()
                if line.strip()
            ]
        raise CollectorError(f"nvidia-smi static query failed: {last_error}")

    def probe(self) -> dict[str, Any]:
        gpus = []
        for index in self.indices:
            row = self.by_index[index]
            entry: dict[str, Any] = {
                "key": f"gpu{index}",
                "index": index,
                "name": row.get("name"),
                "uuid": row.get("uuid"),
                "power_limit_w": _num(row.get("power.limit", "")),
            }
            if "power.default_limit" in self.by_index[index]:
                entry["power_default_limit_w"] = _num(
                    self.by_index[index].get("power.default_limit", "")
                )
            gpus.append(entry)
        driver = next((row.get("driver_version") for row in self.by_index.values()), None)
        return {
            "driver": driver,
            "power_field": self.power_field,
            "energy_counter": "unsupported",
            "gpus": gpus,
        }

    def sample(self) -> list[dict[str, Any]] | None:
        try:
            out = self._run(
                [f"--query-gpu={self.query}", "--format=csv,noheader,nounits"]
            )
        except (CollectorError, OSError, subprocess.SubprocessError):
            return None
        rows: dict[int, list[str]] = {}
        for line in out.splitlines():
            cells = [cell.strip() for cell in line.split(",")]
            if cells and cells[0]:
                index = _num(cells[0])
                if index is not None:
                    rows[int(index)] = cells
        if not rows:
            return None
        devices = []
        for index in self.indices:
            cells = rows.get(index)
            device: dict[str, Any] = {"key": f"gpu{index}", "group": "gpu"}
            if cells is None:
                device.update({"w": None, "util": None, "mem_mib": None,
                               "temp": None, "sm_mhz": None, "mem_mhz": None})
            else:
                device.update(
                    {
                        "w": _num(cells[1]),
                        "util": _num(cells[2]),
                        "mem_mib": _num(cells[3]),
                        "temp": _num(cells[4]),
                        "sm_mhz": _num(cells[5]),
                        "mem_mhz": _num(cells[6]),
                    }
                )
            devices.append(device)
        return devices


class RaplCollector(BaseCollector):
    """Best-effort Linux RAPL package energy (CPU side, not wall power)."""

    group = "system"
    backend = "rapl"

    def __init__(self) -> None:
        self.domains: list[dict[str, Any]] = []
        for entry in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
            energy_path = os.path.join(entry, "energy_uj")
            if not os.path.isfile(energy_path):
                continue
            try:
                with open(os.path.join(entry, "name"), encoding="utf-8") as handle:
                    name = handle.read().strip() or os.path.basename(entry)
            except OSError:
                name = os.path.basename(entry)
            try:
                with open(os.path.join(entry, "max_energy_range_uj"), encoding="utf-8") as handle:
                    max_range = float(handle.read().strip())
            except (OSError, ValueError):
                max_range = 2.0**44
            self.domains.append(
                {
                    "key": "sys-" + sanitize_label(name).lower(),
                    "path": energy_path,
                    "max_range_uj": max_range,
                }
            )
        self._previous: dict[str, tuple[float, float]] = {}

    def probe(self) -> dict[str, Any]:
        return {
            "available": bool(self.domains),
            "domains": [domain["key"] for domain in self.domains],
            "note": (
                "RAPL package counters cover the CPU/SoC only, never the whole node"
                if self.domains
                else "no /sys/class/powercap RAPL counters on this kernel"
            ),
        }

    def sample(self) -> list[dict[str, Any]] | None:
        if not self.domains:
            return None
        devices = []
        now = time.monotonic()
        for domain in self.domains:
            try:
                with open(domain["path"], encoding="ascii") as handle:
                    uj = float(handle.read().strip())
            except OSError:
                return None
            key = domain["key"]
            device: dict[str, Any] = {"key": key, "group": "system", "w": None, "mj": uj / 1000.0}
            previous = self._previous.get(key)
            if previous is not None:
                delta = uj - previous[1]
                if delta < 0:  # counter wrap
                    delta += domain["max_range_uj"]
                dt = now - previous[0]
                if dt > 0:
                    device["w"] = delta / dt / 1e6
            self._previous[key] = (now, uj)
            devices.append(device)
        return devices


def default_collector_factory(
    wanted: set[int] | None,
) -> tuple[BaseCollector, BaseCollector | None]:
    """Open the GPU collector (NVML, else nvidia-smi) and optional RAPL."""
    gpu_collector: BaseCollector | None = None
    gpu_error: Exception | None = None
    try:
        gpu_collector = NvmlCollector(wanted)
    except Exception as exc:  # noqa: BLE001 — fall through to the smi backend
        gpu_error = exc
    if gpu_collector is None:
        try:
            gpu_collector = SmiCollector(wanted)
        except Exception as exc:  # noqa: BLE001
            raise CollectorError(
                "no GPU power backend available (pynvml import failed: "
                f"{gpu_error}; nvidia-smi: {exc}). This sampler measures NVIDIA "
                "GPUs — run it on the GB10 nodes, not on a Mac."
            ) from exc
    system_collector: BaseCollector | None = None
    try:
        rapl = RaplCollector()
        if rapl.domains:
            system_collector = rapl
    except Exception:  # noqa: BLE001 — RAPL is strictly optional
        system_collector = None
    return gpu_collector, system_collector


# ---------------------------------------------------------------------------
# run-time plumbing: JSONL stream, live line, markers
# ---------------------------------------------------------------------------


class JsonlWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._handle = path.open("w", encoding="utf-8", buffering=1) if path else None

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is not None:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class LiveLine:
    """One status line refreshed on stderr, only when it is a TTY."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self._last = -10.0
        self._width = 0

    def update(self, t: float, phase: str, devices: list[dict[str, Any]]) -> None:
        if not self.enabled or t - self._last < 1.0:
            return
        self._last = t
        gpus = [d for d in devices if d.get("group") == "gpu"]
        watts = [d.get("w") for d in gpus if isinstance(d.get("w"), (int, float))]
        total = sum(watts) if watts else None
        parts = [f"{d['key']}={_fmt_w(d.get('w'))}" for d in gpus]
        if len(gpus) > 1:
            parts.append(f"total={_fmt_w(total)}")
        utils = [d.get("util") for d in gpus if isinstance(d.get("util"), (int, float))]
        if utils:
            parts.append(f"util={_fmt_v(sum(utils) / len(utils), 0)}%")
        message = f"[bench-power] {t:8.1f}s {phase:11s} " + " ".join(parts)
        padding = max(0, self._width - len(message))
        self._width = len(message)
        sys.stderr.write("\r" + message + " " * padding)
        sys.stderr.flush()

    def clear(self) -> None:
        if self.enabled and self._width:
            sys.stderr.write("\r" + " " * self._width + "\r")
            sys.stderr.flush()
            self._width = 0


def _fmt_w(value: Any) -> str:
    return f"{value:.1f}W" if isinstance(value, (int, float)) else "n/a"


def _fmt_v(value: Any, digits: int = 1) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def read_marker_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """Return markers appended since offset; survives truncation of the file."""
    try:
        size = path.stat().st_size
    except OSError:
        return [], offset
    if size < offset:  # file was truncated or replaced: start over
        offset = 0
    if size == offset:
        return [], offset
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
            new_offset = handle.tell()
    except OSError:
        return [], offset
    lines: list[str] = []
    carry = b""
    if not chunk.endswith(b"\n"):  # keep the trailing partial line for next poll
        cut = chunk.rfind(b"\n")
        if cut == -1:
            return [], offset
        carry, chunk = chunk[cut + 1:], chunk[: cut + 1]
        new_offset = offset + cut + 1
    for raw in (carry + chunk).splitlines():
        label = raw.decode("utf-8", errors="replace").strip()
        if label:
            lines.append(label[:200])
    return lines, new_offset


class Sampler:
    """Drift-free sampling loop shared by both modes."""

    def __init__(
        self,
        *,
        collectors: list[BaseCollector],
        writer: JsonlWriter,
        live: LiveLine,
        interval: float,
        markers_path: Path | None,
    ) -> None:
        self.collectors = collectors
        self.writer = writer
        self.live = live
        self.interval = interval
        self.markers_path = markers_path
        self.markers_offset = 0
        self.drops = 0
        self.late_ticks = 0
        self.stop = threading.Event()
        self.stop_signal: str | None = None
        self.t0_monotonic = time.monotonic()
        self.t0_wall = time.time()

    def t(self) -> float:
        return time.monotonic() - self.t0_monotonic

    def wall(self) -> float:
        return self.t0_wall + self.t()

    def install_signals(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            self.stop_signal = signal.Signals(signum).name
            self.stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, handler)

    def emit(self, record: dict[str, Any]) -> None:
        self.writer.write(record)

    def emit_event(self, event: str, **fields: Any) -> None:
        record = {"kind": "event", "event": event, "t": round(self.t(), 3),
                  "ts": now_iso(self.wall())}
        record.update(fields)
        self.emit(record)

    def one_sample(self, phase: str, sink: list[dict[str, Any]]) -> None:
        devices: list[dict[str, Any]] = []
        for collector in self.collectors:
            try:
                batch = collector.sample()
            except Exception:  # noqa: BLE001 — a flaky read must not kill the run
                batch = None
            if batch is None:
                self.drops += 1
            else:
                devices.extend(batch)
        record = {
            "kind": "sample",
            "t": round(self.t(), 3),
            "ts": now_iso(self.wall()),
            "phase": phase,
            "devices": devices,
        }
        sink.append(record)
        self.emit(record)
        self.live.update(record["t"], phase, devices)
        self._poll_markers(phase, sink)

    def _poll_markers(self, phase: str, sink: list[dict[str, Any]]) -> None:
        if self.markers_path is None:
            return
        lines, self.markers_offset = read_marker_lines(
            self.markers_path, self.markers_offset
        )
        for label in lines:
            record = {
                "kind": "marker",
                "t": round(self.t(), 3),
                "ts": now_iso(self.wall()),
                "phase": phase,
                "label": label,
            }
            sink.append(record)
            self.emit(record)

    def run_until(
        self, phase: str, sink: list[dict[str, Any]], finished: Callable[[], Any]
    ) -> Any:
        """Sample until finished() returns non-None or stop is set.

        finished() is polled once per sample; its return value ends the phase.
        """
        deadline_tick = time.monotonic()
        while True:
            deadline_tick += self.interval
            self.one_sample(phase, sink)
            result = finished()
            if result is not None or self.stop.is_set():
                return result
            now = time.monotonic()
            delay = deadline_tick - now
            if delay < 0:
                self.late_ticks += 1  # sampler starved: re-anchor, never burst
                deadline_tick = now
                delay = 0.0
            self.stop.wait(delay)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def _column(samples: list[dict[str, Any]]) -> dict[str, dict[str, list[Any]]]:
    columns: dict[str, dict[str, list[Any]]] = {}
    for record in samples:
        for device in record["devices"]:
            column = columns.setdefault(device["key"], {"group": device["group"]})
            for field, value in device.items():
                if field in ("key", "group"):
                    continue
                column.setdefault(field, []).append(value)
    return columns


def summarize_phase(
    samples: list[dict[str, Any]], max_gap: float = MAX_GAP
) -> dict[str, Any] | None:
    """Aggregate one phase: per-device stats + energy, plus GPU totals."""
    if len(samples) < 2:
        return None
    start_t, end_t = samples[0]["t"], samples[-1]["t"]
    duration = end_t - start_t
    columns = _column(samples)

    gpu_entries: dict[str, Any] = {}
    system_entries: dict[str, Any] = {}
    total_energy_j = 0.0
    total_energy_known = True
    total_series: list[tuple[float, float]] = []
    max_gap_seen = max(
        (t1["t"] - t0["t"] for t0, t1 in zip(samples, samples[1:])),
        default=0.0,
    )

    for key, column in columns.items():
        watts = [w for w in column.get("w", []) if isinstance(w, (int, float))]
        entry: dict[str, Any] = {"group": column["group"], **series_stats(watts)}

        watt_points, mj_points = _pairs_for(samples, key)
        traj_j, excluded_s, excluded_n = trapezoid_energy(
            [(t, w) for t, w in watt_points if isinstance(w, (int, float))], max_gap
        )
        counted_j = counter_energy([(t, mj) for t, mj in mj_points])
        if counted_j is not None:
            entry["energy_method"] = "counter"
            entry["energy_j"] = round(counted_j, 1)
            entry["trapezoid_energy_j"] = round(traj_j, 1)
        elif watt_points:
            entry["energy_method"] = "trapezoid"
            entry["energy_j"] = round(traj_j, 1)
        else:
            entry["energy_method"] = "none"
            entry["energy_j"] = None
            total_energy_known = False
        if entry["energy_j"] is not None and column["group"] == "gpu":
            total_energy_j += entry["energy_j"]
        entry["energy_wh"] = (
            round(entry["energy_j"] / 3600.0, 4) if entry["energy_j"] is not None else None
        )
        entry["excluded_gap_seconds"] = round(excluded_s, 3)
        entry["excluded_gaps"] = excluded_n
        for nice_field, source in (
            ("mean_util_pct", "util"),
            ("mean_temp_c", "temp"),
            ("max_temp_c", "temp"),
            ("mean_clock_sm_mhz", "sm_mhz"),
            ("mean_mem_mib", "mem_mib"),
        ):
            values = [v for v in column.get(source, []) if isinstance(v, (int, float))]
            if not values:
                continue
            if nice_field == "max_temp_c":
                entry[nice_field] = round(max(values), 1)
            elif nice_field == "mean_util_pct":
                entry[nice_field] = round(_mean(values), 2)
            else:
                entry[nice_field] = round(_mean(values), 1)
        if column["group"] == "gpu":
            gpu_entries[key] = entry
        else:
            system_entries[key] = entry

    # per-sample GPU totals for mean/peak reporting
    for record in samples:
        watts = [
            d["w"]
            for d in record["devices"]
            if d.get("group") == "gpu" and isinstance(d.get("w"), (int, float))
        ]
        if watts:
            total_series.append((record["t"], sum(watts)))
    totals = series_stats([w for _, w in total_series])
    if totals.get("samples"):
        totals["energy_j"] = round(total_energy_j, 1) if total_energy_known else None
        totals["energy_wh"] = (
            round(total_energy_j / 3600.0, 4) if total_energy_known else None
        )

    phase: dict[str, Any] = {
        "start_t": round(start_t, 3),
        "end_t": round(end_t, 3),
        "duration_s": round(duration, 3),
        "max_gap_s": round(max_gap_seen, 3),
        "gpus": gpu_entries,
        "total": totals,
    }
    if system_entries:
        phase["system"] = system_entries
    return phase


def _pairs_for(
    samples: list[dict[str, Any]], key: str
) -> tuple[list[tuple[float, Any]], list[tuple[float, Any]]]:
    """(t, w) and (t, mj) series for one device key, in sampling order."""
    watt_points: list[tuple[float, Any]] = []
    mj_points: list[tuple[float, Any]] = []
    for record in samples:
        for device in record["devices"]:
            if device["key"] == key:
                watt_points.append((record["t"], device.get("w")))
                mj_points.append((record["t"], device.get("mj")))
    return watt_points, mj_points


def idle_baseline(phases: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Duration-weighted idle mean + workload excess, GPU totals only."""
    idle_phases = [p for name, p in phases.items() if name.startswith("idle") and p]
    load = phases.get("load")
    if not idle_phases:
        return None
    weight_sum = sum(p["duration_s"] for p in idle_phases)
    means: list[float] = []
    weights: list[float] = []
    for phase in idle_phases:
        mean = phase.get("total", {}).get("mean_w")
        if mean is not None and phase["duration_s"] > 0:
            means.append(mean)
            weights.append(phase["duration_s"])
    if not means or not weight_sum:
        return None
    baseline = sum(m * w for m, w in zip(means, weights)) / sum(weights)
    result: dict[str, Any] = {
        "source_phases": [name for name, p in phases.items() if name.startswith("idle") and p],
        "mean_w_total": round(baseline, 3),
    }
    if load and load.get("total", {}).get("mean_w") is not None:
        load_mean = load["total"]["mean_w"]
        result["load_mean_w_total"] = load_mean
        # "+ 0.0" normalizes -0.0 away
        result["load_excess_mean_w"] = round(load_mean - baseline, 3) + 0.0
        if load.get("duration_s"):
            result["load_excess_energy_j"] = (
                round((load_mean - baseline) * load["duration_s"], 1) + 0.0
            )
    return result


# ---------------------------------------------------------------------------
# chart (pure-stdlib SVG)
# ---------------------------------------------------------------------------

_SVG_W, _SVG_H = 1040, 330
_M_LEFT, _M_RIGHT, _M_TOP, _M_BOTTOM = 62, 18, 46, 42


def _nice_step(span: float, target: int = 6) -> float:
    if span <= 0:
        return 1.0
    raw = span / target
    power = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if factor * power >= raw:
            return factor * power
    return 10 * power


def _decimate(points: list[tuple[float, float]], limit: int = 1200) -> list[tuple[float, float]]:
    if len(points) <= limit:
        return points
    bucket = math.ceil(len(points) / limit)
    reduced: list[tuple[float, float]] = []
    for start in range(0, len(points), bucket):
        chunk = points[start : start + bucket]
        reduced.append(
            (
                sum(t for t, _ in chunk) / len(chunk),
                sum(w for _, w in chunk) / len(chunk),
            )
        )
    return reduced


def _polyline(points: list[tuple[float, float]], x_of: Callable[[float], float],
              y_of: Callable[[float], float]) -> str:
    return " ".join(
        f"{x_of(t):.1f},{y_of(w):.1f}" for t, w in _decimate(points)
    )


def _nice_ceiling(value: float) -> float:
    """Smallest 1/1.2/1.5/2/2.5/3/4/5/6/8/10 × 10^k at or above value."""
    if value <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(value))
    for factor in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if factor * power >= value:
            return factor * power
    return 10 * power


def render_svg(
    path: Path,
    samples: list[dict[str, Any]],
    markers: list[dict[str, Any]],
    label: str,
    host: str,
) -> bool:
    """Chart GPU power over time as a standalone SVG. Returns False if unplottable."""
    if len(samples) < 2:
        return False
    escape = saxutils.escape
    gpu_keys = sorted(
        {d["key"] for s in samples for d in s["devices"] if d.get("group") == "gpu"}
    )
    series: dict[str, list[tuple[float, float]]] = {key: [] for key in gpu_keys}
    total_points: list[tuple[float, float]] = []
    phase_sums: dict[str, list[float]] = {}
    for record in samples:
        gpu_watts = [
            d["w"]
            for d in record["devices"]
            if d.get("group") == "gpu" and isinstance(d.get("w"), (int, float))
        ]
        for device in record["devices"]:
            if device.get("group") == "gpu" and isinstance(device.get("w"), (int, float)):
                series[device["key"]].append((record["t"], device["w"]))
        if gpu_watts:
            total_points.append((record["t"], sum(gpu_watts)))
            phase_sums.setdefault(record["phase"], []).append(sum(gpu_watts))
    if not total_points:
        return False

    t_max = max(t for t, _ in total_points) or 1.0
    w_max = max(w for _, w in total_points)
    for points in series.values():
        w_max = max(w_max, max((w for _, w in points), default=0.0))
    # Y scale must cover the data: nice ceiling ≥ 1.06 × peak, then snap the
    # top gridline onto the step so the axis always ends on a labelled tick.
    y_top = _nice_ceiling(max(w_max * 1.06, 1.0))
    y_step = _nice_step(y_top, 5)
    y_top = math.ceil((y_top - 1e-9) / y_step) * y_step

    # busiest phase (the load) gets a mean reference line
    mean_phase, mean_w = None, None
    for phase_name, values in phase_sums.items():
        mean = statistics.fmean(values)
        if mean_w is None or mean > mean_w:
            mean_phase, mean_w = phase_name, mean

    plot_w = _SVG_W - _M_LEFT - _M_RIGHT
    plot_h = _SVG_H - _M_TOP - _M_BOTTOM
    plot_right = _SVG_W - _M_RIGHT
    plot_bottom = _M_TOP + plot_h
    x_of = lambda t: _M_LEFT + (t / t_max) * plot_w  # noqa: E731
    y_of = lambda w: _M_TOP + plot_h - (w / y_top) * plot_h  # noqa: E731

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_W}" height="{_SVG_H}" '
        'viewBox="0 0 %d %d" font-family="Menlo,Consolas,monospace">' % (_SVG_W, _SVG_H)
    )
    parts.append(
        f'<defs><clipPath id="plot-clip"><rect x="{_M_LEFT}" y="{_M_TOP}" '
        f'width="{plot_w}" height="{plot_h}"/></clipPath></defs>'
    )
    parts.append(f'<rect width="{_SVG_W}" height="{_SVG_H}" fill="#ffffff"/>')
    parts.append(
        f'<text x="{_M_LEFT}" y="18" font-size="13" fill="#111">'
        f"{escape(sanitize_label(label))} — GPU power over time</text>"
    )
    parts.append(
        f'<text x="{_M_LEFT}" y="33" font-size="10" fill="#666">'
        f'{escape(host)} — {escape(str(samples[0].get("ts", "")))} — '
        f"{len(samples)} samples @ {escape(str(samples[1]['t'] - samples[0]['t']))} s"
        f" — peak {w_max:.1f} W</text>"
    )

    # phase bands (idle shading) drawn behind everything, inside the plot
    bands: list[tuple[str, float, float]] = []
    current_phase = samples[0]["phase"]
    band_start = samples[0]["t"]
    for record in samples[1:] + [{"phase": "__end__", "t": t_max}]:
        if record["phase"] != current_phase:
            bands.append((current_phase, band_start, record["t"]))
            current_phase, band_start = record["phase"], record["t"]

    inner: list[str] = []
    for phase, t0, t1 in bands:
        if not phase.startswith("idle"):
            continue
        inner.append(
            f'<rect x="{x_of(t0):.1f}" y="{_M_TOP}" '
            f'width="{max(x_of(t1) - x_of(t0), 1):.1f}" height="{plot_h}" '
            'fill="#000000" opacity="0.06"/>'
        )

    # grid
    tick = 0.0
    while tick <= y_top + 1e-9:
        y = y_of(tick)
        inner.append(
            f'<line x1="{_M_LEFT}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" '
            'stroke="#e3e3e3" stroke-width="1"/>'
        )
        tick += y_step
    x_step = _nice_step(t_max, 8)
    tick = 0.0
    while tick <= t_max + 1e-9:
        x = x_of(tick)
        inner.append(
            f'<line x1="{x:.1f}" y1="{_M_TOP}" x2="{x:.1f}" y2="{plot_bottom}" '
            'stroke="#f0f0f0" stroke-width="1"/>'
        )
        tick += x_step

    # markers as dashed vertical lines
    for marker in markers[:40]:
        x = x_of(marker["t"])
        inner.append(
            f'<line x1="{x:.1f}" y1="{_M_TOP}" x2="{x:.1f}" y2="{plot_bottom}" '
            'stroke="#888" stroke-dasharray="3,3" stroke-width="1"/>'
        )

    # load-phase mean reference line
    if mean_w is not None and mean_phase is not None:
        inner.append(
            f'<line x1="{_M_LEFT}" y1="{y_of(mean_w):.1f}" x2="{plot_right}" '
            f'y2="{y_of(mean_w):.1f}" stroke="#d0342c" opacity="0.45" '
            'stroke-dasharray="5,4" stroke-width="1"/>'
        )

    palette = ["#3b6fb5", "#2a9d6f", "#c77d2b", "#8a5fb5"]
    if len(gpu_keys) > 1:
        for key, points in sorted(series.items()):
            if len(points) < 2:
                continue
            color = palette[gpu_keys.index(key) % len(palette)]
            inner.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1" '
                f'opacity="0.75" points="{_polyline(points, x_of, y_of)}"/>'
            )
    if len(total_points) >= 2:
        inner.append(
            f'<polyline fill="none" stroke="#d0342c" stroke-width="1.8" '
            f'points="{_polyline(total_points, x_of, y_of)}"/>'
        )
    parts.append(f'<g clip-path="url(#plot-clip)">{"".join(inner)}</g>')

    # axes frame
    parts.append(
        f'<line x1="{_M_LEFT}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" '
        'stroke="#999" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{_M_LEFT}" y1="{_M_TOP}" x2="{_M_LEFT}" y2="{plot_bottom}" '
        'stroke="#999" stroke-width="1"/>'
    )

    # tick labels
    tick = 0.0
    while tick <= y_top + 1e-9:
        y = y_of(tick)
        parts.append(
            f'<text x="{_M_LEFT - 6}" y="{y + 3:.1f}" font-size="9" fill="#555" '
            f'text-anchor="end">{tick:g}</text>'
        )
        tick += y_step
    tick = 0.0
    while tick <= t_max + 1e-9:
        x = x_of(tick)
        parts.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 14}" font-size="9" fill="#555" '
            f'text-anchor="middle">{tick:g}</text>'
        )
        tick += x_step
    parts.append(
        f'<text x="{plot_right}" y="{plot_bottom + 28}" font-size="10" '
        'fill="#444" text-anchor="end">s</text>'
    )
    parts.append(
        f'<text x="10" y="{_M_TOP + 4}" font-size="10" fill="#444">W</text>'
    )

    # phase band names, only when the band is wide enough to hold them
    for phase, t0, t1 in bands:
        if not phase.startswith("idle") or x_of(t1) - x_of(t0) < 48:
            continue
        parts.append(
            f'<text x="{x_of(t0) + 4:.1f}" y="{_M_TOP + 12}" font-size="9" fill="#888">'
            f"{escape(phase)}</text>"
        )

    if mean_w is not None and mean_phase is not None:
        parts.append(
            f'<text x="{plot_right - 4}" y="{y_of(mean_w) - 4:.1f}" font-size="9" '
            'fill="#b03028" text-anchor="end">'
            f"{escape(str(mean_phase))} · moyenne {mean_w:.1f} W</text>"
        )

    # marker labels above the plot, fanned out
    for marker in markers[:40]:
        x = x_of(marker["t"])
        parts.append(
            f'<text x="{x + 2:.1f}" y="{_M_TOP - 4}" font-size="9" fill="#555" '
            f'text-anchor="start">{escape(str(marker.get("label", ""))[:24])}</text>'
        )

    if len(gpu_keys) > 1:
        legend_y = _M_TOP + 12
        for key in gpu_keys:
            color = palette[gpu_keys.index(key) % len(palette)]
            parts.append(
                f'<line x1="{plot_right - 120}" y1="{legend_y - 3}" '
                f'x2="{plot_right - 104}" y2="{legend_y - 3}" stroke="{color}" '
                'stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{plot_right - 100}" y="{legend_y}" font-size="9" '
                f'fill="#444">{escape(key)}</text>'
            )
            legend_y += 13
        parts.append(
            f'<line x1="{plot_right - 120}" y1="{legend_y - 3}" '
            f'x2="{plot_right - 104}" y2="{legend_y - 3}" stroke="#d0342c" '
            'stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{plot_right - 100}" y="{legend_y}" font-size="9" '
            'fill="#444">total</text>'
        )

    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------


def _artifact_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    label = sanitize_label(args.label)
    base = args.out / f"glm53-power-{label}-{stamp}"
    return base.with_suffix(".json"), base.with_suffix(".jsonl"), base.with_suffix(".svg")


def _print_summary_line(summary: dict[str, Any], artifacts: dict[str, Any]) -> None:
    phases = summary["phases"]
    shown = phases.get("load") or phases.get("watch") or next(iter(phases.values()), None)
    duration = summary["timing"]["duration_s"]
    print(f"\nPower summary — label={summary['label']} mode={summary['mode']} "
          f"duration={duration:.1f}s samples={summary['sampling']['samples']}")
    if shown:
        for key, entry in shown.get("gpus", {}).items():
            print(
                f"  {key}: mean {_fmt_v(entry.get('mean_w'))} W  "
                f"peak {_fmt_v(entry.get('max_w'))} W  "
                f"energy {_fmt_v(entry.get('energy_wh'), 3)} Wh  "
                f"({entry.get('energy_method')})"
            )
        total = shown.get("total", {})
        print(
            f"  total: mean {_fmt_v(total.get('mean_w'))} W  "
            f"peak {_fmt_v(total.get('max_w'))} W  "
            f"energy {_fmt_v(total.get('energy_wh'), 3)} Wh"
        )
        if shown.get("system"):
            for key, entry in shown["system"].items():
                print(
                    f"  {key} (RAPL): mean {_fmt_v(entry.get('mean_w'))} W  "
                    f"energy {_fmt_v(entry.get('energy_wh'), 3)} Wh"
                )
    baseline = summary.get("idle_baseline")
    if baseline:
        excess = baseline.get("load_excess_mean_w")
        if excess is not None:
            print(
                f"  idle baseline: {baseline['mean_w_total']:.1f} W → workload excess "
                f"{excess:.1f} W / {baseline.get('load_excess_energy_j', 0) / 3600.0:.3f} Wh"
            )
    for path in (artifacts.get("summary"), artifacts.get("jsonl"), artifacts.get("chart")):
        if path:
            print(f"Wrote {path}")


def _finalize(
    *,
    args: argparse.Namespace,
    sampler: Sampler,
    records: list[dict[str, Any]],
    summary_path: Path,
    jsonl_path: Path | None,
    chart_path: Path | None,
    collector_info: dict[str, Any],
    static_gpus: list[dict[str, Any]],
    mode: str,
    command: list[str] | None,
    child_returncode: int | None,
    timed_out: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples = [r for r in records if r.get("kind") == "sample"]
    markers = [r for r in records if r.get("kind") == "marker"]
    phases: dict[str, dict[str, Any]] = {}
    ordered_phases: list[str] = []
    for record in samples:
        if record["phase"] not in phases:
            phases[record["phase"]] = []
            ordered_phases.append(record["phase"])
        phases[record["phase"]].append(record)
    summarized = {
        name: summarize_phase(bucket)
        for name, bucket in phases.items()
    }
    duration = (samples[-1]["t"] - samples[0]["t"]) if len(samples) >= 2 else 0.0
    summary: dict[str, Any] = {
        "kind": "glm53-power-summary",
        "schema": SCHEMA_VERSION,
        "created_at": now_iso(time.time()),
        "label": sanitize_label(args.label),
        "mode": mode,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "command": command,
        "child_returncode": child_returncode,
        "timed_out": timed_out,
        "interrupted_by": sampler.stop_signal,
        "collector": collector_info,
        "gpus": static_gpus,
        "timing": {
            "started_at": now_iso(sampler.wall() - duration),
            "ended_at": now_iso(sampler.wall()),
            "duration_s": round(duration, 3),
            "interval_s": args.interval,
        },
        "sampling": {
            "samples": len(samples),
            "markers": len(markers),
            "drops": sampler.drops,
            "late_ticks": sampler.late_ticks,
        },
        "phases": {name: summarized[name] for name in ordered_phases if summarized[name]},
        "idle_baseline": idle_baseline(summarized),
        "artifacts": {},
    }
    if extra:
        summary.update(extra)
    if summary_path:
        summary["artifacts"]["summary"] = str(summary_path)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if jsonl_path is not None:
        summary["artifacts"]["jsonl"] = str(jsonl_path)
    if chart_path is not None and render_svg(chart_path, samples, markers, summary["label"],
                                             summary["host"]):
        summary["artifacts"]["chart"] = str(chart_path)
    if summary_path:
        # rewrite once artifact paths are known
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    _print_summary_line(summary, summary["artifacts"])
    return summary


def run_wrap(
    args: argparse.Namespace,
    factory: Callable[[set[int] | None], tuple[BaseCollector, BaseCollector | None]]
    = default_collector_factory,
) -> int:
    summary_path, jsonl_path, chart_path = _artifact_paths(args)
    args.out.mkdir(parents=True, exist_ok=True)
    gpu_collector, system_collector = factory(args.gpu_set)
    collectors = [gpu_collector] + ([system_collector] if system_collector else [])
    collector_info = {
        "gpu_backend": gpu_collector.backend,
        "system_backend": system_collector.backend if system_collector else "none",
    }
    for collector in collectors:
        probe = collector.probe()
        if collector is gpu_collector:
            collector_info.update(probe)
        else:
            collector_info["rapl"] = probe

    writer = JsonlWriter(None if args.no_jsonl else jsonl_path)
    sampler = Sampler(
        collectors=collectors,
        writer=writer,
        live=LiveLine(not args.quiet),
        interval=args.interval,
        markers_path=args.markers,
    )
    sampler.install_signals()
    records: list[dict[str, Any]] = []
    proc: subprocess.Popen[bytes] | None = None
    child_returncode: int | None = None
    timed_out = False
    spawn_error: str | None = None
    exit_code = 0

    try:
        sampler.emit_event(
            "start",
            mode="wrap",
            command=args.command,
            interval_s=args.interval,
            idle_window_s=args.idle_window,
            timeout_s=args.timeout,
            label=sanitize_label(args.label),
        )
        if args.idle_window > 0:
            deadline = time.monotonic() + args.idle_window
            sampler.run_until(
                "idle_before", records,
                lambda: (time.monotonic() >= deadline) or None,
            )

        if sampler.stop.is_set():
            exit_code = 128 + (signal.SIGINT if sampler.stop_signal == "SIGINT" else signal.SIGTERM)
        else:
            try:
                popen_kwargs: dict[str, Any] = {}
                if hasattr(os, "setsid"):
                    popen_kwargs["start_new_session"] = True
                proc = subprocess.Popen(args.command, **popen_kwargs)  # noqa: S603
            except OSError as exc:
                spawn_error = str(exc)
                sampler.emit_event("child_spawn_error", error=spawn_error)
                exit_code = 2
            else:
                timeout_at = time.monotonic() + args.timeout if args.timeout > 0 else None

                def finished() -> tuple[str, int] | None:
                    try:
                        return ("exit", proc.wait(timeout=0))  # type: ignore[union-attr]
                    except subprocess.TimeoutExpired:
                        pass
                    if timeout_at is not None and time.monotonic() >= timeout_at:
                        return ("timeout", -1)
                    return None

                result = sampler.run_until("load", records, finished)
                if result is not None:
                    kind, rc = result
                    child_returncode = rc
                    if kind == "timeout":
                        timed_out = True
                        _terminate_group(proc)
                        child_returncode = proc.wait()
                if sampler.stop.is_set() and proc.poll() is None:
                    _terminate_group(proc)
                    child_returncode = proc.wait()
                if child_returncode is None and proc.poll() is not None:
                    child_returncode = proc.returncode
                sampler.emit_event(
                    "child_exit",
                    returncode=child_returncode,
                    timed_out=timed_out,
                    interrupted=sampler.stop_signal,
                )
                if timed_out:
                    exit_code = 124
                elif sampler.stop.is_set():
                    sig = sampler.stop_signal or "SIGINT"
                    exit_code = 128 + (signal.SIGINT if sig == "SIGINT" else signal.SIGTERM)
                elif child_returncode is not None:
                    exit_code = child_returncode if child_returncode >= 0 else 128 + abs(child_returncode)

                if args.idle_window > 0 and not sampler.stop.is_set():
                    deadline = time.monotonic() + args.idle_window
                    sampler.run_until(
                        "idle_after", records,
                        lambda: (time.monotonic() >= deadline) or None,
                    )
    finally:
        if proc is not None and proc.poll() is None:
            _terminate_group(proc)
            proc.wait()
        sampler.emit_event("stop", exit_code=exit_code, duration_s=round(sampler.t(), 3))
        sampler.live.clear()
        writer.close()
        for collector in collectors:
            collector.close()

    _finalize(
        args=args,
        sampler=sampler,
        records=records,
        summary_path=summary_path,
        jsonl_path=None if args.no_jsonl else jsonl_path,
        chart_path=None if args.no_chart else chart_path,
        collector_info=collector_info,
        static_gpus=collector_info.get("gpus", []),
        mode="wrap",
        command=args.command,
        child_returncode=child_returncode,
        timed_out=timed_out,
        extra={"spawn_error": spawn_error} if spawn_error else None,
    )
    return exit_code


def _terminate_group(proc: subprocess.Popen[Any]) -> None:
    """SIGTERM the child's process group, escalate to SIGKILL after a grace."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    deadline = time.monotonic() + KILL_GRACE
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.05)
    if proc.poll() is None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass


def run_watch(
    args: argparse.Namespace,
    factory: Callable[[set[int] | None], tuple[BaseCollector, BaseCollector | None]]
    = default_collector_factory,
) -> int:
    summary_path, jsonl_path, chart_path = _artifact_paths(args)
    args.out.mkdir(parents=True, exist_ok=True)
    gpu_collector, system_collector = factory(args.gpu_set)
    collectors = [gpu_collector] + ([system_collector] if system_collector else [])
    collector_info = {
        "gpu_backend": gpu_collector.backend,
        "system_backend": system_collector.backend if system_collector else "none",
    }
    for collector in collectors:
        probe = collector.probe()
        if collector is gpu_collector:
            collector_info.update(probe)
        else:
            collector_info["rapl"] = probe

    writer = JsonlWriter(None if args.no_jsonl else jsonl_path)
    sampler = Sampler(
        collectors=collectors,
        writer=writer,
        live=LiveLine(not args.quiet),
        interval=args.interval,
        markers_path=args.markers,
    )
    sampler.install_signals()
    records: list[dict[str, Any]] = []
    try:
        sampler.emit_event("start", mode="watch", interval_s=args.interval,
                           duration_s=args.duration, label=sanitize_label(args.label))
        deadline = time.monotonic() + args.duration if args.duration > 0 else None
        sampler.run_until(
            "watch", records,
            lambda: ((time.monotonic() >= deadline) or None)
            if deadline is not None else None,
        )
    finally:
        sampler.emit_event("stop", duration_s=round(sampler.t(), 3))
        sampler.live.clear()
        writer.close()
        for collector in collectors:
            collector.close()

    _finalize(
        args=args,
        sampler=sampler,
        records=records,
        summary_path=summary_path,
        jsonl_path=None if args.no_jsonl else jsonl_path,
        chart_path=None if args.no_chart else chart_path,
        collector_info=collector_info,
        static_gpus=collector_info.get("gpus", []),
        mode="watch",
        command=None,
        child_returncode=None,
        timed_out=False,
    )
    return 0


def cmd_list_gpus(
    args: argparse.Namespace,
    factory: Callable[[set[int] | None], tuple[BaseCollector, BaseCollector | None]]
    = default_collector_factory,
) -> int:
    gpu_collector, system_collector = factory(args.gpu_set)
    info = {"gpu_backend": gpu_collector.backend, **gpu_collector.probe()}
    if system_collector:
        info["rapl"] = system_collector.probe()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    gpu_collector.close()
    if system_collector:
        system_collector.close()
    return 0


def cmd_rechart(jsonl_path: Path) -> int:
    """Regenerate the SVG chart from an existing JSONL stream (no GPU needed)."""
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") in ("sample", "marker"):
                records.append(record)
    samples = [r for r in records if r["kind"] == "sample"]
    markers = [r for r in records if r["kind"] == "marker"]
    label = re.sub(r"-\d{8}-\d{6}$", "", re.sub(r"^glm53-power-", "", jsonl_path.stem))
    host = socket.gethostname()
    summary_path = jsonl_path.with_suffix(".json")
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            label = summary.get("label", label)
            host = summary.get("host", host)
        except (OSError, json.JSONDecodeError):
            pass
    out_path = jsonl_path.with_suffix(".svg")
    if not render_svg(out_path, samples, markers, label, host):
        print(f"rechart: not enough plottable samples in {jsonl_path}", file=sys.stderr)
        return 2
    print(f"Recharted {out_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample GPU/system power and integrate energy around a command "
        "or while watching. Run this on the GB10 nodes.",
        epilog=(
            "examples:\n"
            "  ./bench-power.py --list-gpus\n"
            "  ./bench-power.py --watch --label idle\n"
            "  ./bench-power.py --label c6 -- python3 bench-glm53.py --runs 3 --concurrency 6\n"
            "  ./bench-power.py --label long-200k --idle-window 15 -- python3 bench-long-context.py "
            "--target-tokens 200000 --cold\n"
            "  ./bench-power.py --watch --markers /tmp/marks.txt   # echo 'prompt 1' >> /tmp/marks.txt\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--watch", action="store_true",
                        help="sample until Ctrl+C instead of wrapping a command given after --")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="sampling period in seconds (default: 0.5)")
    parser.add_argument("--idle-window", type=float, default=10.0,
                        help="seconds of idle baseline sampled before and after the "
                        "wrapped command, 0 disables (default: 10)")
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="hard stop for the wrapped command in seconds, 0 = none")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="auto-stop --watch after this many seconds, 0 = until Ctrl+C")
    parser.add_argument("--label", default="run",
                        help="short tag used in artifact filenames (default: run)")
    parser.add_argument("--out", type=Path, default=Path("results"),
                        help="output directory (default: results/)")
    parser.add_argument("--gpu", default=None,
                        help="comma-separated GPU indices to sample (default: all)")
    parser.add_argument("--markers", type=Path, default=None,
                        help="poll this file; every line appended to it becomes a "
                        "timestamped marker in the JSONL stream (per-prompt attribution)")
    parser.add_argument("--no-jsonl", action="store_true",
                        help="skip the raw JSONL sample stream (summary and chart only)")
    parser.add_argument("--no-chart", action="store_true", help="skip the SVG chart")
    parser.add_argument("--quiet", action="store_true", help="no live stderr status line")
    parser.add_argument("--list-gpus", action="store_true",
                        help="print detected GPUs/system counters as JSON and exit")
    parser.add_argument("--rechart", type=Path, default=None, metavar="JSONL",
                        help="regenerate the SVG chart from an existing .jsonl "
                        "stream and exit (no GPU required)")
    parser.set_defaults(gpu_set=None, command=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    command: list[str] | None = None
    if "--" in raw:
        split = raw.index("--")
        command = raw[split + 1:]
        raw = raw[:split]
    parser = build_parser()
    args = parser.parse_args(raw)

    if not MIN_INTERVAL <= args.interval <= MAX_INTERVAL:
        parser.error(f"--interval must be between {MIN_INTERVAL} and {MAX_INTERVAL} s")
    if args.idle_window < 0 or args.idle_window > 600:
        parser.error("--idle-window must be between 0 and 600 s")
    if args.timeout < 0:
        parser.error("--timeout must be >= 0")
    if args.duration < 0:
        parser.error("--duration must be >= 0")
    if args.gpu is not None:
        try:
            args.gpu_set = {int(item) for item in args.gpu.split(",") if item.strip()}
        except ValueError:
            parser.error("--gpu expects comma-separated integers, e.g. --gpu 0,1")
        if not args.gpu_set:
            parser.error("--gpu is empty")
    else:
        args.gpu_set = None
    if args.markers is not None and not args.markers.exists():
        try:
            args.markers.parent.mkdir(parents=True, exist_ok=True)
            args.markers.touch()
        except OSError as exc:
            parser.error(f"--markers file cannot be created: {exc}")

    if args.rechart is not None:
        return cmd_rechart(args.rechart)
    if args.list_gpus:
        return cmd_list_gpus(args)
    if command is not None:
        if args.watch:
            parser.error("--watch and a command after -- are mutually exclusive")
        if not command:
            parser.error("no command given after --")
        args.command = command
        return run_wrap(args)
    if args.watch:
        return run_watch(args)
    parser.error('nothing to measure: pass a command after "--" or use --watch')
    return 2  # unreachable, parser.error exits


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, CollectorError, json.JSONDecodeError) as exc:
        print(f"power sampler error: {exc}", file=sys.stderr)
        raise SystemExit(2)
