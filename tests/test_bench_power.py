from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bench_power", ROOT / "bench-power.py")
assert SPEC is not None and SPEC.loader is not None
POWER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POWER)


class FakeGpuCollector(POWER.BaseCollector):
    """Deterministic stand-in: 120 W flat, energy counter at 120 kJ/s."""

    backend = "fake"

    def __init__(self) -> None:
        self.t0 = time.monotonic()

    def probe(self) -> dict:
        return {
            "driver": "fake",
            "gpus": [
                {
                    "key": "gpu0",
                    "index": 0,
                    "name": "Fake GPU",
                    "uuid": "fake-uuid-0",
                    "power_limit_w": 240.0,
                }
            ],
        }

    def sample(self) -> list[dict]:
        elapsed = time.monotonic() - self.t0
        return [
            {
                "key": "gpu0",
                "group": "gpu",
                "w": 120.0,
                "mj": 120_000.0 * elapsed,
                "util": 97,
                "temp": 61,
                "mem_mib": 119_000,
                "sm_mhz": 1500,
                "mem_mhz": 2000,
            }
        ]


def fake_factory(wanted):
    return FakeGpuCollector(), None


def make_args(tmp: Path, extra: list[str], command: list[str] | None = None):
    args = POWER.build_parser().parse_args(
        ["--out", str(tmp), "--interval", "0.05", "--idle-window", "0",
         "--quiet", *extra]
    )
    args.command = command
    return args


class PureFunctionTests(unittest.TestCase):
    def test_percentile_is_nearest_rank(self) -> None:
        self.assertEqual(POWER.percentile(list(range(1, 11)), 0.95), 10)
        self.assertIsNone(POWER.percentile([], 0.95))

    def test_series_stats(self) -> None:
        stats = POWER.series_stats([100.0, 120.0])
        self.assertEqual(stats["samples"], 2)
        self.assertEqual(stats["mean_w"], 110.0)
        self.assertEqual(stats["max_w"], 120.0)

    def test_trapezoid_excludes_wide_gaps(self) -> None:
        energy, excluded, segments = POWER.trapezoid_energy(
            [(0.0, 100.0), (1.0, 100.0), (5.0, 100.0), (6.0, 100.0)], max_gap=2.0
        )
        self.assertAlmostEqual(energy, 200.0)
        self.assertAlmostEqual(excluded, 4.0)
        self.assertEqual(segments, 1)

    def test_counter_energy(self) -> None:
        self.assertAlmostEqual(
            POWER.counter_energy([(0.0, 0.0), (1.0, 5_000.0), (2.0, 12_000.0)]), 12.0
        )
        self.assertIsNone(POWER.counter_energy([(0.0, 10.0), (1.0, 5.0)]))
        self.assertIsNone(POWER.counter_energy([(0.0, 5.0)]))

    def test_sanitize_label(self) -> None:
        self.assertEqual(POWER.sanitize_label("c6/DFlash 2"), "c6_DFlash_2")
        self.assertEqual(POWER.sanitize_label("///"), "run")
        self.assertEqual(len(POWER.sanitize_label("x" * 100)), 64)

    def test_smi_value_parsing(self) -> None:
        self.assertAlmostEqual(POWER._num("123.45"), 123.45)
        self.assertAlmostEqual(POWER._num("95"), 95.0)
        for junk in ("[N/A]", "N/A", "[Not Supported]", "not supported", ""):
            self.assertIsNone(POWER._num(junk))

    def test_marker_lines_incremental(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "marks.txt"
            path.write_text("", encoding="utf-8")
            self.assertEqual(POWER.read_marker_lines(path, 0), ([], 0))
            with path.open("a", encoding="utf-8") as handle:
                handle.write("prompt alpha\n")
            lines, offset = POWER.read_marker_lines(path, 0)
            self.assertEqual(lines, ["prompt alpha"])
            with path.open("a", encoding="utf-8") as handle:
                handle.write("prompt beta")
            self.assertEqual(POWER.read_marker_lines(path, offset), ([], offset))
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            lines, offset = POWER.read_marker_lines(path, offset)
            self.assertEqual(lines, ["prompt beta"])
            path.write_text("fresh\n", encoding="utf-8")  # truncated file: start over
            lines, offset = POWER.read_marker_lines(path, offset)
            self.assertEqual(lines, ["fresh"])

    def test_summarize_phase_aggregates_and_counters(self) -> None:
        samples = [
            {
                "kind": "sample",
                "t": t,
                "ts": "2026-08-31T00:00:00.000Z",
                "phase": "load",
                "devices": [
                    {"key": "gpu0", "group": "gpu", "w": 100.0, "mj": 100_000.0 * t,
                     "util": 50, "temp": 50},
                    {"key": "gpu1", "group": "gpu", "w": 60.0, "mj": 60_000.0 * t,
                     "util": 40, "temp": 55},
                ],
            }
            for t in (0.0, 1.0, 2.0, 3.0)
        ]
        phase = POWER.summarize_phase(samples)
        assert phase is not None
        self.assertAlmostEqual(phase["gpus"]["gpu0"]["energy_j"], 300.0, delta=0.5)
        self.assertEqual(phase["gpus"]["gpu0"]["energy_method"], "counter")
        self.assertAlmostEqual(phase["total"]["mean_w"], 160.0, delta=0.01)
        self.assertAlmostEqual(phase["total"]["energy_j"], 480.0, delta=1.0)
        self.assertAlmostEqual(phase["gpus"]["gpu0"]["mean_util_pct"], 50.0)
        self.assertEqual(phase["max_gap_s"], 1.0)


class WrapRunTests(unittest.TestCase):
    def test_wrap_run_produces_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = make_args(
                out,
                ["--label", "wrap-test"],
                command=[sys.executable, "-c", "import time; time.sleep(0.5)"],
            )
            exit_code = POWER.run_wrap(args, factory=fake_factory)
            self.assertEqual(exit_code, 0)

            summary_path = next(out.glob("glm53-power-wrap-test-*.json"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["kind"], "glm53-power-summary")
            self.assertEqual(summary["mode"], "wrap")
            self.assertEqual(summary["child_returncode"], 0)
            self.assertIsNone(summary["idle_baseline"])
            self.assertEqual(summary["collector"]["gpu_backend"], "fake")

            load = summary["phases"]["load"]
            self.assertGreaterEqual(load["duration_s"], 0.3)
            self.assertAlmostEqual(load["total"]["mean_w"], 120.0, delta=10.0)
            gpu = load["gpus"]["gpu0"]
            self.assertEqual(gpu["energy_method"], "counter")
            self.assertGreater(gpu["energy_j"], 15.0)
            self.assertLess(gpu["energy_j"], 120.0)

            jsonl_path = Path(summary["artifacts"]["jsonl"])
            lines = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(lines[0]["kind"], "event")
            self.assertEqual(lines[0]["event"], "start")
            self.assertTrue(any(record["kind"] == "sample" for record in lines))
            self.assertTrue(
                any(record["kind"] == "event" and record["event"] == "stop"
                    for record in lines)
            )

            chart_path = Path(summary["artifacts"]["chart"])
            self.assertTrue(chart_path.read_text(encoding="utf-8").startswith("<svg"))

    def test_wrap_spawn_failure_still_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = make_args(
                out,
                ["--label", "spawn-fail"],
                command=["/nonexistent/command/xyz"],
            )
            exit_code = POWER.run_wrap(args, factory=fake_factory)
            self.assertEqual(exit_code, 2)
            summary_path = next(out.glob("glm53-power-spawn-fail-*.json"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["spawn_error"])

    def test_wrap_with_idle_window_builds_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = POWER.build_parser().parse_args(
                ["--out", str(out), "--interval", "0.05", "--idle-window", "0.2",
                 "--quiet", "--label", "idle-test"]
            )
            args.command = [sys.executable, "-c", "import time; time.sleep(0.3)"]
            exit_code = POWER.run_wrap(args, factory=fake_factory)
            self.assertEqual(exit_code, 0)
            summary = json.loads(
                next(out.glob("glm53-power-idle-test-*.json")).read_text(encoding="utf-8")
            )
            self.assertIn("idle_before", summary["phases"])
            self.assertIn("idle_after", summary["phases"])
            baseline = summary["idle_baseline"]
            self.assertAlmostEqual(baseline["mean_w_total"], 120.0, delta=10.0)
            self.assertIn("load_excess_mean_w", baseline)

    def test_watch_run_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = make_args(out, ["--watch", "--duration", "0.3", "--label", "watch-test"])
            exit_code = POWER.run_watch(args, factory=fake_factory)
            self.assertEqual(exit_code, 0)
            summary = json.loads(
                next(out.glob("glm53-power-watch-test-*.json")).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["mode"], "watch")
            self.assertIn("watch", summary["phases"])

    def test_list_gpus_reports_probe(self) -> None:
        args = POWER.build_parser().parse_args(["--list-gpus"])
        with tempfile.TemporaryDirectory() as tmp:
            args.out = Path(tmp)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = POWER.cmd_list_gpus(args, factory=fake_factory)
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["gpu_backend"], "fake")
        self.assertEqual(payload["gpus"][0]["name"], "Fake GPU")

    def test_render_svg_minimum(self) -> None:
        samples = [
            {
                "kind": "sample",
                "t": float(t) / 10,
                "ts": "2026-08-31T00:00:00.000Z",
                "phase": "load",
                "devices": [{"key": "gpu0", "group": "gpu", "w": 100.0 + t}],
            }
            for t in range(20)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chart.svg"
            self.assertTrue(POWER.render_svg(path, samples, [], "test", "host"))
            content = path.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("<svg"))
        self.assertIn("<polyline", content)


if __name__ == "__main__":
    unittest.main()
