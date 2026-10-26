#!/usr/bin/env python3
"""CPU-only tests for the new vLLM-EXL3 A/B and cache-session tooling.

New file in this lane.  No network, GPU, docker or running server is touched.
The three scripts are loaded by path because their filenames contain hyphens.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
VLLM_DIR = HERE.parent
SCRIPTS = VLLM_DIR / "scripts"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.12 dataclasses look the defining module up in sys.modules.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


bench_ab = _load("bench_ab_under_test", SCRIPTS / "bench-ab.py")
watchdog = _load("prefill_watchdog_under_test", SCRIPTS / "prefill-watchdog.py")
cache_sessions = _load(
    "bench_cache_sessions_under_test", HERE / "bench_cache_sessions.py"
)


def _arm(name: str, *bench_records: dict) -> dict:
    return {
        "name": name,
        "role": "baseline" if name == "baseline" else "candidate",
        "ok": True,
        "benches": list(bench_records),
        "warnings": [],
        "error": None,
    }


def _bench(name: str, samples: dict[str, list[float]]) -> dict:
    return {
        "name": name,
        "ok": True,
        "metrics": {"samples": samples, "summary": {}, "notes": []},
        "artifact": f"/tmp/{name}.json",
    }


class TestArmParsing(unittest.TestCase):
    def test_plain_baseline(self) -> None:
        arm = bench_ab.parse_arm("baseline")
        self.assertEqual(arm.name, "baseline")
        self.assertEqual(arm.env, {})

    def test_overrides_preserve_equals_in_value(self) -> None:
        arm = bench_ab.parse_arm(
            "candidate:MAX_NUM_SEQS=6,GLM53_MIXED_PREFILL_CHUNK=448,EXTRA=a=b"
        )
        self.assertEqual(
            arm.env,
            {
                "MAX_NUM_SEQS": "6",
                "GLM53_MIXED_PREFILL_CHUNK": "448",
                "EXTRA": "a=b",
            },
        )

    def test_first_arm_is_baseline_and_names_are_unique(self) -> None:
        arms = bench_ab.parse_arms(["baseline", "candidate:A=1"])
        self.assertEqual([arm.name for arm in arms], ["baseline", "candidate"])
        with self.assertRaises(bench_ab.ArmSpecError):
            bench_ab.parse_arms(["baseline", "baseline:A=1"])

    def test_invalid_specs(self) -> None:
        for spec in (
            "",
            ":A=1",
            "bad name:A=1",
            "candidate:",
            "candidate:NO_EQUALS",
            "candidate:1BAD=2",
            "candidate:A=1,A=2",
        ):
            with self.subTest(spec=spec), self.assertRaises(bench_ab.ArmSpecError):
                bench_ab.parse_arm(spec)

    def test_empty_value_is_allowed(self) -> None:
        arm = bench_ab.parse_arm("candidate:GLM53_INDEXER_WORKSPACE=")
        self.assertEqual(arm.env["GLM53_INDEXER_WORKSPACE"], "")


class TestEnvMerge(unittest.TestCase):
    def test_merge_does_not_mutate_base(self) -> None:
        base = {"PATH": "/bin", "MAX_NUM_SEQS": "4"}
        merged = bench_ab.merge_env(base, {"MAX_NUM_SEQS": "6", "NEW": "x"})
        self.assertEqual(base, {"PATH": "/bin", "MAX_NUM_SEQS": "4"})
        self.assertEqual(merged["MAX_NUM_SEQS"], "6")
        self.assertEqual(merged["NEW"], "x")
        self.assertEqual(merged["PATH"], "/bin")

    def test_redaction_and_effective_filter(self) -> None:
        env = {
            "PATH": "/bin",
            "VLLM_API_KEY": "super-secret",
            "MAX_NUM_SEQS": "6",
            "GLM53_MIXED_PREFILL_CHUNK": "448",
        }
        redacted = bench_ab.redact_env(env)
        self.assertEqual(redacted["VLLM_API_KEY"], "<redacted>")
        self.assertEqual(redacted["MAX_NUM_SEQS"], "6")
        effective = bench_ab.filter_effective_env(env)
        self.assertNotIn("PATH", effective)
        self.assertEqual(effective["VLLM_API_KEY"], "<redacted>")
        self.assertEqual(effective["GLM53_MIXED_PREFILL_CHUNK"], "448")

    def test_redact_text(self) -> None:
        text = "token super-secret appeared in a log"
        redacted = bench_ab.redact_text(text, ["super-secret"])
        self.assertNotIn("super-secret", redacted)
        self.assertIn("<redacted>", redacted)


class TestStatisticsAndComparison(unittest.TestCase):
    def test_numeric_stats_p99_is_nearest_rank(self) -> None:
        stats = bench_ab.numeric_stats([100.0, 102.0, 98.0])
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["median"], 100.0)
        self.assertEqual(stats["p99"], 102.0)
        self.assertEqual(stats["min"], 98.0)
        self.assertEqual(stats["max"], 102.0)

    def test_build_comparison_delta_and_p99(self) -> None:
        baseline = _arm(
            "baseline",
            _bench(
                "decode_structured",
                {"decode_tok_s": [100.0, 102.0, 98.0], "ttft_s": [1.0, 1.1, 0.9]},
            ),
        )
        candidate = _arm(
            "candidate",
            _bench(
                "decode_structured",
                {"decode_tok_s": [110.0, 112.0, 108.0], "ttft_s": [0.8, 0.9, 0.7]},
            ),
        )
        comparison = bench_ab.build_comparison([baseline, candidate])
        self.assertTrue(comparison["baseline_ok"])
        rows = {row["metric"]: row for row in comparison["rows"]}
        decode_row = rows["decode_structured.decode_tok_s"]
        self.assertEqual(decode_row["baseline"]["median"], 100.0)
        self.assertEqual(decode_row["arms"]["candidate"]["p99"], 112.0)
        self.assertAlmostEqual(decode_row["deltas"]["candidate"], 10.0)

    def test_markdown_has_one_row_per_metric_and_escapes(self) -> None:
        baseline = _arm(
            "baseline",
            _bench("decode_structured", {"decode_tok_s": [100.0, 102.0, 98.0]}),
        )
        candidate = _arm(
            "candidate",
            _bench("decode_structured", {"decode_tok_s": [110.0, 112.0, 108.0]}),
        )
        report = {
            "stamp": "20260101-000000",
            "created_at": "2026-01-01T00:00:00+00:00",
            "base_url": "http://127.0.0.1:8888",
            "workdir": "/tmp/vllm-exl3",
            "git": {"commit": "abc", "branch": "main", "dirty": False},
            "protocol": {"runs": 3, "restart": True, "warmup": True},
            "warnings": ["tree is dirty | check it"],
            "arms": [baseline, candidate],
            "comparison": bench_ab.build_comparison([baseline, candidate]),
        }
        markdown = bench_ab.render_markdown(report)
        self.assertIn("decode_structured.decode_tok_s (n=3)", markdown)
        self.assertIn("+10.0%", markdown)
        self.assertIn("112", markdown)
        self.assertIn("dirty", markdown)
        # Every table row has the same number of separators as its header.
        header_pipes = None
        for line in markdown.splitlines():
            if line.startswith("| metric |"):
                header_pipes = line.count("|")
                self.assertEqual(header_pipes, 8)  # metric + dir + 2 arms x 3
            elif line.startswith("| decode_structured.decode_tok_s"):
                self.assertEqual(line.count("|"), header_pipes)

    def test_failed_baseline_suppresses_deltas(self) -> None:
        failed = _arm("baseline")
        failed["ok"] = False
        candidate = _arm(
            "candidate",
            _bench("short_chat", {"ttft_s": [0.2, 0.3]}),
        )
        comparison = bench_ab.build_comparison([failed, candidate])
        self.assertFalse(comparison["baseline_ok"])
        self.assertTrue(all(not row["deltas"] for row in comparison["rows"]))


class TestArtifactFailClosed(unittest.TestCase):
    def test_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "missing.json"
            with self.assertRaises(bench_ab.BenchArtifactError):
                bench_ab.load_artifact(path)

    def test_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "broken.json"
            path.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(bench_ab.BenchArtifactError) as caught:
                bench_ab.load_artifact(path)
            self.assertIn("malformed JSON", str(caught.exception))

    def test_malformed_decode_shape(self) -> None:
        with self.assertRaises(bench_ab.BenchArtifactError):
            bench_ab._extract_decode({"runs": []})
        with self.assertRaises(bench_ab.BenchArtifactError):
            bench_ab._extract_decode({"runs": [{"tok_s": None}]})


class TestRunArmFailClosed(unittest.TestCase):
    def test_creates_artifact_dir_and_fails_on_health(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        args = SimpleNamespace(
            no_restart=True,
            health_timeout=1.0,
            health_interval=0.05,
            base_url="http://127.0.0.1:1",
            timeout=1.0,
            restart_timeout=1.0,
            settle=0.0,
            model="mock",
            runs=1,
            sessions=1,
            ctx_tokens=400,
            cache_mode="sessions",
        )
        arm = bench_ab.Arm(name="baseline", env={}, spec="baseline")
        with tempfile.TemporaryDirectory() as raw:
            out_dir = Path(raw) / "artifacts" / "nested"
            with mock.patch.object(
                bench_ab,
                "wait_for_health",
                return_value={"ok": False, "code": 0, "attempts": 1},
            ):
                result = bench_ab.run_arm(
                    0,
                    arm,
                    args,
                    ["short_chat"],
                    workdir=VLLM_DIR,
                    tests_dir=VLLM_DIR / "tests",
                    out_dir=out_dir,
                    api_key="",
                )
            self.assertTrue(out_dir.is_dir())
            self.assertFalse(result["ok"])
            self.assertIn("/health", result["error"] or "")


class TestBenchCommands(unittest.TestCase):
    def _context(self, base_url: str) -> bench_ab.BenchContext:
        return bench_ab.BenchContext(
            name="prefix_cache",
            base_url=base_url,
            model="mock",
            runs=2,
            timeout=30.0,
            artifact_path=Path("/tmp/prefix.json"),
            tests_dir=VLLM_DIR / "tests",
            workdir=VLLM_DIR,
            sessions=2,
            ctx_tokens=400,
            cache_mode="sessions",
            api_key="",
        )

    def test_prefix_cache_gets_v1_base(self) -> None:
        for base_url in ("http://127.0.0.1:8888", "http://127.0.0.1:8888/v1"):
            argv = bench_ab._prefix_build(self._context(base_url))
            self.assertEqual(
                argv[argv.index("--base-url") + 1],
                "http://127.0.0.1:8888/v1",
            )


class TestBootWarnings(unittest.TestCase):
    def _arms(self, boots: list[dict]) -> list[dict]:
        return [
            {"name": f"arm{i}", "boot": boot}
            for i, boot in enumerate(boots)
        ]

    def test_same_boot_when_restart_required(self) -> None:
        arms = self._arms(
            [
                {"id": "process-start:1", "source": "metrics-process-start"},
                {"id": "process-start:1", "source": "metrics-process-start"},
            ]
        )
        warnings = bench_ab.boot_warnings(arms, restart_required=True)
        self.assertTrue(any("same boot identity" in warning for warning in warnings))

    def test_distinct_boots_are_clean(self) -> None:
        arms = self._arms(
            [
                {"id": "process-start:1", "source": "metrics-process-start"},
                {"id": "process-start:2", "source": "metrics-process-start"},
            ]
        )
        warnings = bench_ab.boot_warnings(arms, restart_required=True)
        self.assertEqual(warnings, [])

    def test_missing_and_host_only_boot(self) -> None:
        arms = self._arms(
            [
                {"id": None, "source": "unavailable"},
                {"id": "host-boot:abc", "source": "host-boot"},
            ]
        )
        warnings = bench_ab.boot_warnings(arms, restart_required=True)
        self.assertTrue(any("unavailable" in warning for warning in warnings))
        self.assertTrue(any("host boot id" in warning for warning in warnings))

    def test_no_restart_warns(self) -> None:
        arms = self._arms(
            [
                {"id": "process-start:1", "source": "metrics-process-start"},
                {"id": "process-start:2", "source": "metrics-process-start"},
            ]
        )
        warnings = bench_ab.boot_warnings(arms, restart_required=False)
        self.assertTrue(any("--no-restart" in warning for warning in warnings))
        self.assertTrue(any("changed between arms" in warning for warning in warnings))


class TestCacheSessionsPure(unittest.TestCase):
    def test_hit_fraction(self) -> None:
        self.assertEqual(cache_sessions.hit_fraction(1000, 900), 0.9)
        self.assertIsNone(cache_sessions.hit_fraction(1000, None))
        self.assertIsNone(cache_sessions.hit_fraction(None, 900))
        self.assertIsNone(cache_sessions.hit_fraction(0, 0))

    def test_parse_usage_accepts_integral_floats(self) -> None:
        prompt, completion, cached = cache_sessions.parse_usage(
            {
                "prompt_tokens": 100.0,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 90.0},
            }
        )
        self.assertEqual((prompt, completion, cached), (100, 4, 90))

    def test_verdict_thresholds(self) -> None:
        self.assertEqual(cache_sessions.classify_verdict([0.95, 0.92]), "healthy")
        self.assertEqual(cache_sessions.classify_verdict([0.01, 0.02]), "collapsed")
        self.assertEqual(cache_sessions.classify_verdict([0.40, 0.50]), "partial")
        self.assertEqual(cache_sessions.classify_verdict([]), "unknown")
        self.assertEqual(cache_sessions.classify_verdict([None]), "unknown")

    def test_prompt_plan_shapes(self) -> None:
        normal = cache_sessions.build_prompt_plan(
            cache_sessions.PlanConfig(sessions=2, ctx_tokens=4000, salt="salt-1")
        )
        self.assertEqual(len(normal), 2)
        self.assertNotEqual(
            normal[0].messages[0]["content"], normal[1].messages[0]["content"]
        )
        self.assertTrue(normal[0].messages[0]["content"].startswith("Run salt salt-1."))

        shared = cache_sessions.build_prompt_plan(
            cache_sessions.PlanConfig(
                sessions=2, ctx_tokens=4000, shared_prefix=True, salt="salt-1"
            )
        )
        self.assertEqual(
            shared[0].messages[0]["content"], shared[1].messages[0]["content"]
        )
        self.assertNotEqual(
            shared[0].messages[1]["content"], shared[1].messages[1]["content"]
        )

    def test_sub_page_clamps_below_page(self) -> None:
        plan = cache_sessions.build_prompt_plan(
            cache_sessions.PlanConfig(
                sessions=1, ctx_tokens=60000, sub_page=True, page_tokens=3584
            )
        )
        self.assertLessEqual(plan[0].target_tokens, 3000)
        self.assertLess(plan[0].target_tokens, 3584)

    def test_sse_parser_uses_cached_tokens_and_first_content(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"content":"A"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"B"}}]}\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":100,'
            b'"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":90}}}\n',
            b"data: [DONE]\n",
        ]

        class FakeResponse:
            status = 200

            def __iter__(self):
                return iter(lines)

        result = cache_sessions._parse_sse(FakeResponse(), 0.0)
        self.assertTrue(result.ok)
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.cached_tokens, 90)
        self.assertEqual(result.ttft_s is not None, True)
        self.assertEqual(cache_sessions.hit_fraction(
            result.prompt_tokens, result.cached_tokens
        ), 0.9)

    def test_dry_run_is_network_free(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cache_sessions.main(
                ["--dry-run", "--sessions", "2", "--ctx-tokens", "4000", "--rounds", "2"]
            )
        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("round=1 phase=cold", text)
        self.assertIn("round=2 phase=warm", text)
        self.assertIn("no requests sent", text)


class TestWatchdog(unittest.TestCase):
    def _metrics_file(self, path: Path, prompt: float, ttft_sum: float, count: float) -> None:
        path.write_text(
            "vllm:prompt_tokens_total{{model_name=\"m\"}} {}\n"
            "vllm:time_to_first_token_seconds_sum{{model_name=\"m\"}} {}\n"
            "vllm:time_to_first_token_seconds_count{{model_name=\"m\"}} {}\n".format(
                prompt, ttft_sum, count
            ),
            encoding="utf-8",
        )

    def test_series_detection(self) -> None:
        degraded = watchdog.analyze_series([1000, 1000, 1000, 200, 200], window=2)
        self.assertTrue(degraded.degraded)
        self.assertEqual(degraded.status, "degraded")
        self.assertAlmostEqual(degraded.baseline or 0, 1000.0)
        self.assertAlmostEqual(degraded.recent or 0, 200.0)

        stable = watchdog.analyze_series([1000, 990, 1010, 1005], window=2)
        self.assertFalse(stable.degraded)
        self.assertEqual(stable.status, "ok")

        insufficient = watchdog.analyze_series([1000], window=2)
        self.assertEqual(insufficient.status, "insufficient")

    def test_metrics_fixture_degraded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            files = []
            # Cumulative counters: 1000 prompt tokens and 1 TTFT second per
            # interval for the baseline, then 1000 tokens per 5 seconds.
            rows = [(1000, 1, 1), (2000, 2, 2), (3000, 3, 3), (4000, 8, 4), (5000, 13, 5)]
            for index, (prompt, ttft, count) in enumerate(rows):
                path = tmp / f"m{index}.txt"
                self._metrics_file(path, prompt, ttft, count)
                files.append(path)
            snapshots = watchdog.collect_metrics_files(files)
            record = watchdog.analyze_metrics_snapshots(
                snapshots, window=2, threshold=0.30, baseline_window=20
            )
        self.assertEqual(record["status"], "degraded")
        self.assertEqual(record["values"], [1000.0, 1000.0, 200.0, 200.0])
        self.assertTrue(record["degraded"])

    def test_log_fixture_and_cli_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            log = tmp / "head.log"
            log.write_text(
                "\n".join(
                    ["Avg prompt throughput: 1000.0 tokens/s"] * 4
                    + ["Avg prompt throughput: 200.0 tokens/s"] * 3
                )
                + "\n",
                encoding="utf-8",
            )
            record = watchdog.analyze_log_file(
                log, window=3, threshold=0.30, baseline_window=20, tail_lines=100
            )
            self.assertEqual(record["status"], "degraded")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = watchdog.main(
                    ["--log-file", str(log), "--window", "3", "--threshold", "0.3", "--json"]
                )
            self.assertEqual(code, 2)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "DEGRADED")
            self.assertEqual(payload["source"], "log")

            stable = tmp / "stable.log"
            stable.write_text(
                "\n".join(["Avg prompt throughput: 1000.0 tokens/s"] * 7) + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = watchdog.main(
                    ["--log-file", str(stable), "--window", "3", "--threshold", "0.3", "--json"]
                )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "OK")

            missing = tmp / "missing.log"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = watchdog.main(["--log-file", str(missing), "--json"])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.getvalue())["status"], "ERROR")

    def test_zero_throughput_is_idle_not_degraded(self) -> None:
        values, zeros = watchdog.parse_log_values(
            [
                "Avg prompt throughput: 0.0 tokens/s",
                "prefill=934.2 tok/s",
                "prefill_tok_s: 900.5",
            ]
        )
        self.assertEqual(zeros, 1)
        self.assertEqual(values, [934.2, 900.5])

    def test_metrics_without_histogram_uses_wall_time(self) -> None:
        snapshots = [
            watchdog.MetricsSnapshot(source="fixture", ts=float(index), prompt_tokens=value)
            for index, value in enumerate((0.0, 1000.0, 2000.0, 2200.0, 2400.0))
        ]
        observations = watchdog.observations_from_snapshots(snapshots)
        values = [observation.tok_s for observation in observations]
        self.assertEqual(values, [1000.0, 1000.0, 200.0, 200.0])
        analysis = watchdog.analyze_series(values, window=2)
        self.assertTrue(analysis.degraded)

    def test_metrics_histogram_parse(self) -> None:
        snapshot = watchdog.parse_metrics_text(
            "vllm:prompt_tokens_total 12345\n"
            "vllm:time_to_first_token_seconds_sum 12.5\n"
            "vllm:time_to_first_token_seconds_count 10\n"
            "process_start_time_seconds 1000\n",
            source="fixture",
            ts=1.0,
        )
        self.assertEqual(snapshot.prompt_tokens, 12345.0)
        self.assertEqual(snapshot.ttft_sum, 12.5)
        self.assertEqual(snapshot.ttft_count, 10.0)
        self.assertEqual(snapshot.process_start, 1000.0)


class TestBenchAbCli(unittest.TestCase):
    def test_dry_run_is_network_and_write_free(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "ab.json"
            markdown = Path(raw) / "ab.md"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = bench_ab.main(
                    [
                        "--dry-run",
                        "--bench",
                        "short_chat",
                        "--no-restart",
                        "--arm",
                        "baseline",
                        "--arm",
                        "candidate:MAX_NUM_SEQS=6",
                        "--out",
                        str(out),
                        "--markdown",
                        str(markdown),
                    ]
                )
            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("arm 2/2 candidate", text)
            self.assertIn("MAX_NUM_SEQS", text)
            self.assertIn("restart=skipped", text)
            self.assertFalse(out.exists())
            self.assertFalse(markdown.exists())

    def test_invalid_arm_exits_nonzero(self) -> None:
        output = io.StringIO()
        error = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = bench_ab.main(["--dry-run", "--arm", "bad name!"])
        self.assertEqual(code, 2)
        self.assertIn("invalid arm name", error.getvalue())

    def test_list_benches(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = bench_ab.main(["--list-benches"])
        self.assertEqual(code, 0)
        for name in (
            "decode_structured",
            "decode_prose",
            "prefix_cache",
            "cache_sessions",
            "short_chat",
        ):
            self.assertIn(name, output.getvalue())


if __name__ == "__main__":
    unittest.main()
