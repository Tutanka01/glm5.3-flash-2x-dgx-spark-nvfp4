from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bench_long_context", ROOT / "bench-long-context.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class LongContextSafetyTests(unittest.TestCase):
    def test_loopback_detection(self) -> None:
        self.assertTrue(BENCH.is_loopback_url("http://127.0.0.1:8888/v1"))
        self.assertTrue(BENCH.is_loopback_url("http://localhost:8888/v1"))
        self.assertTrue(BENCH.is_loopback_url("http://[::1]:8888/v1"))
        self.assertFalse(BENCH.is_loopback_url("http://10.10.10.1:8888/v1"))

    def test_256k_reliability_profile_passes(self) -> None:
        runtime = {
            "MAX_MODEL_LEN": "262144",
            "MAX_NUM_SEQS": "1",
            "MAX_NUM_BATCHED_TOKENS": "1024",
            "MEM_FRACTION_STATIC": "0.88",
            "DISABLE_CUDA_GRAPH": "1",
            "MTP_NUM_TOKENS": "0",
        }
        self.assertEqual(BENCH.long_context_safety_issues(runtime, 240000), [])

    def test_256k_mtp_profile_is_refused(self) -> None:
        runtime = {
            "MAX_MODEL_LEN": "262144",
            "MAX_NUM_SEQS": "1",
            "MAX_NUM_BATCHED_TOKENS": "4096",
            "MEM_FRACTION_STATIC": "0.90",
            "DISABLE_CUDA_GRAPH": "0",
            "MTP_NUM_TOKENS": "5",
        }
        issues = BENCH.long_context_safety_issues(runtime, 240000)
        self.assertIn("CUDA graphs are enabled", issues)
        self.assertIn("MTP_NUM_TOKENS=5, expected 0", issues)
        self.assertTrue(any("safe ceiling is 2048" in issue for issue in issues))

    def test_128k_and_below_does_not_require_special_profile(self) -> None:
        self.assertEqual(BENCH.long_context_safety_issues({}, 120000), [])

    def test_unsafe_runtime_is_fail_closed(self) -> None:
        runtime = {
            "GLM53_PROFILE_NAME": "256k-mtp",
            "MAX_MODEL_LEN": "262144",
            "MAX_NUM_SEQS": "1",
            "MAX_NUM_BATCHED_TOKENS": "4096",
            "MEM_FRACTION_STATIC": "0.90",
            "DISABLE_CUDA_GRAPH": "0",
            "MTP_NUM_TOKENS": "5",
        }
        with mock.patch.object(
            BENCH, "inspect_local_runtime", return_value=(runtime, "mock")
        ):
            with self.assertRaisesRegex(BENCH.BenchmarkError, "running profile=256k-mtp"):
                BENCH.enforce_long_context_safety(
                    "http://127.0.0.1:8888/v1", 240000, False
                )


if __name__ == "__main__":
    unittest.main()
