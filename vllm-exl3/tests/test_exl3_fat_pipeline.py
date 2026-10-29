#!/usr/bin/env python3
"""Static checks for the opt-in 3-stage cp.async fat-GEMM.

The kernel itself is GPU-only. These tests keep the source contract, the
build wiring and the Python selector reviewable on CPU.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "overlay"
CU = OVERLAY / "exl3_fat_gemm_pipe.cu"
CUH = OVERLAY / "exl3_fat_gemm_pipe.cuh"
PATCH = OVERLAY / "patch_exl3_fat_kernel.py"
EXL3 = OVERLAY / "exl3.py"
START = ROOT / "start.sh"
DOCKERFILE = ROOT / "Dockerfile"


def test_pipe_source_has_three_stages_and_local_wrappers() -> None:
    text = CU.read_text()
    assert "constexpr int FAT_STAGES = 3;" in text
    assert "glm53_cp_async16" in text
    assert "cp.async.cg.shared.global" in text
    assert "glm53_cp_async_commit" in text
    assert "glm53_cp_async_wait" in text
    assert "exl3_fat_gemm_pipe_kernel" in text
    # The OOB A row must be zero-filled synchronously; a predicated cp.async
    # is the Blackwell miscompile the reference explicitly avoids.
    assert "*a_dst = int4{};" in text
    assert "cp_async_pred(" not in text


def test_pipe_public_symbols_are_namespaced() -> None:
    header = CUH.read_text()
    assert "void exl3_fat_gemm_pipe(" in header
    assert "void exl3_fat_gemm_pipe_scatter(" in header
    source = CU.read_text()
    assert source.count("void exl3_fat_gemm_pipe(") == 1
    assert source.count("void exl3_fat_gemm_pipe_scatter(") == 1
    # The stock symbols must stay available as the fallback.
    assert "void exl3_fat_gemm(" in (OVERLAY / "exl3_fat_gemm.cuh").read_text()
    assert "void exl3_fat_gemm_scatter(" in (
        OVERLAY / "exl3_fat_gemm.cuh"
    ).read_text()


def test_build_script_installs_and_binds_both_pairs() -> None:
    text = PATCH.read_text()
    for name in (
        "exl3_fat_gemm.cu",
        "exl3_fat_gemm.cuh",
        "exl3_fat_gemm_pipe.cu",
        "exl3_fat_gemm_pipe.cuh",
    ):
        assert name in text
    assert 'm.def("exl3_fat_gemm"' in text
    assert 'm.def("exl3_fat_gemm_pipe"' in text
    assert 'm.def("exl3_fat_gemm_pipe_scatter"' in text


def test_dockerfile_copies_pipe_sources_and_asserts_symbols() -> None:
    text = DOCKERFILE.read_text()
    assert "exl3_fat_gemm_pipe.cu" in text
    assert "exl3_fat_gemm_pipe.cuh" in text
    assert "exl3_fat_gemm_pipe" in text
    assert "fat_gemm_pipe=yes" in text


def test_python_selector_default_off_and_fail_closed() -> None:
    text = EXL3.read_text()
    assert "def fat_pipeline_enabled()" in text
    assert 'os.environ.get("EXL3_FAT_PIPELINE", "0")' in text
    assert "exl3_fat_gemm_pipe" in text
    assert "EXL3_FAT_PIPELINE=1 requires" in text
    # Both call sites must use the selected symbols, not the hard-coded ones.
    assert "gemm(" in text
    assert "gemm_scatter(" in text
    assert "ext.exl3_fat_gemm_scatter(" not in text


def test_launcher_default_and_forwarding() -> None:
    text = START.read_text()
    assert 'EXL3_FAT_PIPELINE="${EXL3_FAT_PIPELINE:-0}"' in text
    assert "-e EXL3_FAT_PIPELINE=" in text
    assert "EXL3_FAT_PIPELINE" in text.split("serve_env", 1)[1]
    env = (ROOT / ".env.example").read_text()
    assert "EXL3_FAT_PIPELINE=0" in env


def test_no_legacy_cp_async_names_leak() -> None:
    """The local wrappers must not depend on exllamav3's ptx.cuh helper ABI."""
    text = CU.read_text()
    # cp_async( / cp_async_fence() / cp_async_wait<> are the old helper names.
    assert not re.search(r"(?<!glm53_)cp_async\(", text)
    assert "cp_async_fence()" not in text
    assert not re.search(r"(?<!glm53_)cp_async_wait<", text)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("exl3 fat-gemm pipe static checks OK")


def test_gpu_gate_script_is_present_and_pins_both_pairs() -> None:
    gate = ROOT / "tests" / "bench_fat_gemm_pipe.py"
    assert gate.is_file()
    text = gate.read_text()
    for sym in (
        "exl3_fat_gemm",
        "exl3_fat_gemm_pipe",
        "exl3_fat_gemm_scatter",
        "exl3_fat_gemm_pipe_scatter",
    ):
        assert sym in text
    assert "torch.equal" in text
    assert "CUDA unavailable" in text
