#!/usr/bin/env python3
"""Local launcher wiring for verification-only adaptive-K.

The Reederey wiring test targets its own `local/prod-start.sh`; this one pins
the `vllm-exl3/start.sh` contract.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
ENV = ROOT / ".env.example"

KNOBS = (
    "GLM53_ADAPTIVE_K",
    "GLM53_ADAPTIVE_K_CAPTURE",
    "GLM53_ADAPTIVE_K_SET",
    "GLM53_ADAPTIVE_K_ALPHA",
    "GLM53_ADAPTIVE_K_MARGIN",
    "GLM53_ADAPTIVE_K_MIN_STEPS",
    "GLM53_ADAPTIVE_K_SATURATE",
    "GLM53_ADAPTIVE_K_HIST",
    "GLM53_ADAPTIVE_K_BATCH_MARGIN",
    "GLM53_ADAPTIVE_K_POS_TRIM",
    "GLM53_ADAPTIVE_K_CAPTURE_MAX_BATCH",
)


def test_overlay_wired_six_ways() -> None:
    text = START.read_text()
    for var, name in (
        ("ALIGN_FLOOR_PATCH_HOST", "patch_align_floor.py"),
        ("ADAPTIVE_K_PATCH_HOST", "patch_adaptive_k.py"),
    ):
        assert f'{var}="${{{var}:-$SCRIPT_DIR/overlay/{name}}}"' in text
        assert f'[ -f "${var}" ] || die "${var} missing"' in text
        assert f"${{WORKER_SSH}}:/tmp/{name}" in text
        assert f"-v '/tmp/{name}:/opt/glm53/{name}:ro'" in text
        assert f':/opt/glm53/{name}:ro"' in text
        assert text.count(f"python3 /opt/glm53/{name}") == 2


def test_order() -> None:
    text = START.read_text()
    chain = [
        "patch_scheduler_decode_floor.py",
        "patch_align_floor.py",
        "patch_adaptive_k.py",
        "patch_glm5_drafter_group.py",
    ]
    positions = [text.index(f"python3 /opt/glm53/{n}") for n in chain]
    assert positions == sorted(positions)


def test_env_defaults_and_forwarding() -> None:
    text = START.read_text()
    assert 'GLM53_ALIGN_FLOOR="${GLM53_ALIGN_FLOOR:-1}"' in text
    assert 'GLM53_ADAPTIVE_K="${GLM53_ADAPTIVE_K:-off}"' in text
    assert 'GLM53_ADAPTIVE_K_CAPTURE="${GLM53_ADAPTIVE_K_CAPTURE:-0}"' in text
    for knob in KNOBS:
        assert f'-e "{knob}=' in text, knob


def test_capture_lists() -> None:
    text = START.read_text()
    assert "1 2 4 8 16 24 32" in text
    assert "3 5" not in text  # computed, not a literal pair
    assert "GLM53_ADAPTIVE_K_CAPTURE_MAX_BATCH" in text
    assert "MAX_NUM_SEQS" in text
    assert "cudagraph-capture-sizes" in text


def test_dotenv_names_the_knobs() -> None:
    text = ENV.read_text()
    assert "GLM53_ADAPTIVE_K=off" in text
    assert "GLM53_ADAPTIVE_K_CAPTURE=0" in text
    assert "GLM53_ADAPTIVE_K_SET=2,4,7" in text
    assert "GLM53_ADAPTIVE_K_BATCH_MARGIN=0" in text
    assert "GLM53_ADAPTIVE_K_POS_TRIM=0" in text
    assert "GLM53_ALIGN_FLOOR=1" in text


def test_overlay_refuses_draft_hook() -> None:
    text = (ROOT / "overlay" / "patch_adaptive_k.py").read_text()
    assert "num_spec_tokens_to_schedule = _GLM53_ADAPTIVE_K.batch_k(" not in text
    assert "_GLM53_ADAPTIVE_K.apply(" in text
    assert "GLM53_ADAPTIVE_K_PATCH" in text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("adaptive-K local wiring OK")
